"""Orchestrate :class:`JoinWorker` agents in parallel.

``JoinPool`` splits the database schema into smaller chunks and spawns
workers for each subset.  It keeps track of all produced question/SQL
pairs so duplicates are filtered across workers.
"""

import asyncio
import itertools
import logging
import json
import os
from typing import Any, Dict, List

from .prompt_builder import load_template_messages
from .join_worker import JoinWorker
from .openai_responses import ResponsesClient
from .autonomous_job import _clean_sql

DEFAULT_PARALLELISM = 10


class JoinPool:
    """High level coordinator for join-based question generation."""

    def __init__(
        self,
        schema: Dict[str, Any],
        phase_cfg: Dict[str, Any],
        validator_cls,
        writer,
        critic,
        client: ResponsesClient,
        pool_size: int | None = None,
    ) -> None:
        """Create a pool instance.

        Args:
            schema: Mapping of table metadata for the entire database.
            phase_cfg: Phase configuration controlling generation options.
            validator_cls: Callable returning a validator used by workers.
            writer: :class:`ResultWriter` instance for executing SQL.
            critic: :class:`Critic` used to fix invalid SQL.
            client: Shared :class:`ResponsesClient` for OpenAI calls.
        """
        self.schema = schema
        self.cfg = phase_cfg
        self.validator_cls = validator_cls
        self.writer = writer
        self.critic = critic
        self.client = client
        if pool_size is None:
            pool_size = int(os.getenv("DB_COCURRENT_SESSION", "50"))
        self.pool_size = pool_size
        self.seen: set[tuple[str, str]] = set()
        self.lock = asyncio.Lock()
        self.log = logging.getLogger(__name__)

    async def _schema_chunks(self) -> List[Dict[str, Any]]:
        """Return table subsets for each worker using GPT suggestions."""

        n = int(self.cfg.get("parallelism", DEFAULT_PARALLELISM))
        min_joins = int(self.cfg.get("min_joins", 2))
        extra = {"count": n, "min_joins": min_joins}
        self.log.info("Requesting %d joinable table sets", n)
        try:
            messages = load_template_messages(
                "join_set_template.txt", self.schema, "", extra
            )
            text = await self.client.acomplete(messages)
        except Exception as err:  # pragma: no cover - network failures
            self.log.warning("Failed requesting table sets: %s", err)
            text = ""

        sets: List[List[str]] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip("`")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                tbls = obj.get("tables")
            else:
                tbls = obj
            if not isinstance(tbls, list):
                continue
            tbls = [t for t in tbls if t in self.schema]
            if len(tbls) >= min_joins:
                sets.append(tbls)

        if not sets:
            # fallback to using all tables if GPT gave nothing
            sets = [list(self.schema.keys())]

        chunks = [{t: self.schema[t] for t in tbls} for tbls in sets[:n]]
        self.log.info("GPT suggested table sets: %s", sets[:n])
        return chunks

    async def _run_worker(
        self, batch_size: int, worker_id: int, schema_subset: Dict[str, Any]
    ) -> int:
        """Launch a single worker on ``schema_subset``.

        Each worker generates ``batch_size`` pairs using a trimmed schema view.
        Newly produced pairs are merged into ``self.seen`` under a lock so that
        concurrent workers don't create duplicates.
        """
        cfg = dict(self.cfg)
        if cfg.get("use_sample_rows"):
            n_rows = int(cfg.get("n_rows", 2))
            sample_rows = {}
            for t in schema_subset:
                try:
                    sample_rows[t] = self.writer.fetch(
                        f"SELECT * FROM {t} LIMIT {n_rows}", n_rows
                    )
                except Exception as err:
                    self.log.warning(
                        "Worker %d failed fetching rows for %s: %s",
                        worker_id,
                        t,
                        err,
                    )
            if sample_rows:
                cfg["sample_rows"] = sample_rows
                self.log.info(
                    "Worker %d loaded sample rows for %s",
                    worker_id,
                    list(sample_rows),
                )
        self.log.info(
            "Worker %d starting batch size %d with tables %s",
            worker_id,
            batch_size,
            list(schema_subset),
        )
        agent = JoinWorker(
            schema_subset,
            cfg,
            self.validator_cls,
            self.critic,
            self.writer,
            worker_id,
            self.client,
            self.pool_size,
        )
        pairs = await agent.generate(batch_size)
        async with self.lock:
            before = len(self.seen)
            for p in pairs:
                self.seen.add((p["question"], p["sql"]))
            delta = len(self.seen) - before
        self.log.info("Worker %d produced %d new pairs", worker_id, delta)
        return delta

    async def generate(self) -> List[Dict[str, str]]:
        """Run all workers until the desired number of pairs is reached."""

        per_worker = int(self.cfg.get("count", 1))
        schema_chunks = await self._schema_chunks()
        n_workers = len(schema_chunks)
        self.log.info("Spawning %d workers", n_workers)
        attempts = 0
        produced = [0] * n_workers
        self.log.info(
            "Starting join generation: per_worker=%d parallelism=%d",
            per_worker,
            n_workers,
        )

        while any(p < per_worker for p in produced) and attempts < self.cfg.get(
            "max_attempts", 6
        ):
            jobs = []
            for i in range(n_workers):
                remaining = per_worker - produced[i]
                if remaining > 0:
                    jobs.append(self._run_worker(remaining, i, schema_chunks[i]))
                else:
                    jobs.append(asyncio.sleep(0, result=0))

            deltas = await asyncio.gather(*jobs)
            for i, d in enumerate(deltas):
                produced[i] += d

            attempts += 1
            self.log.info(
                "Attempt %d complete, per-worker totals=%s", attempts, produced
            )
            overall = sum(produced)
            self.log.info(
                "Overall progress: %d/%d pairs", overall, per_worker * n_workers
            )

        total_goal = per_worker * n_workers
        self.log.info("Join generation finished with %d pairs", len(self.seen))
        return [
            {"question": q, "sql": _clean_sql(s)}
            for q, s in itertools.islice(self.seen, total_goal)
        ]
