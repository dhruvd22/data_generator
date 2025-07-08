"""Orchestrate JoinWorker agents in parallel."""

import asyncio
import math
import random
import itertools
import logging
from typing import Any, Dict, List

from .join_worker import JoinWorker
from .openai_responses import ResponsesClient
from .autonomous_job import _clean_sql


class JoinPool:
    def __init__(self, schema: Dict[str, Any], phase_cfg: Dict[str, Any], validator_cls, writer, critic, client: ResponsesClient) -> None:
        self.schema = schema
        self.cfg = phase_cfg
        self.validator_cls = validator_cls
        self.writer = writer
        self.critic = critic
        self.client = client
        self.seen: set[tuple[str, str]] = set()
        self.lock = asyncio.Lock()
        self.log = logging.getLogger(__name__)

    def _schema_chunks(self) -> List[Dict[str, Any]]:
        n = int(self.cfg.get("parallelism", 1))
        min_joins = int(self.cfg.get("min_joins", 2))
        table_names = list(self.schema.keys())
        if not table_names:
            return [self.schema]
        chunks: List[Dict[str, Any]] = []
        for _ in range(min(n, len(table_names))):
            k = min(len(table_names), max(min_joins, 2))
            chosen = random.sample(table_names, k=k)
            chunks.append({t: self.schema[t] for t in chosen})
        return chunks

    async def _run_worker(self, batch_size: int, worker_id: int, schema_subset: Dict[str, Any]) -> int:
        cfg = dict(self.cfg)
        if cfg.get("use_sample_rows"):
            n_rows = int(cfg.get("n_rows", 5))
            sample_rows = {}
            for t in schema_subset:
                try:
                    sample_rows[t] = self.writer.fetch(f"SELECT * FROM {t} LIMIT {n_rows}", n_rows)
                except Exception as err:
                    self.log.warning("Worker %d failed fetching rows for %s: %s", worker_id, t, err)
            if sample_rows:
                cfg["sample_rows"] = sample_rows
        self.log.info("Worker %d starting batch size %d", worker_id, batch_size)
        agent = JoinWorker(schema_subset, cfg, self.validator_cls, self.critic, self.writer, worker_id, self.client)
        pairs = await agent.generate(batch_size)
        async with self.lock:
            before = len(self.seen)
            for p in pairs:
                self.seen.add((p["question"], p["sql"]))
            delta = len(self.seen) - before
        self.log.info("Worker %d produced %d new pairs", worker_id, delta)
        return delta

    async def generate(self) -> List[Dict[str, str]]:
        goal = int(self.cfg.get("count", 1))
        schema_chunks = self._schema_chunks()
        n_workers = len(schema_chunks)
        k = math.ceil(goal / max(n_workers, 1))
        attempts = 0
        self.log.info(
            "Starting join generation: goal=%d parallelism=%d batch_size=%d",
            goal,
            n_workers,
            k,
        )
        while len(self.seen) < goal and attempts < self.cfg.get("max_attempts", 6):
            jobs = [self._run_worker(k, i, schema_chunks[i]) for i in range(n_workers)]
            await asyncio.gather(*jobs)
            attempts += 1
            self.log.info("Attempt %d complete, total pairs=%d", attempts, len(self.seen))
        self.log.info("Join generation finished with %d pairs", len(self.seen))
        return [
            {"question": q, "sql": _clean_sql(s)}
            for q, s in itertools.islice(self.seen, goal)
        ]
