"""Coordinate multiple :class:`WorkerAgent` instances in parallel."""

import asyncio
import math
import itertools
import logging
from typing import Any, Dict, List

from .worker_agent import WorkerAgent
from .openai_responses import ResponsesClient


class AgentPool:
    """Simple orchestrator that fans out tasks to worker agents."""
    def __init__(
        self,
        schema: Dict[str, Any],
        phase_cfg: Dict[str, Any],
        validator_cls,
        writer,
        client: ResponsesClient,
    ) -> None:
        """Create the pool.

        Args:
            schema: Full database schema mapping.
            phase_cfg: Phase configuration controlling generation.
            validator_cls: Validator class used by workers.
            writer: Result writer instance used by workers.
            client: Shared :class:`ResponsesClient`.
        """

        self.schema = schema
        self.cfg = phase_cfg
        self.validator_cls = validator_cls
        self.writer = writer
        self.client = client
        self.seen: set[tuple[str, str]] = set()
        self.lock = asyncio.Lock()
        self.log = logging.getLogger(__name__)

    def _schema_chunks(self) -> List[Dict[str, Any]]:
        """Return list of schema subsets split for each worker."""
        n = int(self.cfg.get("parallelism", 1))
        table_names = list(self.schema.keys())
        chunks = [table_names[i::n] for i in range(n)]
        return [{t: self.schema[t] for t in c} for c in chunks]

    async def _run_worker(self, batch_size: int, worker_id: int, schema: Dict[str, Any]) -> int:
        """Launch a single :class:`WorkerAgent` and merge its output."""

        self.log.info("Worker %d starting batch size %d", worker_id, batch_size)
        agent = WorkerAgent(
            schema,
            self.cfg,
            self.validator_cls,
            worker_id,
            self.client,
        )
        pairs = await agent.generate(batch_size)
        async with self.lock:
            before = len(self.seen)
            for p in pairs:
                self.seen.add((p["question"], p["answer"]))
            delta = len(self.seen) - before
        self.log.info("Worker %d produced %d new pairs", worker_id, delta)
        return delta

    async def generate(self) -> List[Dict[str, str]]:
        """Generate unique Q&A pairs across multiple workers."""

        goal = int(self.cfg.get("count", 1))
        n_workers = int(self.cfg.get("parallelism", 1))
        k = math.ceil(goal / n_workers)
        schema_chunks = self._schema_chunks()
        attempts = 0
        self.log.info(
            "Starting generation: goal=%d parallelism=%d batch_size=%d",
            goal,
            n_workers,
            k,
        )
        while len(self.seen) < goal and attempts < self.cfg.get("max_attempts", 6):
            jobs = [self._run_worker(k, i, schema_chunks[i]) for i in range(n_workers)]
            await asyncio.gather(*jobs)
            attempts += 1
            self.log.info("Attempt %d complete, total pairs=%d", attempts, len(self.seen))
        self.log.info("Generation finished with %d pairs", len(self.seen))
        return [{"question": q, "answer": a} for q, a in itertools.islice(self.seen, goal)]
