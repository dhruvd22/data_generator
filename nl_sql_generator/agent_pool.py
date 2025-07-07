import asyncio
import math
import itertools
from typing import Any, Dict, List

from .worker_agent import WorkerAgent
from .openai_responses import ResponsesClient


class AgentPool:
    def __init__(
        self,
        schema: Dict[str, Any],
        phase_cfg: Dict[str, Any],
        validator_cls,
        writer,
        client: ResponsesClient,
    ) -> None:
        self.schema = schema
        self.cfg = phase_cfg
        self.validator_cls = validator_cls
        self.writer = writer
        self.client = client
        self.seen: set[tuple[str, str]] = set()
        self.lock = asyncio.Lock()

    async def _run_worker(self, batch_size: int, worker_id: int) -> int:
        agent = WorkerAgent(
            self.schema,
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
        return delta

    async def generate(self) -> List[Dict[str, str]]:
        goal = int(self.cfg.get("count", 1))
        k = math.ceil(goal / int(self.cfg.get("parallelism", 1)))
        attempts = 0
        while len(self.seen) < goal and attempts < self.cfg.get("max_attempts", 6):
            jobs = [self._run_worker(k, i) for i in range(int(self.cfg.get("parallelism", 1)))]
            await asyncio.gather(*jobs)
            attempts += 1
        return [{"question": q, "answer": a} for q, a in itertools.islice(self.seen, goal)]
