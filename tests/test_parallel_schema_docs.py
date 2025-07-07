import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.agent_pool import AgentPool
import nl_sql_generator.agent_pool as agent_pool


class DummyWorker:
    def __init__(self, schema, cfg, validator_cls, wid):
        self.wid = wid

    async def generate(self, k):
        return [{"question": f"Q{self.wid}-{i}", "answer": f"A{i}"} for i in range(k)]


def test_pool_dedup(monkeypatch):
    monkeypatch.setattr(agent_pool, "WorkerAgent", DummyWorker)
    cfg = {"count": 5, "parallelism": 2}
    pool = AgentPool({}, cfg, lambda: None, None)
    pairs = asyncio.run(pool.generate())
    assert len(pairs) == 5
    assert len({(p["question"], p["answer"]) for p in pairs}) == 5
