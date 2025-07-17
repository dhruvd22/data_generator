import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.utils import limit_pool_size
from nl_sql_generator.autonomous_job import AutonomousJob


class DummyValidator:
    def check(self, sql):
        return True, ""


class DummyWriter:
    def fetch(self, sql, n_rows=5):
        return []


class DummyClient:
    def set_parallelism(self, p):
        pass

    async def acomplete(self, *a, **kw):
        return ""


class DummyPool:
    def __init__(self, *args, **kwargs):
        DummyPool.size = args[-1] if args else kwargs.get("pool_size", 0)

    async def generate(self):
        return []


def test_limit_pool_size(monkeypatch):
    monkeypatch.setenv("MAX_DB_CONCURRENT_LIMIT_ALL", "100")
    monkeypatch.setenv("DB_COCURRENT_SESSION", "20")
    assert limit_pool_size(5) == 20  # 100//5=20 -> min(20,20)
    assert limit_pool_size(10) == 10  # 100//10=10
    assert limit_pool_size(25) == 4  # 100//25=4


def test_join_pool_pool_size_limited(monkeypatch):
    monkeypatch.setenv("MAX_DB_CONCURRENT_LIMIT_ALL", "100")
    monkeypatch.setenv("DB_COCURRENT_SESSION", "50")
    monkeypatch.setattr("nl_sql_generator.join_pool.JoinPool", DummyPool)
    job = AutonomousJob(
        {},
        client=DummyClient(),
        validator=DummyValidator(),
        writer=DummyWriter(),
        critic=None,
    )
    task = {"phase": "joins", "question": "", "metadata": {"parallelism": 20}}
    job.phase_cfg = task["metadata"]
    asyncio.run(job._run_joins_async(task))
    assert DummyPool.size == 5
