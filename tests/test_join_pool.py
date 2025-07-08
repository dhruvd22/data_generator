import os
import sys
import asyncio
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.join_pool import JoinPool


class DummyWriter:
    def __init__(self):
        self.queries = []

    def fetch(self, sql, n_rows=5):
        self.queries.append(sql)
        return [{"id": 1}]


class DummyClient:
    async def acomplete(self, *args, **kwargs):
        return ""


class DummyWorker:
    def __init__(self, schema, cfg, validator_cls, critic, writer, wid, client):
        self.cfg = cfg
        DummyWorker.last_cfg = cfg

    async def generate(self, batch_size):
        return []


def test_join_pool_passes_sample_rows(monkeypatch):
    schema = {"a": {}, "b": {}}
    writer = DummyWriter()
    client = DummyClient()

    monkeypatch.setattr("nl_sql_generator.join_pool.JoinWorker", DummyWorker)

    async def _chunks(self):
        return [{"a": {}, "b": {}}]

    monkeypatch.setattr("nl_sql_generator.join_pool.JoinPool._schema_chunks", _chunks)

    pool = JoinPool(
        schema,
        {"use_sample_rows": True, "n_rows": 1, "parallelism": 1},
        object,
        writer,
        None,
        client,
    )
    asyncio.run(pool.generate())
    # writer.fetch should be called for each table
    assert len(writer.queries) >= 2
    # config passed to worker should include sample_rows
    assert "sample_rows" in DummyWorker.last_cfg
    assert set(DummyWorker.last_cfg["sample_rows"].keys()) == {"a", "b"}


def test_join_pool_cleans_sql(monkeypatch):
    class Worker(DummyWorker):
        async def generate(self, batch_size):
            return [
                {"question": "Q1", "sql": "SELECT * FROM a;"},
                {"question": "Q2", "sql": "SELECT * FROM b ;"},
            ]

    schema = {"a": {}, "b": {}}
    writer = DummyWriter()
    client = DummyClient()
    monkeypatch.setattr("nl_sql_generator.join_pool.JoinWorker", Worker)

    async def _chunks(self):
        return [schema]

    monkeypatch.setattr("nl_sql_generator.join_pool.JoinPool._schema_chunks", _chunks)

    pool = JoinPool(
        schema, {"parallelism": 1, "count": 2}, object, writer, None, client
    )
    result = asyncio.run(pool.generate())
    assert sorted(p["sql"] for p in result) == ["SELECT * FROM a", "SELECT * FROM b"]
