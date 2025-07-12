import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.complex_sql_pool import ComplexSqlPool


class DummyWriter:
    def __init__(self):
        self.queries = []

    def fetch(self, sql, n_rows=5):
        self.queries.append(sql)
        return [{"id": 1}]


class DummyClient:
    async def acomplete(self, *args, **kwargs):
        # single set with two tables
        return '{"tables": ["a", "b"]}'


class DummyWorker:
    def __init__(self, schema, cfg, validator_cls, critic, writer, wid, client):
        self.cfg = cfg
        DummyWorker.last_cfg = cfg

    async def generate(self, batch_size):
        return []


def test_complex_pool_fetches_rows_for_chunks(monkeypatch):
    schema = {"a": {}, "b": {}}
    writer = DummyWriter()
    client = DummyClient()

    pool = ComplexSqlPool(
        schema,
        {"use_sample_rows": True, "n_rows": 1, "parallelism": 1},
        object,
        writer,
        None,
        client,
    )
    chunks = asyncio.run(pool._schema_chunks())
    assert len(writer.queries) == 2
    assert chunks and list(chunks[0].keys()) == ["a", "b"]


def test_complex_pool_default_min_joins(monkeypatch):
    schema = {"a": {}, "b": {}, "c": {}}
    writer = DummyWriter()
    client = DummyClient()

    monkeypatch.setattr(
        "nl_sql_generator.complex_sql_pool.JoinWorker", DummyWorker
    )

    async def _chunks(self):
        return [{"a": {}, "b": {}, "c": {}}]

    monkeypatch.setattr(
        "nl_sql_generator.complex_sql_pool.ComplexSqlPool._schema_chunks", _chunks
    )

    pool = ComplexSqlPool(
        schema,
        {"parallelism": 1, "count": 1},
        object,
        writer,
        None,
        client,
    )
    asyncio.run(pool.generate())
    assert DummyWorker.last_cfg["min_joins"] == 3
