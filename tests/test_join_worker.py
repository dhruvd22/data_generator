import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.join_worker import JoinWorker
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


class DummyValidator:
    def check(self, sql):
        return True, ""


class DummyCritic:
    async def areview(self, q, sql, schema_md):
        return {}


class DummyWriter:
    def fetch(self, sql, n_rows=5):
        return []


class DummyClient:
    async def acomplete(self, messages, return_message=False, model=None):
        text = '{"question": "Q1", "sql": "SELECT * FROM a JOIN b"}'
        if return_message:
            return {"role": "assistant", "content": text}
        return text


def test_chat_log_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schema = {
        "a": TableInfo("a", [ColumnInfo("id", "int")]),
        "b": TableInfo("b", [ColumnInfo("id", "int")]),
    }
    worker = JoinWorker(
        schema,
        {"enable_worker_chat_log": True},
        DummyValidator,
        DummyCritic(),
        DummyWriter(),
        1,
        DummyClient(),
    )
    asyncio.run(worker.generate(1))
    log_files = list((tmp_path / "logs").glob("join-worker-1-*.jsonl"))
    assert len(log_files) == 1
    data = [json.loads(l) for l in log_files[0].read_text().splitlines()]
    assert any(m["role"] == "assistant" for m in data)
