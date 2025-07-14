import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.autonomous_job import AutonomousJob
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


class DummyClient:
    def __init__(self):
        self.calls = []

    async def acomplete(self, messages, return_message=False, model=None):
        self.calls.append(messages)
        text = '{"question": "Q' + str(len(self.calls)) + '", "sql": "SELECT 1"}'
        if return_message:
            return {"role": "assistant", "content": text}
        return text


class DummyValidator:
    def check(self, sql):
        return True, ""


class DummyWriter:
    def fetch(self, sql, n_rows=5):
        return []


def test_single_table_api_answer_count():
    schema = {"t": TableInfo("t", [ColumnInfo("id", "int")])}
    client = DummyClient()
    job = AutonomousJob(
        schema,
        {"api_answer_count": 1},
        client=client,
        validator=DummyValidator(),
        critic=None,
        writer=DummyWriter(),
    )
    task = {
        "phase": "single_table",
        "question": "",
        "metadata": {"table": "t", "count": 2},
    }
    result = asyncio.run(job._run_single_table_async(task))
    assert len(result.rows) == 2
    assert len(client.calls) == 2
