import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.autonomous_job import AutonomousJob, JobResult
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


class DummyClient:
    def remaining_budget(self):
        return 1.0


class DummyValidator:
    def check(self, sql):
        return True, ""


class DummyWriter:
    def __init__(self):
        self.seen = []

    def fetch(self, sql, n_rows=5):
        return []

    def append_jsonl(self, row, path):
        self.seen.append(row)
        with open(path, "a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps(row))
            fh.write("\n")


def test_dataset_only_question_sql(tmp_path, monkeypatch):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult(t["question"], "SELECT 1", [])

    job.run_task = _rt
    t = {
        "phase": "demo",
        "question": "foo?",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [{"question": "foo?", "sql": "SELECT 1"}]


def test_schema_docs_dataset(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _sd(t):
        return JobResult("", "", [{"question": "Q?", "answer": "A"}])

    job._run_schema_docs_async = _sd

    t = {
        "phase": "schema_docs",
        "question": "",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [{"question": "Q?", "answer": "A"}]


def test_run_version_overwrites(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult(t["question"], "SELECT 1", [])

    job.run_task = _rt
    t = {
        "phase": "demo",
        "question": "foo?",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }

    p = tmp_path / "dataset_v1.jsonl"
    p.write_text("old\n")
    job.run_tasks([t, t], run_version="v1")
    content = p.read_text().strip().splitlines()
    assert len(content) == 1


def test_builtins_skip_fail(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult(t["question"], "FAIL", [])

    job.run_task = _rt
    t = {
        "phase": "builtins",
        "question": "bad?",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == []


def test_deduplicate_pairs(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult(t["question"], "SELECT 1", [])

    job.run_task = _rt
    t = {
        "phase": "demo",
        "question": "same?",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t, t])
    assert writer.seen == [{"question": "same?", "sql": "SELECT 1"}]


def test_single_table_multiple_pairs(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult("", "", [
            {"question": "Q1", "sql": "S1"},
            {"question": "Q2", "sql": "S2"},
        ])

    job.run_task = _rt
    t = {
        "phase": "single_table",
        "question": "",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [
        {"question": "Q1", "sql": "S1"},
        {"question": "Q2", "sql": "S2"},
    ]


def test_joins_multiple_pairs(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult("", "", [
            {"question": "JQ1", "sql": "JS1"},
            {"question": "JQ2", "sql": "JS2"},
        ])

    job.run_task = _rt
    t = {
        "phase": "joins",
        "question": "",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [
        {"question": "JQ1", "sql": "JS1"},
        {"question": "JQ2", "sql": "JS2"},
    ]


def test_tag_schema_json(tmp_path):
    writer = DummyWriter()
    schema = {
        "users": TableInfo(
            "users",
            [ColumnInfo("id", "int"), ColumnInfo("name", "text")],
            "id",
        )
    }
    job = AutonomousJob(
        schema, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    async def _rt(t):
        return JobResult(t["question"], "SELECT * FROM users", [])

    job.run_task = _rt
    t = {
        "phase": "demo",
        "question": "foo?",
        "metadata": {"dataset_output_file_dir": str(tmp_path), "tag_schema_json": True},
    }
    job.run_tasks([t])
    row = writer.seen[0]
    assert "schema" in row
    assert "users" in row["schema"].get("tables", {})
