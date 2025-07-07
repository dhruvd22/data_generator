import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.autonomous_job import AutonomousJob, JobResult


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


def test_dataset_only_question_sql(tmp_path, monkeypatch):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )
    job.run_task = lambda t: JobResult(t["question"], "SELECT 1", [])
    t = {
        "phase": "demo",
        "question": "foo?",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [{"question": "foo?", "sql": "SELECT 1"}]


def test_schema_doc_dataset(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    job._run_schema_doc = lambda t: JobResult(
        "", "", [{"table_doc": "doc", "sample_questions": ["Q1"]}]
    )

    t = {
        "phase": "schema_doc",
        "question": "",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [{"question": "Q1", "doc": "doc"}]


def test_schema_docs_dataset(tmp_path):
    writer = DummyWriter()
    job = AutonomousJob(
        {}, writer=writer, client=DummyClient(), validator=DummyValidator(), critic=None
    )

    job._run_schema_docs = lambda t: JobResult("", "", [{"question": "Q?", "answer": "A"}])

    t = {
        "phase": "schema_docs",
        "question": "",
        "metadata": {"dataset_output_file_dir": str(tmp_path)},
    }
    job.run_tasks([t])
    assert writer.seen == [{"question": "Q?", "answer": "A"}]
