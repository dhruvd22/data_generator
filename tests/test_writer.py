import json
import os
import sys
from faker import Faker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.writer import ResultWriter


class _FakeResult:
    def __init__(self):
        self._rows = [(1, "Alice", "2024-07-01")]

    def keys(self):
        return ["id", "name", "created_at"]

    def fetchmany(self, n):
        return self._rows[:n]


class _FakeConn:
    def execute(self, *_):
        return _FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class _FakeEngine:
    def connect(self):
        return _FakeConn()


def test_fetch_deterministic(monkeypatch):
    Faker.seed(99)
    w = ResultWriter.__new__(ResultWriter)
    w.eng = _FakeEngine()
    w.fake = Faker()

    rows1 = w.fetch("SELECT")

    Faker.seed(99)
    w.fake = Faker()
    rows2 = w.fetch("SELECT")

    assert rows1 == rows2


def test_write_jsonl(tmp_path):
    w = ResultWriter.__new__(ResultWriter)
    path = tmp_path / "rows.jsonl"
    w.write_jsonl([{"a": 1}], path)
    assert json.loads(path.read_text().strip()) == {"a": 1}
