import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.sql_validator import SQLValidator


class _FakeConn:
    def execute(self, sql):
        if "FAIL" in str(sql):
            raise Exception("boom")
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class _FakeEngine:
    def connect(self):
        return _FakeConn()


def test_validator_success():
    v = SQLValidator.__new__(SQLValidator)
    v.eng = _FakeEngine()
    ok, err = v.check("SELECT 1")
    assert ok and err is None


def test_validator_failure():
    v = SQLValidator.__new__(SQLValidator)
    v.eng = _FakeEngine()
    ok, err = v.check("FAIL")
    assert not ok
