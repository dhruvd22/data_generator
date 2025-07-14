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


def test_validator_respects_env(monkeypatch):
    captured = {}

    def _create_engine(url, **kwargs):
        captured.update(kwargs)

        class _Eng:
            def connect(self):
                return _FakeConn()

        return _Eng()

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pass@host/db")
    monkeypatch.setenv("DB_COCURRENT_SESSION", "33")
    monkeypatch.setattr(
        "nl_sql_generator.sql_validator.create_engine", _create_engine
    )
    SQLValidator()
    assert captured.get("pool_size") == 33
    assert captured.get("max_overflow") == 0
