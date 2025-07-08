import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.autonomous_job import _clean_sql


def test_clean_sql_removes_newlines_and_backslashes():
    raw = 'SELECT "payers";\n'
    assert _clean_sql(raw) == 'SELECT "payers"'


def test_clean_sql_collapses_whitespace():
    raw = "SELECT *\nFROM tbl\tWHERE id = 1"
    assert _clean_sql(raw) == "SELECT * FROM tbl WHERE id = 1"


def test_clean_sql_strips_semicolon():
    raw = "SELECT * FROM tbl;"
    assert _clean_sql(raw) == "SELECT * FROM tbl"
