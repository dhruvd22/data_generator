import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.input_loader import load_tasks


def test_load_tasks_happy(tmp_path):
    cfg = '''
phases:
  - name: demo
    count: 2
    builtins: [COUNT]
'''
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    schema = {"patients": object()}
    tasks = load_tasks(str(path), schema)
    assert len(tasks) == 2
    assert tasks[0]["phase"] == "demo"
    assert "question" in tasks[0]
    assert "patients" in tasks[0]["question"]


def test_load_tasks_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("::notyaml")
    with pytest.raises(ValueError):
        load_tasks(str(bad))

