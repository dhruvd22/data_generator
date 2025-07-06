import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.input_loader import load_tasks


def test_load_tasks_happy(tmp_path):
    cfg = """
phases:
  - name: demo
    builtins:
      COUNT: 2
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    schema = {"patients": object()}
    tasks = load_tasks(str(path), schema)
    assert len(tasks) == 2
    assert tasks[0]["phase"] == "demo"
    assert "question" in tasks[0]
    assert "patients" in tasks[0]["question"]


def test_metadata_includes_output_dir(tmp_path):
    cfg = """
phases:
  - name: demo
    count: 1
    dataset_output_file_dir: out/demo
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"t": object()})
    assert tasks[0]["metadata"]["dataset_output_file_dir"] == "out/demo"


def test_load_tasks_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("::notyaml")
    with pytest.raises(ValueError):
        load_tasks(str(bad))


def test_load_tasks_single_phase(tmp_path):
    cfg = """
phases:
  - name: first
    count: 1
  - name: second
    count: 2
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"t": object()}, phase="second")
    assert len(tasks) == 2
    assert all(t["phase"] == "second" for t in tasks)


def test_builtin_default_count(tmp_path):
    cfg = """
phases:
  - name: demo
    builtins: [MAX, AVG]
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"t": object()})
    # two functions * default 5 each
    assert len(tasks) == 10
    assert {t["metadata"]["builtins"][0] for t in tasks} == {"MAX", "AVG"}


def test_sample_data_phase(tmp_path):
    cfg = """
phases:
  - name: sample_data
    n_rows: 2
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    schema = {"tbl1": object(), "tbl2": object()}
    tasks = load_tasks(str(path), schema)

    assert len(tasks) == 2
    assert all("sample rows" in t["question"] for t in tasks)
    assert all(t["metadata"]["n_rows"] == 2 for t in tasks)
