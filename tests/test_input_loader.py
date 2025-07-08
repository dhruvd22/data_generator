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


def test_builtins_list_with_count(tmp_path):
    cfg = """
phases:
  - name: demo
    count: 3
    builtins: [SUM, MIN]
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"t": object()})
    # two functions * count 3 each
    assert len(tasks) == 6
    assert all(t["metadata"]["builtins"][0] in {"SUM", "MIN"} for t in tasks)


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


def test_prompt_template_phase(tmp_path):
    cfg = """
phases:
  - name: docs_phase
    count: 2
    prompt_template: doc.txt
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"tbl": object()})
    assert len(tasks) == 2
    assert all(t["question"] == "" for t in tasks)
    assert all(t["metadata"]["prompt_template"] == "doc.txt" for t in tasks)


def test_schema_docs_phase(tmp_path):
    cfg = """
phases:
  - name: schema_docs
    count: 3
    prompt_template: schema_doc_template.txt
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"tbl": object()})
    assert len(tasks) == 1
    assert tasks[0]["phase"] == "schema_docs"
    assert tasks[0]["metadata"]["count"] == 3


def test_schema_relationship_phase(tmp_path):
    cfg = """
phases:
  - name: schema_relationship
    n_rows: 4
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    tasks = load_tasks(str(path), {"t": object()})
    assert len(tasks) == 1
    assert tasks[0]["phase"] == "schema_relationship"
    assert tasks[0]["metadata"]["n_rows"] == 4


def test_single_table_phase(tmp_path):
    cfg = """
phases:
  - name: single_table
    count: 2
    prompt_template: single_table_sql_template.txt
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg)

    schema = {"a": object(), "b": object()}
    tasks = load_tasks(str(path), schema)

    # two tables * count 2 each
    assert len(tasks) == 4
    assert {t["metadata"]["table"] for t in tasks} == {"a", "b"}
    assert all(t["metadata"]["prompt_template"] == "single_table_sql_template.txt" for t in tasks)
