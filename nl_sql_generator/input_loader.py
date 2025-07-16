"""Configuration-driven NL task loader.

This module reads ``config.yaml`` style files and expands each phase into a list
of :class:`NLTask` objects. Phases can define explicit questions or instruct the
loader to generate function-specific prompts.

Example:
    >>> tasks = load_tasks("config.yaml")
    >>> tasks[0]["question"]
    'Write a query using COUNT on patients'
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any
import random
import yaml


def _natural_builtin_question(fn: str, table: str) -> str:
    """Return an instruction asking the LLM to create a question and SQL."""

    fn = fn.upper()
    table = table.replace("_", " ")
    return (
        f"Create a natural language question about the {table} table that uses the {fn} function."
        " Then provide the SQL query answering it as JSON with keys 'question' and 'sql'."
    )


def _natural_table_question(table: str, count: int = 1) -> str:
    """Return an instruction asking the LLM to craft NL/SQL for ``table``."""

    table = table.replace("_", " ")
    if count <= 1:
        return (
            f"Create a natural language question about the {table} table."
            " Then provide the SQL query answering it as JSON with keys 'question' and 'sql'."
        )
    return (
        f"Create {count} unique natural language questions about the {table} table."
        " Provide the SQL answering each question. Return one JSON object per line with keys 'question' and 'sql'."
    )


class NLTask(TypedDict):
    """A single natural-language question with context.

    Attributes:
        phase: Phase name the task belongs to.
        question: The NL question text.
        metadata: Extra parameters controlling generation.
    """

    phase: str
    question: str
    metadata: Dict[str, Any]


def load_tasks(
    config_path: str,
    schema: Dict[str, Any] | None = None,
    phase: str | None = None,
) -> List[NLTask]:
    """Return tasks parsed from a configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        schema: Optional mapping of table metadata used for placeholders.
        phase: If provided, only tasks for this phase are returned.

    Returns:
        A list of :class:`NLTask` dictionaries.

    Raises:
        ValueError: If the configuration file is invalid YAML.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - YAML parser detail
        raise ValueError("Invalid YAML configuration") from exc
    if not isinstance(cfg, dict):
        raise ValueError("Invalid YAML configuration")
    defaults = cfg.get("defaults", {})

    tasks: List[NLTask] = []
    table_names = list(schema.keys()) if schema else []
    for phase_def in cfg.get("phases", []):
        name = phase_def.get("name", "unknown")
        if phase and name != phase:
            continue
        meta = {
            **defaults,
            **{
                k: v
                for k, v in phase_def.items()
                if k not in {"name", "questions", "count", "builtins"}
            },
        }

        if name.lower() == "sample_data":
            n_rows = int(phase_def.get("n_rows", 2))
            for tbl in table_names:
                q = f"Show me {n_rows} sample rows from the {tbl} table."
                meta_with_rows = {**meta, "n_rows": n_rows}
                tasks.append({"phase": name, "question": q, "metadata": meta_with_rows})
            continue
        if name.lower() == "schema_docs":
            count = int(phase_def.get("count", 1))
            q = str(
                phase_def.get(
                    "question",
                    f"Generate {count} unique question-answer pairs about the schema.",
                )
            )
            meta_with_count = {**meta, "count": count}
            tasks.append({"phase": name, "question": q, "metadata": meta_with_count})
            continue
        if name.lower() == "schema_relationship":
            q = str(
                phase_def.get(
                    "question",
                    "Generate relationship pairs between tables using sample rows.",
                )
            )
            n_rows = int(phase_def.get("n_rows", 5))
            meta_with_rows = {**meta, "n_rows": n_rows}
            tasks.append({"phase": name, "question": q, "metadata": meta_with_rows})
            continue
        if name.lower() == "single_table":
            count = int(phase_def.get("count", 1))
            table_list = table_names or ["table_1"]
            max_par = int(meta.get("parallelism", len(table_list)))
            parallel = min(len(table_list), max_par)
            for tbl in table_list:
                q = _natural_table_question(tbl, count)
                meta_with_tbl = {
                    **meta,
                    "table": tbl,
                    "count": count,
                    "parallelism": parallel,
                }
                tasks.append({"phase": name, "question": q, "metadata": meta_with_tbl})
            continue
        if name.lower() == "joins":
            count = int(phase_def.get("count", 1))
            min_joins = int(phase_def.get("min_joins", 2))
            q = str(
                phase_def.get(
                    "question",
                    f"Generate {count} question/SQL pairs requiring joins.",
                )
            )
            meta_with_count = {**meta, "count": count, "min_joins": min_joins}
            tasks.append({"phase": name, "question": q, "metadata": meta_with_count})
            continue
        if name.lower() == "complex_sqls":
            count = int(phase_def.get("count", 1))
            min_joins = int(phase_def.get("min_joins", 3))
            q = str(
                phase_def.get(
                    "question",
                    f"Generate {count} complex question/SQL pairs requiring joins.",
                )
            )
            meta_with_count = {**meta, "count": count, "min_joins": min_joins}
            tasks.append({"phase": name, "question": q, "metadata": meta_with_count})
            continue
        questions = phase_def.get("questions")
        if questions:
            for q in questions:
                tasks.append({"phase": name, "question": str(q), "metadata": meta})
            continue

        if phase_def.get("prompt_template") and not phase_def.get("builtins"):
            count = int(phase_def.get("count", 1))
            q = str(phase_def.get("question", ""))
            for _ in range(count):
                tasks.append({"phase": name, "question": q, "metadata": meta})
            continue

        builtins_spec = phase_def.get("builtins")
        count = int(phase_def.get("count", 0))

        if isinstance(builtins_spec, dict):
            for fn, cnt in builtins_spec.items():
                for i in range(int(cnt or 5)):
                    table = (
                        random.choice(table_names) if table_names else f"table_{i + 1}"
                    )
                    q = _natural_builtin_question(fn, table)
                    meta_with_fn = {**meta, "builtins": [fn], "table": table}
                    tasks.append(
                        {"phase": name, "question": q, "metadata": meta_with_fn}
                    )
            continue

        if isinstance(builtins_spec, list):
            per_fn = count or 5
            for fn in builtins_spec:
                for i in range(per_fn):
                    table = (
                        random.choice(table_names) if table_names else f"table_{i + 1}"
                    )
                    q = _natural_builtin_question(fn, table)
                    meta_with_fn = {**meta, "builtins": [fn], "table": table}
                    tasks.append(
                        {"phase": name, "question": q, "metadata": meta_with_fn}
                    )
            continue

        builtins = meta.get("builtins", [])
        for i in range(count):
            builtin = random.choice(builtins) if builtins else "COUNT"
            table = random.choice(table_names) if table_names else f"table_{i + 1}"
            q = _natural_builtin_question(builtin, table)
            meta_with_fn = {**meta, "builtins": [builtin], "table": table}
            tasks.append({"phase": name, "question": q, "metadata": meta_with_fn})

    return tasks
