"""Configuration-driven NL task loader."""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any
import random
import yaml


class NLTask(TypedDict):
    """A single natural-language question with context."""

    phase: str
    question: str
    metadata: Dict[str, Any]


def load_tasks(
    config_path: str,
    schema: Dict[str, Any] | None = None,
    phase: str | None = None,
) -> List[NLTask]:
    """Return a list of :class:`NLTask` parsed from ``config_path``.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.
    schema:
        Optional mapping of table names used to craft questions.
    phase:
        If provided, only tasks for this phase are returned.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - YAML parser detail
        raise ValueError("Invalid YAML configuration") from exc
    if not isinstance(cfg, dict):
        raise ValueError("Invalid YAML configuration")

    tasks: List[NLTask] = []
    table_names = list(schema.keys()) if schema else []
    for phase_def in cfg.get("phases", []):
        name = phase_def.get("name", "unknown")
        if phase and name != phase:
            continue
        meta = {
            k: v
            for k, v in phase_def.items()
            if k not in {"name", "questions", "count", "builtins"}
        }
        questions = phase_def.get("questions")
        if questions:
            for q in questions:
                tasks.append({"phase": name, "question": str(q), "metadata": meta})
            continue

        builtins_spec = phase_def.get("builtins")
        count = int(phase_def.get("count", 0))

        if isinstance(builtins_spec, dict):
            for fn, cnt in builtins_spec.items():
                for i in range(int(cnt or 5)):
                    table = (
                        random.choice(table_names)
                        if table_names
                        else f"table_{i + 1}"
                    )
                    q = f"Write a query using {fn} on {table}"
                    meta_with_fn = {**meta, "builtins": [fn]}
                    tasks.append({"phase": name, "question": q, "metadata": meta_with_fn})
            continue

        if isinstance(builtins_spec, list):
            if count:
                for i in range(count):
                    builtin = random.choice(builtins_spec)
                    table = (
                        random.choice(table_names)
                        if table_names
                        else f"table_{i + 1}"
                    )
                    q = f"Write a query using {builtin} on {table}"
                    meta_with_fn = {**meta, "builtins": [builtin]}
                    tasks.append({"phase": name, "question": q, "metadata": meta_with_fn})
                continue
            else:
                for fn in builtins_spec:
                    for i in range(5):
                        table = (
                            random.choice(table_names)
                            if table_names
                            else f"table_{i + 1}"
                        )
                        q = f"Write a query using {fn} on {table}"
                        meta_with_fn = {**meta, "builtins": [fn]}
                        tasks.append({"phase": name, "question": q, "metadata": meta_with_fn})
                continue

        builtins = meta.get("builtins", [])
        for i in range(count):
            builtin = random.choice(builtins) if builtins else "COUNT"
            table = random.choice(table_names) if table_names else f"table_{i + 1}"
            q = f"Write a query using {builtin} on {table}"
            tasks.append({"phase": name, "question": q, "metadata": meta})

    return tasks
