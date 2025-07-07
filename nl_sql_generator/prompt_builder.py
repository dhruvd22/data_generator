"""Prompt builder utilities.

Helpers here assemble the final prompt string sent to OpenAI. ``build_prompt``
accepts a natural language question and optional few-shot examples to craft the
chat message.

Example:
    >>> prompt = build_prompt("count users", schema, {})
    >>> print(prompt.splitlines()[0])
    You are a PostgreSQL expert.
"""

__all__ = ["build_prompt", "load_template_messages"]

from textwrap import dedent
import random
from typing import Dict, List, Any
import json
import os
from .schema_loader import SchemaLoader, TableInfo


def _schema_as_markdown(schema: Dict[str, Any]) -> str:
    """Render tables and columns as a compact markdown block."""
    lines = []
    for tbl, info in schema.items():
        cols = ", ".join(f'"{c.name}"' for c in info.columns)
        lines.append(f"- **{tbl}**: {cols}")
    return "\n".join(lines)


def load_template_messages(
    template_name: str, schema: Dict[str, Any], nl_question: str
) -> List[Dict[str, str]]:
    """Return chat messages rendered from ``template_name``.

    The template should contain blocks starting with ``### role: <role>``.
    Supported placeholders include ``{{schema_json}}`` and ``{{nl_question}}``.
    """

    path = os.path.join(os.path.dirname(__file__), "prompt_template", template_name)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # ``schema`` may use :class:`TableInfo` dataclasses. Convert to plain
    # dictionaries so ``json.dumps`` succeeds when rendering the template.
    example_val = next(iter(schema.values()), None)
    if isinstance(example_val, TableInfo):
        schema = SchemaLoader.to_json(schema)

    replacements = {
        "schema_json": json.dumps(schema, indent=2),
        "nl_question": nl_question,
    }
    for key, val in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", val)

    messages: List[Dict[str, str]] = []
    role = None
    buf: List[str] = []
    for line in text.splitlines():
        if line.startswith("### role:"):
            if role is not None:
                messages.append({"role": role, "content": "\n".join(buf).strip()})
            role = line.split(":", 1)[1].strip()
            buf = []
        else:
            buf.append(line)
    if role is not None:
        messages.append({"role": role, "content": "\n".join(buf).strip()})
    return messages


def build_prompt(
    nl_question: str,
    schema: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    fewshot: List[Dict[str, str]] | None = None,
) -> Any:
    """Return a chat prompt for the OpenAI API.

    Args:
        nl_question: User question in natural language.
        schema: Database schema mapping.
        phase_cfg: Phase configuration options.
        fewshot: Optional few-shot examples.

    Returns:
        Either a prompt string or list of chat messages.
    """
    template_name = phase_cfg.get("prompt_template")
    if template_name:
        return load_template_messages(template_name, schema, nl_question)

    # Few-shot examples first (if any)
    example_block = ""
    if fewshot:
        chosen = random.sample(fewshot, k=min(2, len(fewshot)))  # keep it short
        example_lines = []
        for ex in chosen:
            example_lines.append(f"Q: {ex['question']}\nA: ```SQL\n{ex['sql']}\n```")
        example_block = "\n\n".join(example_lines) + "\n\n"

    required_fn = phase_cfg.get("builtins", [None])[0]  # e.g., COUNT
    fn_clause = f"Ensure the SQL **includes** the function `{required_fn}`." if required_fn else ""

    prompt = dedent(
        f"""
        You are a PostgreSQL expert. Given the database schema and a question,
        output **only** the SQL query (no explanation) using valid PostgreSQL syntax.
        {fn_clause}

        Schema:
        {_schema_as_markdown(schema)}

        {example_block}Question: {nl_question}
        SQL:
        """
    ).strip()
    return prompt
