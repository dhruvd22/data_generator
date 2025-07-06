"""Prompt builder utilities.

Helpers here assemble the final prompt string sent to OpenAI. ``build_prompt``
accepts a natural language question and optional few-shot examples to craft the
chat message.

Example:
    >>> prompt = build_prompt("count users", schema, {})
    >>> print(prompt.splitlines()[0])
    You are a PostgreSQL expert.
"""

__all__ = ["build_prompt"]

from textwrap import dedent
import random
from typing import Dict, List, Any


def _schema_as_markdown(schema: Dict[str, Any]) -> str:
    """Render tables and columns as a compact markdown block."""
    lines = []
    for tbl, info in schema.items():
        cols = ", ".join(f'"{c.name}"' for c in info.columns)
        lines.append(f"- **{tbl}**: {cols}")
    return "\n".join(lines)


def build_prompt(
    nl_question: str,
    schema: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    fewshot: List[Dict[str, str]] | None = None,
) -> str:
    """Return a chat prompt for the OpenAI API.

    Args:
        nl_question: User question in natural language.
        schema: Database schema mapping.
        phase_cfg: Phase configuration options.
        fewshot: Optional few-shot examples.

    Returns:
        Prompt string to send as ``user`` content.
    """
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
        All column names are quoted using double quotes; ensure your SQL does the same.
        {fn_clause}

        Schema:
        {_schema_as_markdown(schema)}

        {example_block}Question: {nl_question}
        SQL:
        """
    ).strip()
    return prompt
