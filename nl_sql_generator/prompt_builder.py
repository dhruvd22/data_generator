"""Builds the few-shot prompt for the LLM."""

from textwrap import dedent
import random, json
from typing import Dict, List, Any

def _schema_as_markdown(schema: Dict[str, Any]) -> str:
    """Render tables & columns as a compact markdown block."""
    lines = []
    for tbl, info in schema.items():
        cols = ", ".join(c.name for c in info.columns)
        lines.append(f"- **{tbl}**: {cols}")
    return "\n".join(lines)

def build_prompt(
    nl_question: str,
    schema: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    fewshot: List[Dict[str, str]] | None = None,
) -> str:
    """Return a single chat prompt string ready for the OpenAI API."""
    # Few-shot examples first (if any)
    example_block = ""
    if fewshot:
        chosen = random.sample(fewshot, k=min(2, len(fewshot)))  # keep it short
        example_lines = []
        for ex in chosen:
            example_lines.append(
                f"Q: {ex['question']}\nA: ```SQL\n{ex['sql']}\n```"
            )
        example_block = "\n\n".join(example_lines) + "\n\n"

    required_fn = phase_cfg.get("builtins", [None])[0]  # e.g., COUNT
    fn_clause = (
        f"Ensure the SQL **includes** the function `{required_fn}`."
        if required_fn
        else ""
    )

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
