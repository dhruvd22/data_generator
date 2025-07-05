"""LLM-powered SQL review helper."""

from __future__ import annotations

__all__ = ["Critic"]

import json
import os
from datetime import datetime
from typing import Dict, List, Any

from .openai_responses import ResponsesClient


class Critic:
    """Review generated SQL using a GPT-4.1 critic persona."""

    def __init__(self, client: ResponsesClient | None = None, log_dir: str = "logs") -> None:
        self.client = client or ResponsesClient(model="gpt-4.1")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _log(self, data: Dict[str, Any]) -> None:
        """Append ``data`` to the daily JSONL log file."""

        log_path = os.path.join(
            self.log_dir, f"critic-{datetime.utcnow():%Y%m%d}.jsonl"
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def review(self, question: str, sql_candidate: str, schema_markdown: str) -> str:
        """Return vetted SQL, applying model fixes when score < 0.7."""
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Critic persona: Review SQL for correctness, index usage, security. "
                    "Respond with a JSON object: {'score': 0-1, 'reason': str, 'fixed_sql': str}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nSQL Candidate:\n{sql_candidate}\n\nSchema:\n{schema_markdown}"
                ),
            },
        ]

        response = self.client.run_jobs([messages])[0]
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"score": 0.0, "reason": "Invalid JSON", "fixed_sql": sql_candidate}

        self._log({
            "question": question,
            "candidate_sql": sql_candidate,
            "schema": schema_markdown,
            "model_response": result,
        })

        score = float(result.get("score", 0))
        fixed_sql = str(result.get("fixed_sql", sql_candidate))
        return fixed_sql if score < 0.7 else sql_candidate
