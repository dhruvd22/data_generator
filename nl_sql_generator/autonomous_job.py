from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Callable
import os

from .input_loader import NLTask

from .prompt_builder import build_prompt, _schema_as_markdown
from .openai_responses import ResponsesClient
from .sql_validator import SQLValidator
from .critic import Critic
from .writer import ResultWriter
from .logger import log_call
import logging

__all__ = ["AutonomousJob", "JobResult"]

log = logging.getLogger(__name__)


def _clean_sql(sql: str) -> str:
    """Return SQL without newlines or backslashes."""
    if not isinstance(sql, str):
        return ""
    sanitized = sql.replace("\n", " ").replace("\r", " ")
    sanitized = sanitized.replace("\t", " ").replace("\\", "")
    sanitized = " ".join(sanitized.split())
    return sanitized.strip()


@dataclass
class JobResult:
    """Container for the final output of a single NL→SQL job."""

    question: str
    sql: str
    rows: List[Dict[str, Any]]


class AutonomousJob:
    """Run the NL→SQL pipeline driven by the OpenAI tools API."""

    def __init__(
        self,
        schema: Dict[str, Any],
        phase_cfg: Dict[str, Any] | None = None,
        client: ResponsesClient | None = None,
        validator: SQLValidator | None = None,
        critic: Critic | None = None,
        writer: ResultWriter | None = None,
    ) -> None:
        self.schema = schema
        self.phase_cfg = phase_cfg or {}
        self.client = client or ResponsesClient()
        self.validator = validator or SQLValidator()
        self.critic = critic or Critic(client=self.client)
        self.writer = writer or ResultWriter()

        self._tool_map: Dict[str, Callable[..., Any]] = {
            "generate_sql": self._tool_generate_sql,
            "validate_sql": self._tool_validate_sql,
            "critic": self._tool_critic,
            "writer": self._tool_writer,
        }

        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_sql",
                    "parameters": {
                        "type": "object",
                        "properties": {"nl_question": {"type": "string"}},
                        "required": ["nl_question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_sql",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "critic",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "writer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "n_rows": {"type": "integer", "default": 5},
                        },
                        "required": ["sql"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @log_call
    def _tool_generate_sql(self, nl_question: str) -> str:
        """LLM tool: generate SQL for ``nl_question``."""
        prompt = build_prompt(nl_question, self.schema, self.phase_cfg)
        log.info("Generating SQL for question: %s", nl_question)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        sql = self.client.run_jobs([messages])[0]
        sql = sql.strip().strip("`")
        sql = re.sub(r"(?i)^sql\s*", "", sql)
        return _clean_sql(sql)

    @log_call
    def _tool_validate_sql(self, sql: str) -> Dict[str, Any]:
        """LLM tool: validate SQL via :class:`SQLValidator`."""
        log.info("Validating SQL: %s", sql)
        ok, err = self.validator.check(sql)
        return {"ok": ok, "error": err}

    @log_call
    def _tool_critic(self, sql: str) -> Dict[str, Any]:
        """LLM tool: run the critic."""
        log.info("Running critic on SQL")
        result = self.critic.review("", sql, _schema_as_markdown(self.schema))
        return result

    @log_call
    def _tool_writer(self, sql: str, n_rows: int | None = None) -> List[Dict[str, Any]]:
        """LLM tool: execute SQL and return fake rows."""
        if n_rows is None:
            n_rows = int(self.phase_cfg.get("n_rows", 5))
        log.info("Fetching %d rows for SQL", n_rows)
        return self.writer.fetch(sql, n_rows)

    # ------------------------------------------------------------------
    # main entrypoints
    # ------------------------------------------------------------------
    @log_call
    def run_task(self, task: NLTask) -> JobResult:
        """Process ``task`` letting the LLM drive via tools."""
        import json

        self.phase_cfg = task.get("metadata", {})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data-engineer agent. Column names in the database are quoted using double quotes. "
                    "Return a JSON object with only a 'sql' field containing the valid query. "
                    "Here is the schema:\n"
                    f"{_schema_as_markdown(self.schema)}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": task["phase"],
                        "question": task["question"],
                        **task.get("metadata", {}),
                        "budget_left": self.client.remaining_budget(),
                    }
                ),
            },
        ]

        use_rows = bool(self.phase_cfg.get("use_sample_rows", False))
        tools = (
            self._tools
            if use_rows
            else [t for t in self._tools if t["function"]["name"] != "writer"]
        )

        while True:
            msg = self.client.run_jobs(
                [messages], tools=tools, return_message=True
            )[0]
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                messages.append(
                    msg if isinstance(msg, dict) else msg.model_dump()
                )
                for call in tool_calls:
                    fn = self._tool_map.get(call.function.name)
                    args = json.loads(call.function.arguments or "{}")
                    log.info(
                        "Invoking tool %s with args %s", call.function.name, args
                    )
                    result = fn(**args)
                    log.info("Tool %s returned %s", call.function.name, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result),
                        }
                    )
                continue

            content = msg.get("content") if isinstance(msg, dict) else msg.content
            try:
                data = json.loads(content or "{}")
            except Exception:
                data = {"sql": content or "", "rows": []}
            sql = _clean_sql(data.get("sql", ""))
            return JobResult(task["question"], sql, data.get("rows", []))

    @log_call
    def run_tasks(self, tasks: List[NLTask]) -> List[JobResult]:
        """Process many tasks synchronously."""

        results = []
        total = len(tasks)
        for idx, t in enumerate(tasks, 1):
            log.info("Running task %d/%d: %s", idx, total, t.get("question"))
            res = self.run_task(t)
            results.append(res)

            out_dir = t.get("metadata", {}).get("dataset_output_file_dir")
            if out_dir:
                path = os.path.join(out_dir, "dataset.jsonl")
                self.writer.append_jsonl(
                    {"question": res.question, "sql": res.sql},
                    path,
                )
                log.info("Wrote dataset entry to %s", path)
        return results
