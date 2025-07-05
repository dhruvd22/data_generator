from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .prompt_builder import build_prompt, _schema_as_markdown
from .openai_responses import ResponsesClient
from .sql_validator import SQLValidator
from .critic import Critic
from .writer import ResultWriter
from .logger import log_call
import logging

log = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Container for the final output of a single NL→SQL job."""

    question: str
    sql: str
    rows: List[Dict[str, Any]]


class AutonomousJob:
    """Run the NL→SQL pipeline for one or more questions."""

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

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @log_call
    def _generate_sql(self, question: str) -> str:
        prompt = build_prompt(question, self.schema, self.phase_cfg)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        sql = self.client.run_jobs([messages])[0]
        sql = sql.strip().strip("`")
        sql = re.sub(r"(?i)^sql\s*", "", sql)
        return sql

    # ------------------------------------------------------------------
    # main entrypoints
    # ------------------------------------------------------------------
    @log_call
    def run_sync(self, nl_question: str) -> JobResult:
        """Process a single question synchronously."""
        sql = self._generate_sql(nl_question)
        ok, err = self.validator.check(sql)
        if not ok:
            raise RuntimeError(f"Invalid SQL: {err}")

        sql = self.critic.review(nl_question, sql, _schema_as_markdown(self.schema))
        ok, err = self.validator.check(sql)
        if not ok:
            raise RuntimeError(f"Invalid SQL after critic: {err}")

        rows = self.writer.fetch(sql)
        return JobResult(nl_question, sql, rows)

    @log_call
    def run_async(self, nl_questions: List[str]) -> List[JobResult]:
        """Process many questions concurrently."""

        async def worker(q: str) -> JobResult:
            return await asyncio.to_thread(self.run_sync, q)

        async def runner() -> List[JobResult]:
            tasks = [asyncio.create_task(worker(q)) for q in nl_questions]
            return await asyncio.gather(*tasks)

        return asyncio.run(runner())
