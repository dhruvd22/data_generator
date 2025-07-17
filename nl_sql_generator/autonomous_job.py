"""Autonomous NL→SQL workflow orchestrator.

This module exposes :class:`AutonomousJob` which coordinates the prompt
generation, SQL validation and result fetching steps. It ties together the
``ResponsesClient`` and other helpers to execute the end-to-end pipeline.

Example:
    >>> from nl_sql_generator.autonomous_job import AutonomousJob
    >>> job = AutonomousJob(schema)
    >>> result = job.run_task({"phase": "demo", "question": "list payers"})
    >>> print(result.sql)
    SELECT * FROM payers
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import asyncio
import inspect
import os

from .input_loader import NLTask

from .prompt_builder import build_prompt, _schema_as_markdown
from .openai_responses import ResponsesClient
from .sql_validator import SQLValidator
from .critic import Critic
from .writer import ResultWriter
from .schema_loader import SchemaLoader, TableInfo
from .logger import log_call
from .utils import limit_pool_size
import logging

__all__ = ["AutonomousJob", "JobResult"]

log = logging.getLogger(__name__)


def _clean_sql(sql: str) -> str:
    """Return SQL without newlines or backslashes.

    Args:
        sql: Raw SQL text.

    Returns:
        Sanitised single-line SQL.
    """

    # The OpenAI API occasionally returns formatted SQL snippets. Cleaning
    # ensures consistent validator input.
    if not isinstance(sql, str):
        return ""
    sanitized = sql.replace("\n", " ").replace("\r", " ")
    sanitized = sanitized.replace("\t", " ").replace("\\", "")
    sanitized = " ".join(sanitized.split())
    sanitized = sanitized.strip()
    if sanitized.endswith(";"):
        sanitized = sanitized[:-1].strip()
    return sanitized


@dataclass
class JobResult:
    """Container for the final output of a single NL→SQL job.

    Attributes:
        question: The natural language question.
        sql: SQL generated for the question.
        rows: Fake result rows returned by the writer.
    """

    question: str
    sql: str
    rows: List[Dict[str, Any]]

    def __post_init__(self) -> None:
        """Normalise SQL to a single line."""
        self.sql = _clean_sql(self.sql)


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
        pool_size: int | None = None,
    ) -> None:
        """Create a new job instance.

        Args:
            schema: Database schema mapping.
            phase_cfg: Configuration overrides for the current phase.
            client: Optional :class:`ResponsesClient` instance.
            validator: Optional :class:`SQLValidator` for syntax checks.
            critic: Optional :class:`Critic` for SQL review.
            writer: Optional :class:`ResultWriter` for dataset generation.
        """

        self.schema = schema
        self.phase_cfg = phase_cfg or {}
        self.client = client or ResponsesClient()
        self.pool_size = pool_size or int(os.getenv("DB_COCURRENT_SESSION", "50"))
        self.validator = validator or SQLValidator(pool_size=self.pool_size)
        self.critic = critic or Critic(client=self.client)
        self.writer = writer or ResultWriter()
        self._write_lock = asyncio.Lock()
        # maximum number of tasks that may run concurrently
        self.tasks_parallelism = 1

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

    def _extract_tables(self, text: str) -> list[str]:
        """Return list of schema tables mentioned in ``text``."""
        lower = (text or "").lower()
        return [t for t in self.schema if t.lower() in lower]

    def _schema_subset_json(self, tables: list[str]) -> dict:
        """Return JSON-serialisable subset of ``self.schema``."""
        subset = {t: self.schema[t] for t in tables if t in self.schema}
        log.debug("Creating schema subset for tables: %s", list(subset))
        example = next(iter(subset.values()), None)
        if example is not None and isinstance(example, TableInfo):
            return SchemaLoader.to_json(subset)
        return subset

    async def _run_schema_docs_async(self, task: NLTask) -> JobResult:
        """Generate NL⇄schema documentation pairs using worker agents."""

        from .agent_pool import AgentPool
        from .validator import NoOpValidator

        pool = AgentPool(
            self.schema,
            task.get("metadata", {}),
            NoOpValidator,
            self.writer,
            self.client,
        )
        pairs = await pool.generate()
        return JobResult(task.get("question", ""), "", pairs)

    async def _run_schema_relationship_async(self, task: NLTask) -> JobResult:
        """Generate table relationship pairs using sample rows."""

        from .schema_relationship import discover_relationships

        self.pool_size = limit_pool_size(self.tasks_parallelism, self.pool_size)
        log.info(
            "DB pool size adjusted to %d across %d tasks", self.pool_size, self.tasks_parallelism
        )

        n_rows = int(task.get("metadata", {}).get("n_rows", 5))
        # ``discover_relationships`` currently accepts a ``sample_limit``
        # argument for controlling how many rows to analyse.  The
        # configuration uses ``n_rows`` for this value, so we map it through
        # here.  Additional metadata such as ``parallelism`` may be supplied
        # but is ignored as the helper does not implement it yet.
        pairs = await discover_relationships(
            self.schema,
            self.writer.eng,
            sample_limit=n_rows,
            pool_size=self.pool_size,
        )
        return JobResult(task.get("question", ""), "", pairs)

    async def _run_single_table_async(self, task: NLTask) -> JobResult:
        """Generate multiple NL/SQL pairs for a single table.

        The request is broken into batches controlled by ``api_answer_count`` so
        that the worker can keep asking for additional pairs until ``count`` is
        reached.  This mirrors the behaviour of the schema documentation workers
        and avoids very long single prompts.
        """

        from .worker_agent import _parse_pairs

        table = task.get("metadata", {}).get("table")
        if not table or table not in self.schema:
            return JobResult(task.get("question", ""), "", [])

        k = int(task.get("metadata", {}).get("count", 1))
        subset = {table: self.schema[table]}
        api_count = int(self.phase_cfg.get("api_answer_count", k))
        max_attempts = int(self.phase_cfg.get("max_attempts", 6))
        parallel = int(self.phase_cfg.get("parallelism", 1))
        parallel = min(parallel, len(self.schema))
        if hasattr(self.client, "set_parallelism"):
            self.client.set_parallelism(parallel)
        log.info("Assigned %d concurrent OpenAI sessions", parallel)
        self.pool_size = limit_pool_size(self.tasks_parallelism, self.pool_size)
        log.info(
            "DB pool size adjusted to %d across %d tasks", self.pool_size, self.tasks_parallelism
        )

        extra = {"table": table, "count": min(api_count, k)}
        if self.phase_cfg.get("use_sample_rows"):
            n = int(self.phase_cfg.get("n_rows", 5))
            try:
                rows = self.writer.fetch(f"SELECT * FROM {table} LIMIT {n}", n)
                extra["sample_rows"] = {table: rows}
                log.info("Loaded sample rows for %s: %s", table, rows)
            except Exception as err:
                log.warning("Failed fetching sample rows for %s: %s", table, err)

        log.info("Sending schema to OpenAI: %s", subset)

        messages = build_prompt(
            task.get("question", ""), subset, {**self.phase_cfg, **extra}
        )
        if not isinstance(messages, list):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages},
            ]

        log.info(
            "Generating %d single-table pairs using api_answer_count=%d",
            k,
            api_count,
        )

        results: list[dict[str, str]] = []
        attempts = 0
        n_rows = int(self.phase_cfg.get("n_rows", 5))

        validation_sem = asyncio.Semaphore(self.pool_size)

        async def _validate_pair(p: dict[str, str]) -> dict[str, str]:
            if "sql" not in p:
                return p
            sql = _clean_sql(p["sql"])
            async with validation_sem:
                ok, err = await asyncio.to_thread(self.validator.check, sql)
                if not ok:
                    log.warning("SQL validation failed: %s", err)
                    p["sql"] = "FAIL"
                else:
                    try:
                        await asyncio.to_thread(self.writer.fetch, sql, n_rows)
                        p["sql"] = sql
                    except Exception as err:
                        log.warning("Execution failed for %s: %s", sql, err)
                        p["sql"] = "FAIL"
            return p

        while len(results) < k and attempts < max_attempts:
            log.info(
                "API request %d with history length %d", attempts + 1, len(messages)
            )
            msg = await self.client.acomplete(messages, return_message=True)
            msg_dict = msg if isinstance(msg, dict) else msg.model_dump()
            pairs = _parse_pairs(msg_dict.get("content", ""))
            log.info(
                "Received %d candidate pairs (%d/%d total)",
                len(pairs),
                len(results),
                k,
            )

            remaining = k - len(results)
            batch = pairs[:remaining]
            log.info(
                "Validating %d SQL statements with DB pool size %d",
                len(batch),
                self.pool_size,
            )
            validated = await asyncio.gather(*[_validate_pair(p) for p in batch])
            results.extend(validated)

            messages.append(msg_dict)
            attempts += 1
            if len(results) >= k:
                break
            if attempts < max_attempts:
                remaining = max(0, min(api_count, k - len(results)))
                log.info(
                    "Requesting %d additional pairs (%d/%d remaining)",
                    remaining,
                    k - len(results),
                    k,
                )
                follow = {
                    "role": "user",
                    "content": f"Generate {remaining} more question-SQL pairs about the table.",
                }
                messages.append(follow)

        if len(results) < k:
            log.warning(
                "Only %d/%d single-table pairs produced after %d attempts",
                len(results),
                k,
                attempts,
            )

        return JobResult(task.get("question", ""), "", results[:k])

    async def _run_joins_async(self, task: NLTask) -> JobResult:
        """Generate NL/SQL pairs that join multiple tables."""

        from .join_pool import JoinPool
        from .sql_validator import SQLValidator as ValCls

        from functools import partial

        parallel = int(self.phase_cfg.get("parallelism", 1))
        if hasattr(self.client, "set_parallelism"):
            self.client.set_parallelism(parallel)
        log.info("Assigned %d concurrent OpenAI sessions", parallel)
        self.pool_size = limit_pool_size(parallel * self.tasks_parallelism, self.pool_size)
        log.info(
            "DB pool size adjusted to %d for %d workers across %d tasks",
            self.pool_size,
            parallel,
            self.tasks_parallelism,
        )

        pool = JoinPool(
            self.schema,
            task.get("metadata", {}),
            partial(ValCls, pool_size=self.pool_size),
            self.writer,
            self.critic,
            self.client,
            self.pool_size,
        )
        pairs = await pool.generate()

        validation_sem = asyncio.Semaphore(self.pool_size)
        n_rows = int(self.phase_cfg.get("n_rows", 5))

        async def _validate_pair(p: dict[str, str]) -> dict[str, str]:
            sql = _clean_sql(p.get("sql", ""))
            if not sql:
                p["sql"] = "FAIL"
                return p
            async with validation_sem:
                ok, err = await asyncio.to_thread(self.validator.check, sql)
                if not ok:
                    log.warning("SQL validation failed: %s", err)
                    p["sql"] = "FAIL"
                else:
                    try:
                        await asyncio.to_thread(self.writer.fetch, sql, n_rows)
                        p["sql"] = sql
                    except Exception as err:
                        log.warning("Execution failed for %s: %s", sql, err)
                        p["sql"] = "FAIL"
            return p

        pairs = await asyncio.gather(*[_validate_pair(p) for p in pairs])

        return JobResult(task.get("question", ""), "", list(pairs))

    async def _run_complex_sqls_async(self, task: NLTask) -> JobResult:
        """Generate NL/SQL pairs joining multiple tables with GPT-selected sets."""

        from .complex_sql_pool import ComplexSqlPool
        from .sql_validator import SQLValidator as ValCls

        from functools import partial

        parallel = int(self.phase_cfg.get("parallelism", 1))
        self.pool_size = limit_pool_size(parallel * self.tasks_parallelism, self.pool_size)
        log.info(
            "DB pool size adjusted to %d for %d workers across %d tasks (complex)",
            self.pool_size,
            parallel,
            self.tasks_parallelism,
        )
        pool = ComplexSqlPool(
            self.schema,
            task.get("metadata", {}),
            partial(ValCls, pool_size=self.pool_size),
            self.writer,
            self.critic,
            self.client,
        )
        pairs = await pool.generate()
        checked: list[dict[str, str]] = []
        n_rows = int(self.phase_cfg.get("n_rows", 5))
        for p in pairs:
            sql = _clean_sql(p.get("sql", ""))
            ok, err = await asyncio.to_thread(self.validator.check, sql)
            if not ok:
                log.warning("Validation failed for %s: %s", sql, err)
                checked.append({"question": p.get("question", ""), "sql": "FAIL"})
                continue
            try:
                self.writer.fetch(sql, n_rows)
                checked.append({"question": p.get("question", ""), "sql": sql})
                log.info("SQL validated successfully: %s", sql)
            except Exception as err:
                log.warning("Execution failed for %s: %s", sql, err)
                checked.append({"question": p.get("question", ""), "sql": "FAIL"})

        return JobResult(task.get("question", ""), "", checked)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @log_call
    async def _tool_generate_sql(self, nl_question: str) -> Dict[str, str]:
        """LLM tool: generate SQL for ``nl_question`` and echo it back."""

        def _detect_tables(question: str) -> list[str]:
            lower = question.lower()
            return [t for t in self.schema if t.lower() in lower]

        tables = _detect_tables(nl_question)
        subset = {t: self.schema[t] for t in tables} if tables else self.schema
        cfg = dict(self.phase_cfg)
        if cfg.get("use_sample_rows") and tables:
            n = int(cfg.get("n_rows", 5))
            rows: dict[str, list[dict]] = {}
            for t in tables:
                try:
                    rows[t] = self.writer.fetch(f"SELECT * FROM {t} LIMIT {n}", n)
                except Exception as err:
                    log.warning("Failed fetching sample rows for %s: %s", t, err)
            if rows:
                cfg["sample_rows"] = rows
        prompt_obj = build_prompt(nl_question, subset, cfg)
        log.info("Generating SQL for question: %s", nl_question)
        if isinstance(prompt_obj, list):
            messages = prompt_obj
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_obj},
            ]
        sql = (await self.client.arun_jobs([messages]))[0]
        sql = sql.strip().strip("`")
        sql = re.sub(r"(?i)^sql\s*", "", sql)
        return {"question": nl_question, "sql": _clean_sql(sql)}

    @log_call
    async def _tool_validate_sql(self, sql: str) -> Dict[str, Any]:
        """LLM tool: validate SQL via :class:`SQLValidator`."""
        log.info("Validating SQL: %s", sql)
        ok, err = await asyncio.to_thread(self.validator.check, sql)
        return {"ok": ok, "error": err}

    @log_call
    async def _tool_critic(self, sql: str) -> Dict[str, Any]:
        """LLM tool: run the critic."""
        log.info("Running critic on SQL")
        result = await self.critic.areview("", sql, _schema_as_markdown(self.schema))
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
    async def run_task(self, task: NLTask) -> JobResult:
        """Process ``task`` letting the LLM drive via tools.

        Args:
            task: Task dictionary from :func:`load_tasks`.

        Returns:
            Final :class:`JobResult` for the task.
        """

        self.phase_cfg = task.get("metadata", {})

        if task.get("phase") == "schema_docs":
            return await self._run_schema_docs_async(task)
        if task.get("phase") == "schema_relationship":
            return await self._run_schema_relationship_async(task)
        if task.get("phase") == "single_table" and task.get("metadata", {}).get(
            "count"
        ):
            return await self._run_single_table_async(task)
        if task.get("phase") == "joins":
            return await self._run_joins_async(task)
        if task.get("phase") == "complex_sqls":
            return await self._run_complex_sqls_async(task)

        base_content = (
            "You are a data-engineer agent. "
            "Here is the schema:\n"
            f"{_schema_as_markdown(self.schema)}"
        )
        if task.get("phase") == "builtins":
            base_content += (
                " Return a JSON object with 'question' and 'sql'. "
                "Come up with a question that demonstrates the given builtin function "
                "and use the tools to generate the SQL."
            )
        else:
            base_content += " Return a JSON object with only a 'sql' field containing the valid query."

        messages = [
            {
                "role": "system",
                "content": base_content,
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
            msg = (
                await self.client.arun_jobs(
                    [messages], tools=tools, return_message=True
                )
            )[0]
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                messages.append(msg if isinstance(msg, dict) else msg.model_dump())
                for call in tool_calls:
                    fn = self._tool_map.get(call.function.name)
                    args = json.loads(call.function.arguments or "{}")
                    log.info("Invoking tool %s with args %s", call.function.name, args)
                    result = fn(**args)
                    if inspect.isawaitable(result):
                        result = await result
                    log.info("Tool %s returned %s", call.function.name, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                continue

            content = msg.get("content") if isinstance(msg, dict) else msg.content
            try:
                data = json.loads(content or "{}")
            except Exception:
                data = {"sql": content or "", "rows": []}
            sql = _clean_sql(data.get("sql", ""))
            question = data.get("question", task["question"])
            result = JobResult(question, sql, data.get("rows", []))
            ok, err = await asyncio.to_thread(self.validator.check, result.sql)
            if not ok:
                log.warning("SQL validation failed: %s", err)
                result.sql = "FAIL"
                result.rows = []
            return result

    @log_call
    async def _run_tasks_async(
        self, tasks: List[NLTask], run_version: str | None, parallelism: int
    ) -> List[JobResult]:
        """Process many tasks concurrently."""

        results: list[JobResult | None] = [None] * len(tasks)
        cleared: set[str] = set()
        dedup: dict[str, set] = {}
        sem = asyncio.Semaphore(max(parallelism, 1))

        async def _runner(idx: int, t: NLTask) -> None:
            async with sem:
                total = len(tasks)
                log.info("Running task %d/%d: %s", idx + 1, total, t.get("question"))
                res = await self.run_task(t)
                results[idx] = res
                log.info("Completed task %d/%d", idx + 1, total)

                out_dir = t.get("metadata", {}).get("dataset_output_file_dir")
                if out_dir:
                    file_name = "dataset.jsonl"
                    if run_version:
                        file_name = f"dataset_{run_version}.jsonl"
                    path = os.path.join(out_dir, file_name)
                    async with self._write_lock:
                        if path not in cleared:
                            os.makedirs(out_dir, exist_ok=True)
                            open(path, "w").close()
                            cleared.add(path)
                        if path not in dedup:
                            dedup[path] = set()
                        phase = t.get("phase")
                        if phase in {"schema_docs", "schema_relationship"}:
                            for pair in res.rows:
                                if (
                                    phase == "schema_relationship"
                                    and "confidence" in pair
                                ):
                                    pair = {
                                        k: v
                                        for k, v in pair.items()
                                        if k != "confidence"
                                    }
                                key = (pair.get("question"), pair.get("answer"))
                                if key not in dedup[path]:
                                    self.writer.append_jsonl(pair, path)
                                    dedup[path].add(key)
                            log.info("Wrote schema QA pairs to %s", path)
                        elif (
                            phase in {"single_table", "joins", "complex_sqls"}
                            and res.rows
                        ):
                            for pair in res.rows:
                                sql = _clean_sql(pair.get("sql", ""))
                                if sql == "FAIL":
                                    log.info("Skipping failed pair for %s", path)
                                    continue
                                key = sql
                                if key not in dedup[path]:
                                    row = {
                                        "question": pair.get("question", ""),
                                        "sql": sql,
                                    }
                                    if t.get("metadata", {}).get("tag_schema_json"):
                                        tables = self._extract_tables(sql)
                                        log.info(
                                            "Tagging schema for tables: %s", tables
                                        )
                                        row["schema"] = self._schema_subset_json(tables)
                                    self.writer.append_jsonl(row, path)
                                    dedup[path].add(key)
                            log.info("Wrote %s pairs to %s", phase, path)
                        else:
                            if res.sql == "FAIL":
                                log.info("Skipping failed pair for %s", path)
                            else:
                                if phase == "single_table":
                                    key = res.sql
                                else:
                                    key = (res.question, res.sql)
                                if key not in dedup[path]:
                                    row = {"question": res.question, "sql": res.sql}
                                    if t.get("metadata", {}).get("tag_schema_json"):
                                        tables = self._extract_tables(res.sql)
                                        log.info(
                                            "Tagging schema for tables: %s", tables
                                        )
                                        row["schema"] = self._schema_subset_json(tables)
                                    self.writer.append_jsonl(row, path)
                                    dedup[path].add(key)
                                    log.info("Wrote dataset entry to %s", path)
                                else:
                                    log.info("Skipped duplicate pair for %s", path)

        await asyncio.gather(*[_runner(i, t) for i, t in enumerate(tasks)])
        log.info("All %d tasks completed", len(tasks))
        log.info(
            "Token usage in=%d out=%d requests=%d",
            getattr(self.client, "tokens_in", 0),
            getattr(self.client, "tokens_out", 0),
            getattr(self.client, "request_count", 0),
        )
        return [r for r in results if r is not None]

    @log_call
    def run_tasks(
        self, tasks: List[NLTask], run_version: str | None = None
    ) -> List[JobResult]:
        """Public wrapper around the async task runner.

        Args:
            tasks: Sequence of tasks to run.
            run_version: Optional suffix for dataset files.

        Returns:
            List of :class:`JobResult` objects in the same order.
        """

        if not tasks:
            return []

        parallelism = max(
            int(t.get("metadata", {}).get("parallelism", 1)) for t in tasks
        )
        env_par = os.getenv("DG_PARALLELISM")
        if env_par:
            parallelism = max(parallelism, int(env_par))
        self.tasks_parallelism = parallelism
        return asyncio.run(self._run_tasks_async(tasks, run_version, parallelism))
