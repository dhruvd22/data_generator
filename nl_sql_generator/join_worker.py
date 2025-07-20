import logging
import json
import os
from datetime import datetime
import re
import asyncio
from typing import Any, Dict, List, Optional
from .openai_responses import ResponsesClient
from .prompt_builder import load_template_messages, _schema_as_markdown
from .worker_agent import _parse_pairs
from .autonomous_job import _clean_sql

log = logging.getLogger(__name__)


class JoinWorker:
    """LLM-powered worker that produces join queries for a table subset."""

    def __init__(
        self,
        schema: Dict[str, Any],
        cfg: Dict[str, Any],
        validator_cls,
        critic,
        writer,
        wid: int,
        client: ResponsesClient,
        pool_size: int | None = None,
    ) -> None:
        """Create a worker tied to a schema slice.

        Args:
            schema: Mapping of tables visible to this worker.
            cfg: Phase configuration with generation options.
            validator_cls: Callable returning a validator instance.
            critic: :class:`Critic` used to fix invalid SQL.
            writer: :class:`ResultWriter` for executing SQL.
            wid: Worker identifier used in logs.
            client: Shared :class:`ResponsesClient` for OpenAI calls.
        """

        self.schema = schema
        self.cfg = cfg
        if pool_size is None:
            pool_size = int(os.getenv("DB_COCURRENT_SESSION", "50"))
        self.pool_size = pool_size
        self.validator = validator_cls()
        self.critic = critic
        self.writer = writer
        self.wid = wid
        self.client = client

        self.chat_history: List[Dict[str, str]] = []
        self.chat_log_path: str | None = None
        self._chat_log_fh: Optional[Any] = None
        if self.cfg.get("enable_worker_chat_log"):
            os.makedirs("logs", exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.chat_log_path = os.path.join(
                "logs", f"join-worker-{self.wid}-{ts}.jsonl"
            )
            self._chat_log_fh = open(self.chat_log_path, "w", encoding="utf-8")
            log.info(
                "Join worker %d chat log enabled at %s", self.wid, self.chat_log_path
            )

    def _to_dict(self, msg: Any) -> Dict[str, Any]:
        """Return ``msg`` as a plain dictionary."""
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        return {
            "role": getattr(msg, "role", ""),
            "content": getattr(msg, "content", str(msg)),
        }

    def _write_chat(self, messages: List[Any]) -> None:
        """Append ``messages`` to the chat log file if enabled."""
        if not self._chat_log_fh:
            return
        for m in messages:
            json.dump(self._to_dict(m), self._chat_log_fh)
            self._chat_log_fh.write("\n")
        self._chat_log_fh.flush()

    def _join_table_count(self, sql: str) -> int:
        """Return number of tables referenced via FROM/JOIN clauses."""
        pattern = re.compile(r"\b(?:FROM|JOIN)\s+([\w\.\"`]+)", re.IGNORECASE)
        tables = set()
        for match in pattern.findall(sql):
            tbl = match.split()[0]
            tbl = tbl.strip('`"')
            # strip schema prefix if present
            if "." in tbl:
                tbl = tbl.split(".")[-1]
            tables.add(tbl)
        return len(tables)

    async def generate(self, k: int) -> List[Dict[str, str]]:
        """Return ``k`` validated join questions for this worker's tables."""

        log.info("Worker %d generating %d join pairs", self.wid, k)
        api_count = int(self.cfg.get("api_answer_count", k))
        max_attempts = int(self.cfg.get("max_attempts", 6))
        extra = {
            "count": min(api_count, k),
            "min_joins": self.cfg.get("min_joins", 2),
        }
        if "sample_rows" in self.cfg:
            extra["sample_rows"] = self.cfg["sample_rows"]
        messages = load_template_messages(
            "join_sql_template.txt", self.schema, "", extra
        )
        if self.chat_log_path:
            self.chat_history.extend(messages)
            self._write_chat(messages)
        log.info(
            "Worker %d sending prompt with tables %s",
            self.wid,
            list(self.schema),
        )
        total: List[Dict[str, str]] = []
        attempts = 0
        n_rows = int(self.cfg.get("n_rows", 5))
        min_joins = int(self.cfg.get("min_joins", 2))

        validation_sem = asyncio.Semaphore(self.pool_size)

        async def _validate_pair(p: Dict[str, str]) -> Dict[str, str] | None:
            q = p.get("question", "")
            sql = _clean_sql(p.get("sql", ""))
            attempts_v = 0
            ok = False
            while attempts_v <= 2:
                async with validation_sem:
                    ok, err = await asyncio.to_thread(self.validator.check, sql)
                if ok:
                    break
                log.warning("Worker %d validation failed: %s", self.wid, err)
                if not self.critic or attempts_v >= 2:
                    return None
                fix = await self.critic.areview(
                    q, sql, _schema_as_markdown(self.schema)
                )
                sql = _clean_sql(fix.get("fixed_sql", sql))
                attempts_v += 1
            if not ok or self._join_table_count(sql) < min_joins:
                return None
            try:
                await asyncio.to_thread(self.writer.fetch, sql, n_rows)
                return {"question": q, "sql": sql}
            except Exception as err:
                log.warning("Worker %d execution failed: %s", self.wid, err)
                return None

        while len(total) < k and attempts < max_attempts:
            log.info(
                "API request %d with history length %d",
                attempts + 1,
                len(messages),
            )
            message = await self.client.acomplete(messages, return_message=True)
            msg_dict = self._to_dict(message)
            if self.chat_log_path:
                self.chat_history.append(msg_dict)
                self._write_chat([msg_dict])
            pairs = _parse_pairs(msg_dict.get("content", ""))
            log.info(
                "Received %d candidate pairs (%d/%d total)",
                len(pairs),
                len(total),
                k,
            )

            remaining = k - len(total)
            batch = pairs[:remaining]
            log.info(
                "Validating %d SQL statements with DB pool size %d",
                len(batch),
                self.pool_size,
            )
            validated = await asyncio.gather(*[_validate_pair(p) for p in batch])
            for v in validated:
                if v:
                    total.append(v)

            messages.append(msg_dict)
            attempts += 1
            if len(total) >= k:
                break
            if attempts < max_attempts:
                remaining = max(0, min(api_count, k - len(total)))
                follow = {
                    "role": "user",
                    "content": f"Generate {remaining} more question-SQL pairs about the tables.",
                }
                messages.append(follow)
                if self.chat_log_path:
                    self.chat_history.append(follow)
                    self._write_chat([follow])

        log.info("Worker %d produced %d valid pairs", self.wid, len(total))
        if self._chat_log_fh:
            self._chat_log_fh.close()
            log.info(
                "Join worker %d chat history saved to %s",
                self.wid,
                self.chat_log_path,
            )
        return total[:k]
