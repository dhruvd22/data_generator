import logging
import json
import os
from datetime import datetime
import re
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

    def _write_chat(self, messages: List[Dict[str, str]]) -> None:
        """Append ``messages`` to the chat log file if enabled."""
        if not self._chat_log_fh:
            return
        for m in messages:
            json.dump(m, self._chat_log_fh)
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
        extra = {
            "count": k,
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
        message = await self.client.acomplete(messages, return_message=True)
        if self.chat_log_path:
            self.chat_history.append(message)
            self._write_chat([message])
        pairs = _parse_pairs(message.get("content", ""))
        results: List[Dict[str, str]] = []
        min_joins = int(self.cfg.get("min_joins", 2))
        for p in pairs:
            q = p.get("question", "")
            sql = _clean_sql(p.get("sql", ""))
            attempts = 0
            while attempts <= 2:
                ok, err = self.validator.check(sql)
                if ok:
                    break
                log.warning("Worker %d validation failed: %s", self.wid, err)
                if attempts >= 2:
                    sql = None
                    break
                fix = await self.critic.areview(
                    q, sql, _schema_as_markdown(self.schema)
                )
                sql = _clean_sql(fix.get("fixed_sql", sql))
                attempts += 1
            if sql and ok and self._join_table_count(sql) >= min_joins:
                try:
                    self.writer.fetch(sql, int(self.cfg.get("n_rows", 5)))
                    results.append({"question": q, "sql": sql})
                except Exception as err:
                    log.warning("Worker %d execution failed: %s", self.wid, err)
        log.info("Worker %d produced %d valid pairs", self.wid, len(results))
        if self._chat_log_fh:
            self._chat_log_fh.close()
            log.info(
                "Join worker %d chat history saved to %s",
                self.wid,
                self.chat_log_path,
            )
        return results
