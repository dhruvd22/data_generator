import logging
import json
from typing import Any, Dict, List
from .openai_responses import ResponsesClient
from .prompt_builder import load_template_messages, _schema_as_markdown
from .worker_agent import _parse_pairs

log = logging.getLogger(__name__)

class JoinWorker:
    def __init__(self, schema: Dict[str, Any], cfg: Dict[str, Any], validator_cls, critic, writer, wid: int, client: ResponsesClient) -> None:
        self.schema = schema
        self.cfg = cfg
        self.validator = validator_cls()
        self.critic = critic
        self.writer = writer
        self.wid = wid
        self.client = client

    async def generate(self, k: int) -> List[Dict[str, str]]:
        log.info("Worker %d generating %d join pairs", self.wid, k)
        extra = {
            "count": k,
            "min_joins": self.cfg.get("min_joins", 2),
        }
        if "sample_rows" in self.cfg:
            extra["sample_rows"] = self.cfg["sample_rows"]
        messages = load_template_messages("join_sql_template.txt", self.schema, "", extra)
        completion = await self.client.acomplete(messages)
        pairs = _parse_pairs(completion)
        results: List[Dict[str, str]] = []
        for p in pairs:
            q = p.get("question", "")
            sql = p.get("sql", "")
            attempts = 0
            while attempts <= 2:
                ok, err = self.validator.check(sql)
                if ok:
                    break
                log.warning("Worker %d validation failed: %s", self.wid, err)
                if attempts >= 2:
                    sql = None
                    break
                fix = await self.critic.areview(q, sql, _schema_as_markdown(self.schema))
                sql = fix.get("fixed_sql", sql)
                attempts += 1
            if sql and ok:
                try:
                    self.writer.fetch(sql, int(self.cfg.get("n_rows", 5)))
                    results.append({"question": q, "sql": sql})
                except Exception as err:
                    log.warning("Worker %d execution failed: %s", self.wid, err)
        log.info("Worker %d produced %d valid pairs", self.wid, len(results))
        return results
