import json
from typing import Any, Dict, List

from .prompt_builder import build_schema_doc_prompt
from .openai_responses import ResponsesClient


class WorkerAgent:
    def __init__(
        self,
        schema: Dict[str, Any],
        cfg: Dict[str, Any],
        validator_cls,
        wid: int,
        client: ResponsesClient,
    ) -> None:
        self.schema = schema
        self.cfg = cfg
        self.validator = validator_cls()
        self.wid = wid
        self.client = client

    async def generate(self, k: int) -> List[Dict[str, str]]:
        prompt = build_schema_doc_prompt(self.schema, k=k)
        completion = await self.client.acomplete(prompt, model=self.cfg.get("openai_model"))
        pairs = [json.loads(line) for line in completion.strip().splitlines() if line.strip()]
        return pairs
