import json
from typing import Any, Dict, List


def _parse_pairs(text: str) -> List[Dict[str, str]]:
    """Return JSON objects from ``text`` one per valid line.

    The OpenAI responses occasionally include markdown fences or extra prose.
    This helper skips such lines and ignores JSON decode errors so that callers
    only receive well formed ``{"question": ..., "answer": ...}`` mappings.
    """

    pairs: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        # Drop common list prefixes like "-" or "1." and code fence ticks
        line = line.lstrip("-*0123456789. ").strip("`")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            pairs.append(obj)
    return pairs


from .prompt_builder import build_schema_doc_prompt  # noqa: E402
from .openai_responses import ResponsesClient  # noqa: E402


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
        pairs = _parse_pairs(completion)
        return pairs
