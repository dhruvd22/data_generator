"""Per-worker agent that produces schema question/answer pairs."""

import json
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def _parse_pairs(text: str) -> List[Dict[str, str]]:
    """Return valid JSON objects from ``text`` one per line.

    The OpenAI API sometimes prefixes responses with formatting such as
    Markdown fences or bullet lists.  This helper filters such noise and
    quietly skips lines that fail ``json.loads`` so that callers only see
    well-formed ``{"question": ..., "answer": ...}`` dictionaries.
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
    log.debug("Parsed %d QA pairs", len(pairs))
    return pairs


from .prompt_builder import build_schema_doc_prompt  # noqa: E402
from .openai_responses import ResponsesClient  # noqa: E402


class WorkerAgent:
    """Generate schema question/answer pairs using a single OpenAI worker."""

    def __init__(
        self,
        schema: Dict[str, Any],
        cfg: Dict[str, Any],
        validator_cls,
        wid: int,
        client: ResponsesClient,
    ) -> None:
        """Create a worker.

        Args:
            schema: Table metadata subset for this worker.
            cfg: Phase configuration dictionary.
            validator_cls: Callable returning a validator instance.
            wid: Worker identifier used in logs.
            client: Shared :class:`ResponsesClient`.
        """

        self.schema = schema
        self.cfg = cfg
        self.validator = validator_cls()
        self.wid = wid
        self.client = client

    async def generate(self, k: int) -> List[Dict[str, str]]:
        """Return ``k`` Q&A pairs generated from this worker's schema slice.

        The initial request uses ``api_answer_count`` from the configuration to
        limit the number of pairs returned.  The conversation history is then
        reused with follow-up prompts requesting the same number of pairs until
        ``k`` total pairs have been collected.  The schema definition (and any
        sample rows if supplied) is included only in the very first request so
        it does not get appended every time the chat history is resent.
        """

        api_count = int(self.cfg.get("api_answer_count", k))
        log.info(
            "Worker %d generating %d pairs using api_answer_count=%d",
            self.wid,
            k,
            api_count,
        )

        messages: List[Dict[str, str]] = build_schema_doc_prompt(
            self.schema, k=api_count
        )
        total: List[Dict[str, str]] = []

        while len(total) < k:
            msg = await self.client.acomplete(
                messages, return_message=True, model=self.cfg.get("openai_model")
            )
            text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            pairs = _parse_pairs(text)
            for p in pairs:
                total.append(p)
                if len(total) >= k:
                    break
            messages.append(msg)
            if len(total) < k:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Generate {api_count} more question-answer pairs about the schema.",
                    }
                )

        log.info("Worker %d produced %d pairs", self.wid, len(total))
        return total[:k]
