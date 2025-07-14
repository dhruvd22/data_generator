"""Per-worker agent that produces schema question/answer pairs."""

import json
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def _parse_pairs(text: str) -> List[Dict[str, str]]:
    """Return valid JSON objects from ``text``.

    The OpenAI API often returns one JSON object per line but may also
    respond with a JSON array or wrap the pairs inside another object. This
    helper tries a few strategies to recover the pairs while skipping any
    malformed lines or surrounding markdown/numbering noise.
    """

    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        # Drop common list prefixes like "-" or "1." and code fence ticks
        line = line.lstrip("-*0123456789. ").strip("`")
        # Skip chatter like "Here are the pairs:" which breaks JSON parsing
        if not line or line[0] not in "[{]}":
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    pairs: List[Dict[str, str]] = []

    # First try parsing the cleaned text as a single JSON structure
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "pairs" in obj and isinstance(obj["pairs"], list):
            obj = obj["pairs"]
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    pairs.append(item)
        elif isinstance(obj, dict):
            pairs.append(obj)
    except json.JSONDecodeError:
        # Fallback to line-by-line parsing
        for line in lines:
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
        max_attempts = int(self.cfg.get("max_attempts", 6))
        log.info(
            "Worker %d generating %d pairs using api_answer_count=%d",
            self.wid,
            k,
            api_count,
        )

        first_request = min(api_count, k)
        messages: List[Dict[str, str]] = build_schema_doc_prompt(
            self.schema, k=first_request
        )
        total: List[Dict[str, str]] = []

        attempts = 0
        while len(total) < k and attempts < max_attempts:
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
            attempts += 1
            if len(total) >= k:
                break
            if attempts < max_attempts:
                remaining = min(api_count, k - len(total))
                log.info(
                    "Worker %d received %d pairs, requesting %d more (%d/%d remaining)",
                    self.wid,
                    len(pairs),
                    remaining,
                    k - len(total),
                    k,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Generate {remaining} more question-answer pairs about the schema.",
                    }
                )

        if len(total) < k:
            log.warning(
                "Worker %d produced only %d/%d pairs after %d attempts",
                self.wid,
                len(total),
                k,
                attempts,
            )
        else:
            log.info("Worker %d produced %d pairs", self.wid, len(total))
        return total[:k]
