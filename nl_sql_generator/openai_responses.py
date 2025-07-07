"""Async batching client for the OpenAI Responses API.

This helper wraps :class:`openai.AsyncOpenAI` to manage request concurrency,
budget tracking and retries.

Example:
    >>> client = ResponsesClient(model="gpt-4o-mini-resp", budget_usd=1.0)
    >>> resp = client.run_jobs([[{"role": "user", "content": "Say hi"}]])
    >>> print(resp[0])
"""

from __future__ import annotations

__all__ = ["ResponsesClient", "Usage"]

import asyncio
import os
from dataclasses import dataclass
from typing import Any, List

from openai import AsyncOpenAI, OpenAI
import logging

log = logging.getLogger(__name__)


@dataclass
class Usage:
    """Track token usage and cost.

    Attributes:
        input_tokens: Number of prompt tokens used.
        output_tokens: Number of completion tokens returned.
        cost_usd: Total estimated cost in USD.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, inp: int, out: int, cost: float) -> None:
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Return USD cost for the given token counts.

    Args:
        input_tokens: Prompt tokens sent.
        output_tokens: Completion tokens received.
        model: Model name for pricing.

    Returns:
        Estimated USD cost based on static pricing.
    """
    # Hard-coded pricing for gpt-4o-mini-resp
    in_rate = 0.005 / 1000
    out_rate = 0.015 / 1000
    return input_tokens * in_rate + output_tokens * out_rate


class ResponsesClient:
    """Thin wrapper around the OpenAI Responses API.

    Attributes:
        model: The chat model used for completions.
        budget_usd: Maximum spend in USD.
        usage: Aggregate token and cost tracking.
    """

    def __init__(self, model: str = "gpt-4o-mini-resp", budget_usd: float = 0.0) -> None:
        """Instantiate the client.

        Args:
            model: OpenAI model to use.
            budget_usd: Spending cap in USD.
        """

        self.model = model
        self.budget_usd = budget_usd
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_spent = 0.0
        self.usage = Usage()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY first")

        self._client = AsyncOpenAI(api_key=api_key)
        self._sync_client = OpenAI(api_key=api_key)
        self._max_parallel = int(os.getenv("OPENAI_MAX_PARALLEL", "5"))
        self._sem = asyncio.Semaphore(self._max_parallel)
        self._lock = asyncio.Lock()

    def run_jobs(
        self,
        messages_list: List[List[dict]],
        stream: bool = False,
        tools: List[dict] | None = None,
        return_message: bool = False,
    ) -> List[Any]:
        """Execute a batch of message lists and return the responses.

        Args:
            messages_list: List of conversations to send.
            stream: Whether to stream tokens.
            tools: Optional tools definition for tool-calling models.
            return_message: If ``True`` return the full message objects.

        Returns:
            List of responses or messages in the same order as the input.
        """

        async def runner() -> List[str]:
            tasks = [
                asyncio.create_task(self._worker(m, stream, tools, return_message))
                for m in messages_list
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(runner())

    async def acomplete(
        self,
        messages: List[dict],
        stream: bool = False,
        tools: List[dict] | None = None,
        return_message: bool = False,
    ) -> Any:
        async with self._sem:
            return await self._request_with_retry(messages, stream, tools, return_message)

    async def _worker(
        self,
        messages: List[dict],
        stream: bool,
        tools: List[dict] | None,
        return_message: bool,
    ) -> Any:
        """Execute a single request with concurrency control.

        Args:
            messages: Conversation to send.
            stream: Whether to stream tokens.
            tools: Optional tools definition.
            return_message: If ``True`` return the message object.

        Returns:
            Response text or message.
        """
        async with self._sem:
            return await self._request_with_retry(messages, stream, tools, return_message)

    async def _request_with_retry(
        self,
        messages: List[dict],
        stream: bool,
        tools: List[dict] | None,
        return_message: bool,
    ) -> Any:
        """Call the API with exponential backoff retry logic.

        Returns:
            Response text or message depending on ``return_message``.
        """
        delay = 1.0
        # Convert any OpenAI message objects to plain dicts for clean logging
        serializable = [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]

        # Log the prompt without JSON escape sequences
        lines = [f"{m.get('role')}: {m.get('content', '')}" for m in serializable]
        log.info("Prompt:\n%s", "\n".join(lines))
        for attempt in range(5):
            log.info("OpenAI call attempt %d using model %s", attempt + 1, self.model)
            try:
                if stream:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto" if tools else None,
                        stream=True,
                    )
                    text = ""
                    async for chunk in response:
                        text += chunk.choices[0].delta.content or ""
                    usage = response.usage
                else:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto" if tools else None,
                    )
                    message = response.choices[0].message
                    text = message.content or ""
                    usage = response.usage

                in_tok = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
                out_tok = (
                    getattr(
                        usage,
                        "output_tokens",
                        getattr(usage, "completion_tokens", 0),
                    )
                    or 0
                )
                est_cost = _estimate_cost(in_tok, out_tok, self.model)

                if self.cost_spent + est_cost > self.budget_usd:
                    log.error(
                        "Budget exceeded: cost=$%.4f spent=$%.4f budget=$%.4f",
                        est_cost,
                        self.cost_spent,
                        self.budget_usd,
                    )
                    raise RuntimeError(
                        (
                            "Budget exceeded: spent=$%.4f + est=$%.4f > budget=$%.4f"
                        )
                        % (self.cost_spent, est_cost, self.budget_usd)
                    )

                async with self._lock:
                    self.tokens_in += in_tok
                    self.tokens_out += out_tok
                    self.cost_spent += est_cost
                    self.usage.add(in_tok, out_tok, est_cost)
                log.info(
                    "OpenAI response: in=%d out=%d cost=$%.4f budget_left=$%.4f",
                    in_tok,
                    out_tok,
                    est_cost,
                    self.remaining_budget(),
                )
                if return_message:
                    if stream:
                        msg = {"role": "assistant", "content": text}
                        return msg
                    return message
                return text

            except Exception as err:
                log.warning("OpenAI attempt %d failed: %s", attempt + 1, err)
                if attempt == 4:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError("Failed to get response after retries")

    def remaining_budget(self) -> float:
        """Return remaining budget in USD.

        Returns:
            Remaining spendable budget.
        """

        return self.budget_usd - self.cost_spent


_default_client: ResponsesClient | None = None


async def acomplete(prompt: list[dict] | str, model: str | None = None) -> str:
    """Convenience wrapper using a global :class:`ResponsesClient`."""

    global _default_client
    model = model or "gpt-4o-mini-resp"
    if _default_client is None or _default_client.model != model:
        budget = float(os.getenv("OPENAI_BUDGET_USD", "0"))
        _default_client = ResponsesClient(model=model, budget_usd=budget)
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    return await _default_client.acomplete(messages)
