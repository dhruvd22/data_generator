"""Async batching client for the OpenAI Responses API."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI, OpenAI


@dataclass
class Usage:
    """Track token usage and cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, inp: int, out: int, cost: float) -> None:
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd += cost


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Return USD cost for the given token counts."""
    # Hard-coded pricing for gpt-4o-mini-resp
    in_rate = 0.005 / 1000
    out_rate = 0.015 / 1000
    return input_tokens * in_rate + output_tokens * out_rate


class ResponsesClient:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(self, model: str = "gpt-4o-mini-resp", budget_usd: float = 0.0) -> None:
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

    def run_jobs(self, messages_list: List[List[dict]], stream: bool = False) -> List[str]:
        """Execute a batch of message lists and return the responses."""

        async def runner() -> List[str]:
            tasks = [asyncio.create_task(self._worker(m, stream)) for m in messages_list]
            return await asyncio.gather(*tasks)

        return asyncio.run(runner())

    async def _worker(self, messages: List[dict], stream: bool) -> str:
        async with self._sem:
            return await self._request_with_retry(messages, stream)

    async def _request_with_retry(self, messages: List[dict], stream: bool) -> str:
        delay = 1.0
        for attempt in range(5):
            try:
                if stream:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
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
                    )
                    text = response.choices[0].message.content
                    usage = response.usage

                in_tok = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
                out_tok = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0
                est_cost = _estimate_cost(in_tok, out_tok, self.model)

                if self.cost_spent + est_cost > self.budget_usd:
                    raise RuntimeError("Budget exceeded")

                self.tokens_in += in_tok
                self.tokens_out += out_tok
                self.cost_spent += est_cost
                self.usage.add(in_tok, out_tok, est_cost)
                return text

            except Exception:
                if attempt == 4:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError("Failed to get response after retries")

    def remaining_budget(self) -> float:
        """Return remaining budget in USD."""
        return self.budget_usd - self.cost_spent
