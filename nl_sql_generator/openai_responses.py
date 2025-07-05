"""Async batching client for the OpenAI Responses API."""

from __future__ import annotations

import os
from typing import List

import openai


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

        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("Set OPENAI_API_KEY first")

    def run_jobs(self, messages_list: List[List[dict]]) -> List[str]:
        """Execute a batch of message lists and return the responses."""
        outputs: List[str] = []
        for messages in messages_list:
            resp = openai.responses.create(
                model=self.model,
                input=messages,
                stream=False,
            )
            in_tok = resp.usage.input_tokens or 0
            out_tok = resp.usage.output_tokens or 0
            est_cost = _estimate_cost(in_tok, out_tok, self.model)
            if self.cost_spent + est_cost > self.budget_usd:
                raise RuntimeError("Budget exceeded")
            self.tokens_in += in_tok
            self.tokens_out += out_tok
            self.cost_spent += est_cost
            outputs.append(resp.choices[0].message.content)
        return outputs

    def remaining_budget(self) -> float:
        """Return remaining budget in USD."""
        return self.budget_usd - self.cost_spent
