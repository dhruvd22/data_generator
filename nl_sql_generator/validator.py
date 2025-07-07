from __future__ import annotations

from typing import Tuple


class NoOpValidator:
    """Validator that always succeeds."""

    def check(self, sql: str) -> Tuple[bool, str | None]:
        return True, None
