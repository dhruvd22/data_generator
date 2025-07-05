"""Rich-based logging utilities.

This module provides ``init_logger`` to configure a logger with a pretty
console output, a rotating log file and an optional JSONL event stream.
The ``log_call`` decorator can be used to trace function execution.

Example
-------
>>> from nl_sql_generator.logger import init_logger, Settings, log_call
>>> log = init_logger(Settings(events_file="logs/events.jsonl"))
>>> @log_call
... def add(a, b):
...     return a + b
>>> add(1, 2)
3
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Optional

from rich.logging import RichHandler


@dataclass
class Settings:
    """Optional logger settings."""

    events_file: Optional[str] = None


def init_logger(settings: Settings | None = None) -> logging.Logger:
    """Configure and return the package logger.

    Parameters
    ----------
    settings:
        Optional :class:`Settings` to enable the JSONL event stream.
    """

    settings = settings or Settings()
    logger = logging.getLogger("nl_sql_generator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    # console handler
    console = RichHandler(rich_tracebacks=True)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # file handler
    os.makedirs("logs", exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"logs/run-{ts}.log", encoding="utf-8")
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # optional JSONL events handler
    if settings.events_file:
        json_handler = logging.FileHandler(settings.events_file, encoding="utf-8")

        class _JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                data = {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    data["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(data)

        json_handler.setFormatter(_JSONFormatter())
        json_handler.setLevel(logging.INFO)
        logger.addHandler(json_handler)

    return logger


def log_call(func: Callable) -> Callable:
    """Decorator that logs function calls and execution time."""

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logging.getLogger(func.__module__)
            log.debug("%s args=%r kwargs=%r", func.__name__, args, kwargs)
            start = perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = perf_counter() - start
                log.debug("%s completed in %.2fs", func.__name__, elapsed)
                return result
            except Exception:
                elapsed = perf_counter() - start
                log.exception("%s failed after %.2fs", func.__name__, elapsed)
                raise
        return async_wrapper
    else:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logging.getLogger(func.__module__)
            log.debug("%s args=%r kwargs=%r", func.__name__, args, kwargs)
            start = perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = perf_counter() - start
                log.debug("%s completed in %.2fs", func.__name__, elapsed)
                return result
            except Exception:
                elapsed = perf_counter() - start
                log.exception("%s failed after %.2fs", func.__name__, elapsed)
                raise
        return wrapper
