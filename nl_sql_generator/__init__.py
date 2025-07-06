"""Convenience re-exports for the public API.

Importing this package exposes commonly used classes and helpers so consumers
can simply ``from nl_sql_generator import AutonomousJob`` without digging into
submodules.

Example:
    >>> from nl_sql_generator import AutonomousJob
    >>> job = AutonomousJob(schema)
"""

from .autonomous_job import AutonomousJob, JobResult
from .input_loader import NLTask, load_tasks
from .logger import Settings, init_logger, log_call

__all__ = [
    "AutonomousJob",
    "JobResult",
    "NLTask",
    "load_tasks",
    "Settings",
    "init_logger",
    "log_call",
]
