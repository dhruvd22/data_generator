"""Convenience re-exports for the public API.

Importing this package exposes commonly used classes and helpers so consumers
can simply ``from nl_sql_generator import AutonomousJob`` without digging into
submodules.

Example:
    >>> from nl_sql_generator import AutonomousJob ( Clean up rate limit for child worker agents )
    >>> job = AutonomousJob(schema)
"""


from importlib import import_module

def _load(name: str):
    module_map = {
        "AutonomousJob": "autonomous_job",
        "JobResult": "autonomous_job",
        "NLTask": "input_loader",
        "load_tasks": "input_loader",
        "Settings": "logger",
        "init_logger": "logger",
        "log_call": "logger",
        "SchemaLoader": "schema_loader",
        "TableInfo": "schema_loader",
    }
    mod = import_module(f".{module_map[name]}", __name__)
    return getattr(mod, name)


__all__ = [
    "AutonomousJob",
    "JobResult",
    "NLTask",
    "load_tasks",
    "Settings",
    "init_logger",
    "log_call",
    "SchemaLoader",
    "TableInfo",
]


def __getattr__(name: str):
    if name in __all__:
        return _load(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
