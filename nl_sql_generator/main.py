"""CLI entrypoints for the NL→SQL generator."""

__all__ = ["cli", "gen"]

import argparse
import json
import logging
import os
import sys

import yaml

from nl_sql_generator.input_loader import load_tasks
from nl_sql_generator.schema_loader import SchemaLoader
from nl_sql_generator.autonomous_job import AutonomousJob
from nl_sql_generator.logger import init_logger
from nl_sql_generator.openai_responses import ResponsesClient
from nl_sql_generator.sql_validator import SQLValidator
from nl_sql_generator.critic import Critic
from nl_sql_generator.writer import ResultWriter
import typer
from typer.core import TyperArgument
import click
import inspect

# ---------------------------------------------------------------------------
# Temporary compatibility patch for Click>=8.1 where ``Parameter.make_metavar``
# expects a ``ctx`` argument. Typer's ``TyperArgument`` implementation in
# 0.12.x does not accept this parameter which leads to ``TypeError`` when the
# CLI shows help or error messages on Python 3.12+. We override the method with
# a shim that ignores the ``ctx`` parameter if present.
# ---------------------------------------------------------------------------
if "ctx" in inspect.signature(TyperArgument.make_metavar).parameters:
    # Already compatible
    pass
else:
    def _patched_make_metavar(self, ctx=None):  # type: ignore[override]
        if self.metavar is not None:
            return self.metavar
        var = (self.name or "").upper()
        if not self.required:
            var = f"[{var}]"
        type_var = self.type.get_metavar(self)
        if type_var:
            var += f":{type_var}"
        if self.nargs != 1:
            var += "..."
        return var

    TyperArgument.make_metavar = _patched_make_metavar  # type: ignore[assignment]

# If Click's ``Parameter.make_metavar`` requires a ``ctx`` argument, wrap it so
# calls without ``ctx`` still work. Some versions of Typer call this method
# without arguments.
if "ctx" in inspect.signature(click.core.Parameter.make_metavar).parameters:
    _orig_param_make_metavar = click.core.Parameter.make_metavar

    def _param_make_metavar(self, ctx=None):  # type: ignore[override]
        ctx = ctx or click.Context(click.Command("_dummy"))
        return _orig_param_make_metavar(self, ctx)

    click.core.Parameter.make_metavar = _param_make_metavar  # type: ignore[assignment]

log = init_logger()
app = typer.Typer(add_completion=False)


def cli() -> None:
    """Original quick-start demo using argparse (kept for compatibility)."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if not args.config.lower().endswith((".yaml", ".yml")):
        raise SystemExit("Configuration must be a YAML file")

    cfg = yaml.safe_load(open(args.config))
    schema = SchemaLoader.load_schema()

    client = ResponsesClient(model=cfg["openai_model"], budget_usd=cfg["budget_usd"])
    job = AutonomousJob(
        schema,
        client=client,
        validator=SQLValidator(),
        critic=Critic(client=client),
        writer=ResultWriter(),
    )
    tasks = load_tasks(args.config, schema)
    for res in job.run_tasks(tasks[:1]):
        log.info(json.dumps({"question": res.question, "sql": res.sql}))


@app.command()
def gen(config: str = "config.yaml", stream: bool = False) -> None:
    """Generate SQL and result rows for tasks defined in ``config``."""

    if not config.lower().endswith((".yaml", ".yml")):
        raise typer.BadParameter("Configuration must be YAML")

    cfg = yaml.safe_load(open(config))
    schema = SchemaLoader.load_schema()

    client = ResponsesClient(model=cfg["openai_model"], budget_usd=cfg["budget_usd"])
    job = AutonomousJob(
        schema,
        client=client,
        validator=SQLValidator(),
        critic=Critic(client=client),
        writer=ResultWriter(),
    )

    tasks = load_tasks(config, schema)
    results = job.run_tasks(tasks)
    for r in results:
        typer.echo(json.dumps({"question": r.question, "sql": r.sql, "rows": r.rows}))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        # Delegate to Typer when subcommand is provided
        app()
    else:
        cli()
