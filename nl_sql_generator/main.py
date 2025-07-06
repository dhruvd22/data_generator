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
