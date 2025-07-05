"""CLI entrypoints for the NL→SQL generator."""

__all__ = ["cli", "gen"]

import argparse
import json
import logging
import os
import random
import re
import sys

import openai
import yaml

from nl_sql_generator.prompt_builder import build_prompt
from nl_sql_generator.sql_validator import SQLValidator
from nl_sql_generator.schema_loader import SchemaLoader
from nl_sql_generator.autonomous_job import AutonomousJob
from nl_sql_generator.logger import init_logger
import typer

log = init_logger()
app = typer.Typer(add_completion=False)


def cli() -> None:
    """Original quick-start demo using argparse (kept for compatibility)."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    schema = SchemaLoader.load_schema()

    log.info("Loaded %d tables from DB\n---", len(schema))
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Set OPENAI_API_KEY first")

    validator = SQLValidator()

    for phase in cfg["phases"][:1]:  # just the first phase for now
        log.info("\n=== Phase: %s ===", phase['name'])
        question = "Give me the total number of patients"  # stub NL question
        prompt = build_prompt(question, schema, phase)

        response = openai.chat.completions.create(
            model=cfg["openai_model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw_sql = response.choices[0].message.content.strip().strip("`")
        sql = re.sub(r"(?i)^sql\s*", "", raw_sql)

        ok, err = validator.check(sql)
        log.info("📝 NL question: %s", question)
        log.info("🏗️  Generated SQL: %s", sql)
        if ok:
            log.info("✅ Valid")
        else:
            log.info("❌ Invalid: %s", err)
        break  # one sample is enough for this milestone


@app.command()
def gen(file: str, config: str = "config.yaml") -> None:
    """Generate SQL and rows for questions listed in ``file``."""
    cfg = yaml.safe_load(open(config))
    schema = SchemaLoader.load_schema()

    questions = [q.strip() for q in open(file, "r", encoding="utf-8") if q.strip()]
    job = AutonomousJob(schema, cfg.get("phases", [{}])[0])
    results = job.run_async(questions)
    for r in results:
        typer.echo(json.dumps({"question": r.question, "sql": r.sql, "rows": r.rows}))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gen":
        # Delegate to Typer when subcommand is provided
        app()
    else:
        cli()
