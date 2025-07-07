# data_generator

**LLM-ready SQL data generator.** That automatically drives through various phases as per config.yaml and generates NL to PostgreSQL datasets. Each dataset entry contains only the natural language question and the validated SQL. The pipeline guards budget via an
OpenAI cost tracker and modular phases that you can swap out or extend. Use it
to produce high quality NL ⇢ SQL ⇢ answer triples for training or evaluation.

## Features

* 🗄️ Schema introspection from your database
* 🤖 GPT-driven prompt builder and critic loop
* ✅ `EXPLAIN`-based SQL validation
* 🔌 Plug-n-play phases for easy customisation

## Quick start

```bash
# clone & install
git clone https://github.com/dhruvd22/data_generator.git
cd data_generator
pip install -r nl_sql_generator/requirements.txt

# set credentials
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export OPENAI_API_KEY="sk-..."

# generate via CLI (tasks come from config.yaml)
python -m nl_sql_generator.main gen --config config.yaml
# run a single phase only
# python -m nl_sql_generator.main gen --config config.yaml --phase joins
# no separate questions file is needed

# or from Python
from nl_sql_generator import AutonomousJob, SchemaLoader, load_tasks
schema = SchemaLoader.load_schema()
job = AutonomousJob(schema)
tasks = load_tasks("config.yaml", phase="joins")
result = job.run_task(tasks[0])
print(result.sql, result.rows)
```

## 🔌 How it works

1. **InputLoader** – loads tasks from `config.yaml`
2. **AutonomousJob** – orchestrates the flow
3. **SchemaLoader** – pulls table metadata
4. **PromptBuilder** – crafts the few-shot prompt
5. **ResponsesClient** – queries OpenAI
6. **SQLValidator** – checks syntax via `EXPLAIN`
7. **Critic** – reviews and optionally fixes SQL
8. **Writer** – executes and fakes result rows

All configuration lives in `config.yaml`.

## Architecture overview

The generator is composed of small modules that interact through
`AutonomousJob`. Tasks are loaded from the configuration file and then flow
through a series of tool calls until a validated query and optional sample data
are produced.

```
config.yaml ──▶ InputLoader ──▶ AutonomousJob
                      │
                      ├─▶ SchemaLoader
                      ├─▶ PromptBuilder
                      ├─▶ ResponsesClient
                      ├─▶ SQLValidator
                      ├─▶ Critic
                      └─▶ Writer
```

### Execution flow

1. ``InputLoader`` reads ``config.yaml`` and expands each phase into one or more
   NL tasks.
2. ``SchemaLoader`` introspects the PostgreSQL database and provides table
   metadata for prompts and validation.
3. ``AutonomousJob`` processes each task using LLM tool-calling:
   - ``PromptBuilder`` crafts the user/system messages or template based on the
     phase settings.
   - ``ResponsesClient`` queries OpenAI to generate SQL.
   - ``SQLValidator`` runs ``EXPLAIN`` against the database and reports errors.
   - ``Critic`` reviews the SQL and can return a fixed version.
   - ``Writer`` executes the final SQL and returns anonymised rows.

4. Each result is optionally appended to a JSONL dataset as specified in the
   phase metadata.

## Modules

- `input_loader` – reads tasks from `config.yaml`
- `schema_loader` – introspects database tables
- `schema_relationship` – infers relationships between tables
- `prompt_builder` – crafts prompts for the LLM
- `openai_responses` – async wrapper around OpenAI
- `sql_validator` – validates generated SQL
- `critic` – reviews and fixes SQL
- `writer` – executes SQL and writes dataset entries
- `agent_pool` & `worker_agent` – parallel schema documentation helpers
- `autonomous_job` – orchestrates the entire pipeline
- `logger` – sets up rich console and file logging
- `main` – CLI entrypoint

## Schema relationship phase

This phase automatically discovers how tables relate to each other. It pulls
sample rows from every table, analyses column pairs for overlap and correlation
and then prompts the LLM using the `schema_relationship_template.txt` prompt.
The generated question/relationship pairs are written to
`generated_datasets/schema_relationship/dataset.jsonl` and can be invoked via
`--phase schema_relationship` when running the CLI.

## Running tests

This project ships with a small test suite. After installing the runtime
dependencies, install ``pytest`` and run the tests from the repository root:

```bash
pip install pytest
pytest -q
```

## Development

Formatting and linting are enforced with **black** and **ruff**. Run the
following before committing changes:

```bash
black --line-length 100 .
ruff check --fix .
```

## Scaling & Parallelism

You can control how many helper agents run concurrently via the `parallelism` setting.
Set a global default under `defaults` in `config.yaml` or override per phase. The
`DG_PARALLELISM` environment variable takes precedence. The dispatcher fans out
async helper agents which merge their unique question–answer pairs before
writing the dataset.
