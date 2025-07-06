# data_generator

**LLM-ready SQL data generator.** That automatically drives through various phases as per config.yaml and generates NL to PostgreSQL datasets plus anonymised result rows. The pipeline guards budget via an
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

## Running tests

This project ships with a small test suite. After installing the runtime
dependencies, install ``pytest`` and run the tests from the repository root:

```bash
pip install pytest
pytest -q
```
