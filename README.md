# NL SQL Data Generator

A lightweight pipeline that turns natural language questions into
validated PostgreSQL queries using OpenAI. It introspects your database,
crafts prompts, validates results and writes JSONL datasets ready for
fine-tuning or evaluation.

[![Tests](https://img.shields.io/badge/tests-pytest-blue)](./tests)
![Python](https://img.shields.io/badge/python-3.12+-blue)

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Process Flows](#process-flows)
  - [Data Preparation](#data-preparation)
  - [Model Training / Fine-Tuning](#model-training--fine-tuning)
  - [Inference API / Batch Jobs](#inference-api--batch-jobs)
  - [Evaluation & Monitoring](#evaluation--monitoring)
- [Quick-Start Guide](#quick-start-guide)
- [Detailed Usage](#detailed-usage)
- [Generation Phases](#generation-phases)
- [Repository Structure](#repository-structure)
- [Configuration & Secrets Management](#configuration--secrets-management)
- [Logging, Testing & CI](#logging-testing--ci)
- [Contribution Guide](#contribution-guide)
- [License & Attribution](#license--attribution)
- [Changelog / Release Notes](#changelog--release-notes)

## Architecture Overview
The pipeline loads tasks from a YAML config and lets OpenAI drive the
SQL generation using tool-calling. Results are validated and optionally
logged for review.

```mermaid
flowchart TD
    C[config.yaml] --> L[InputLoader]
    L --> J[AutonomousJob]
    J --> S[SchemaLoader]
    J --> P[PromptBuilder]
    J --> O[ResponsesClient]
    J --> V[SQLValidator]
    J --> R[Critic]
    J --> W[ResultWriter]
    W --> D[(dataset.jsonl)]
```

## Process Flows
### Data Preparation
1. Export `DATABASE_URL` and `OPENAI_API_KEY`.
2. Run `gen` CLI with your YAML config.
3. Load phase tasks from the file.
4. Introspect tables via `SchemaLoader`.
5. Build prompts and query OpenAI.
6. For join-heavy phases `JoinPool` or `ComplexSqlPool` spawn workers
   to explore table combinations.
7. Validate SQL using `EXPLAIN`.
8. Optionally let the critic fix bad SQL.
9. Fetch anonymised rows and append to JSONL.

```mermaid
flowchart TD
    A[env vars] --> B[gen command]
    B --> C[InputLoader]
    C --> D[SchemaLoader]
    D --> E[PromptBuilder]
    E --> F[OpenAI]
    F --> G[SQLValidator]
    G --> H[Critic]
    H --> I[ResultWriter]
    I --> J[dataset]
```

### Model Training / Fine-Tuning
1. Install your ML framework (e.g. `transformers`).
2. Convert generated JSONL to training features.
3. Launch fine-tuning with the dataset.
4. Save the new checkpoint.

```mermaid
flowchart TD
    D1[dataset.jsonl] --> D2[preprocess]
    D2 --> D3[train script]
    D3 --> D4[model.ckpt]
```

### Inference API / Batch Jobs
1. Load schema with `SchemaLoader`.
2. Create an `AutonomousJob` instance.
3. Pass a question to `run_task` or `run_tasks`.
4. Receive validated SQL and fake rows.

```mermaid
flowchart TD
    I1[question] --> I2[AutonomousJob]
    I2 --> I3[OpenAI + tools]
    I3 --> I4[SQLValidator]
    I4 --> I5[ResultWriter]
    I5 --> I6[(rows)]
```

### Evaluation & Monitoring
1. `ResponsesClient` tracks token usage and spend.
2. `Critic` scores SQL and logs reviews.
3. All logs go to `logs/run-<timestamp>.log`.
4. Inspect JSONL events for debugging.

```mermaid
flowchart TD
    M1[ResponsesClient] --> M2[usage stats]
    M1 --> M3[logs]
    R1[Critic] --> M3
    M3 --> M4[analysis]
```

## Quick-Start Guide
### Prerequisites
- Python 3.12+
- Access to a PostgreSQL database
- OpenAI API key

### Install and Run
```bash
pip install -r nl_sql_generator/requirements.txt
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export OPENAI_API_KEY="sk-..."
python -m nl_sql_generator.main gen \
  --config nl_sql_generator/config.yaml --run-version demo
```

## Detailed Usage
- `--phase` limits generation to a single phase.
- `--run-version` appends a suffix to dataset files.

Example Python call:
```python
from nl_sql_generator import AutonomousJob, SchemaLoader, load_tasks
schema = SchemaLoader.load_schema()
job = AutonomousJob(schema)
tasks = load_tasks("nl_sql_generator/config.yaml", phase="single_table")
res = job.run_task(tasks[0])
print(res.sql, res.rows)
```
Expected output:
```
SELECT COUNT(*) FROM payers
[{"count": 123}]
```

## Generation Phases
The `config.yaml` file defines several phases used when creating datasets:

- **builtins** – showcase individual SQL functions on a single table.
- **schema_docs** – generate question/answer pairs describing the schema.
- **single_table** – create NL/SQL examples querying a single table.
- **joins** – produce questions that join multiple tables together.
- **complex_sqls** – larger join queries based on GPT suggested table sets.
- **sample_data** – export anonymised sample rows from each table.
- **schema_relationship** – discover potential relationships between tables.

Select a phase via `--phase` or by passing the name to `load_tasks`.

## Repository Structure
```text
.
├── nl_sql_generator/       # library modules and CLI
│   ├── autonomous_job.py       # orchestrates the tool pipeline
│   ├── input_loader.py         # reads YAML tasks
│   ├── schema_loader.py        # introspects PostgreSQL
│   ├── prompt_builder.py       # constructs chat prompts
│   ├── openai_responses.py     # async OpenAI client with budget tracking
│   ├── sql_validator.py        # EXPLAIN-based validation
│   ├── critic.py               # optional SQL review and fixes
│   ├── writer.py               # executes SQL and anonymises rows
│   ├── agent_pool.py           # parallel schema doc workers
│   ├── worker_agent.py         # individual schema QA generator
│   ├── join_pool.py            # coordinates join generation workers
│   ├── join_worker.py          # produces multi-table SQL joins
│   ├── complex_sql_pool.py     # larger join sets using GPT suggestions
│   ├── schema_relationship.py  # infers relationships between tables
│   ├── logger.py               # structured logging helpers
│   ├── main.py                 # CLI entry points
│   ├── validator.py            # no-op validator used by agents
│   ├── prompt_template/        # Jinja templates for prompts
│   ├── table_ddls.sql          # example DDLs for tests
│   ├── config.yaml             # sample phases configuration
│   └── requirements.txt        # Python dependencies
├── tests/                  # pytest suite
└── README.md
```

## Configuration & Secrets Management
`nl_sql_generator/config.yaml` defines phases with fields like
`name`, `count`, `dataset_output_file_dir` and `prompt_template`.
Environment variables:
- `DATABASE_URL` – PostgreSQL connection string
- `OPENAI_API_KEY` – OpenAI credentials
- `DG_PARALLELISM` – optional override for helper agents

## Logging, Testing & CI
- Logs live in `logs/` with one file per run.
- Run tests with:
  ```bash
  pytest -q
  ```
- CI status: ![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

## Contribution Guide
1. Fork the repo and create a feature branch.
2. Format code with `black` and `ruff` before opening a PR.
3. Ensure tests pass and include a clear description.

## License & Attribution
No license file is present. Use at your own risk and credit the
original authors when sharing derivatives.

## Changelog / Release Notes
- Improved relationship discovery and validation heuristics
- Added structured logging and error handling
- Fixed database URL handling across modules
