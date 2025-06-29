# data_generator
NL‑to‑PostgreSQL Synthetic Data Generator

Turn natural language questions into validated PostgreSQL queries + synthetic result rows.Built for creating LLM training data that pairs NL → SQL → answers.

Features

🗄️ Schema‑aware: connects to a read‑only Supabase Postgres instance and introspects tables/columns.

✨ LLM‑first pipeline: orchestrates GPT‑4o via LangChain, with critic/validator loops.

✅ SQL safety net: runs EXPLAIN before execution, type‑checks, and catches runtime errors.

🔌 Modular: each pipeline phase (loader, prompt builder, generator, critic, writer) lives in its own file.

Quick Start

# 1. Clone & install
git clone git@github.com:dhruvd22/data_generator.git
cd data_generator
poetry install            # or: pip install -r requirements.txt

# 2. Set creds
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export OPENAI_API_KEY="sk-..."

# 3. Run the skeleton
python -m nl_sql_generator.main --config=config.yaml

Config

All runtime knobs live in config.yaml. Start with the provided template; tweak counts, budget or model as you iterate.

Repo Layout

l_sql_generator/
├── main.py          # CLI entrypoint
├── schema_loader.py # introspects DB
├── prompt_builder.py
├── sql_validator.py
└── ...
fewshot/             # prompt example YAMLs
tests/

Roadmap

Milestone 3 – prompt builder & first end‑to‑end sample

Milestone 4 – batch generation loop

Milestone 5 – dataset packaging + CI tests

Contributing

Fork → branch → PR.

Follow the commit style: feat: …, fix: …, docs: ….

Run pytest before pushing.
