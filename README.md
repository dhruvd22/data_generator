# data_generator

**LLM-ready SQL data generator.** Feed it plain-English questions and get vetted
PostgreSQL queries plus anonymised result rows. The pipeline guards budget via an
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

# generate via CLI
python -m nl_sql_generator.main gen questions.txt

# or from Python
from nl_sql_generator import AutonomousJob, SchemaLoader
schema = SchemaLoader.load_schema()
job = AutonomousJob(schema)
result = job.run_sync("count the patients")
print(result.sql, result.rows)
```

## 🔌 How it works

1. **main.AutonomousJob** – orchestrates the flow
2. **SchemaLoader** – pulls table metadata
3. **PromptBuilder** – crafts the few-shot prompt
4. **ResponsesClient** – queries OpenAI
5. **SQLValidator** – checks syntax via `EXPLAIN`
6. **Critic** – reviews and optionally fixes SQL
7. **Writer** – executes and fakes result rows

All configuration lives in `config.yaml`.
