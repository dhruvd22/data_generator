import os, random, openai
from nl_sql_generator.prompt_builder import build_prompt
from nl_sql_generator.sql_validator import SQLValidator
import argparse, json, yaml
from nl_sql_generator.schema_loader import SchemaLoader

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    schema = SchemaLoader.load_schema()

    print(f"Loaded {len(schema)} tables from DB\n---")
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Set OPENAI_API_KEY first")

    validator = SQLValidator()

    for phase in cfg["phases"][:1]:  # just the first phase for now
        print(f"\n=== Phase: {phase['name']} ===")
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
        sql = response.choices[0].message.content.strip().strip("`").replace("SQL", "")

        ok, err = validator.check(sql)
        print("📝 NL question:", question)
        print("🏗️  Generated SQL:", sql)
        print("✅ Valid" if ok else f"❌ Invalid: {err}")
        break  # one sample is enough for this milestone

if __name__ == "__main__":
    cli()
