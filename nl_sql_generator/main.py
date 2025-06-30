import argparse, json, yaml
from nl_sql_generator.schema_loader import SchemaLoader

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    schema = SchemaLoader.load_schema()

    print(f"Loaded {len(schema)} tables from DB\n---")
    print(
        json.dumps(
            {t: [c.name for c in info.columns] for t, info in schema.items()},
            indent=2,
        )
    )
    # later phases will plug in here

if __name__ == "__main__":
    cli()
