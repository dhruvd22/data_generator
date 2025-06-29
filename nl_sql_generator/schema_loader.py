# nl_sql_generator/schema_loader.py
from dataclasses import dataclass
from typing import Dict, List
from sqlalchemy import create_engine, inspect
import os, json

@dataclass
class ColumnInfo:
    name: str
    type_: str

@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]

class SchemaLoader:
    @staticmethod
    def load_schema(db_url: str | None = None) -> Dict[str, TableInfo]:
        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("Provide DATABASE_URL env var or param")

        eng = create_engine(db_url, pool_pre_ping=True)
        insp = inspect(eng)

        schema: Dict[str, TableInfo] = {}
        for tbl in insp.get_table_names():
            cols = [
                ColumnInfo(c["name"], str(c["type"]))
                for c in insp.get_columns(tbl)
            ]
            schema[tbl] = TableInfo(tbl, cols)

        return schema

if __name__ == "__main__":
    # quick manual test
    print(
        json.dumps(
            SchemaLoader.load_schema(),
            default=lambda o: o.__dict__,
            indent=2,
        )
    )
