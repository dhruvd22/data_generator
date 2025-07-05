# nl_sql_generator/schema_loader.py
from dataclasses import dataclass
from typing import Dict, List
from sqlalchemy import create_engine, inspect
import os, json


@dataclass
class ColumnInfo:
    name: str
    type_: str
    comment: str | None = None


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    primary_key: str | None = None


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
            cols = []
            for c in insp.get_columns(tbl):
                cols.append(
                    ColumnInfo(
                        c["name"],
                        str(c["type"]),
                        c.get("comment"),
                    )
                )
            pk_cols = insp.get_pk_constraint(tbl).get("constrained_columns")
            pk = pk_cols[0] if pk_cols else None
            schema[tbl] = TableInfo(tbl, cols, pk)

        return schema

    @staticmethod
    def to_json(schema: Dict[str, TableInfo]) -> dict:
        tables: Dict[str, dict] = {}
        for name, info in schema.items():
            cols = {
                c.name: {"type": c.type_, "comment": c.comment} for c in info.columns
            }
            tables[name] = {
                "columns": cols,
                "primary_key": info.primary_key,
            }
        return {"tables": tables}


if __name__ == "__main__":
    # quick manual test
    schema = SchemaLoader.load_schema()
    print(json.dumps(SchemaLoader.to_json(schema), indent=2))
