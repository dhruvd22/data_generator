# nl_sql_generator/schema_loader.py
"""Database schema introspection helpers.

``SchemaLoader`` connects to PostgreSQL and returns lightweight table metadata
used by the prompt builder. The ``to_json`` helper converts that structure into
a JSON-serialisable form.

Example:
    >>> schema = SchemaLoader.load_schema()
    >>> list(schema)
    ['patients', 'appointments']
"""

from dataclasses import dataclass
from typing import Dict, List
from sqlalchemy import create_engine, inspect
import os
import json
import logging

__all__ = ["SchemaLoader", "ColumnInfo", "TableInfo"]

log = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Metadata for a database column.

    Attributes:
        name: Column name.
        type_: Type string as reported by SQLAlchemy.
        comment: Optional column comment.
    """

    name: str
    type_: str
    comment: str | None = None


@dataclass
class TableInfo:
    """Metadata for a database table.

    Attributes:
        name: Table name.
        columns: Sequence of :class:`ColumnInfo` objects.
        primary_key: Name of the primary key column if present.
    """

    name: str
    columns: List[ColumnInfo]
    primary_key: str | None = None


class SchemaLoader:
    """Load table definitions from a PostgreSQL database."""

    @staticmethod
    def load_schema(db_url: str | None = None) -> Dict[str, TableInfo]:
        """Return a mapping of table name to :class:`TableInfo`.

        Args:
            db_url: Optional PostgreSQL connection URL.

        Returns:
            Mapping of table names to their metadata.

        Raises:
            ValueError: If no database URL is provided.
        """
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
        """Convert ``schema`` to a JSON-serialisable dict."""
        tables: Dict[str, dict] = {}
        for name, info in schema.items():
            cols = {c.name: {"type": c.type_, "comment": c.comment} for c in info.columns}
            tables[name] = {
                "columns": cols,
                "primary_key": info.primary_key,
            }
        return {"tables": tables}


if __name__ == "__main__":
    # quick manual test
    schema = SchemaLoader.load_schema()
    log = logging.getLogger(__name__)
    log.info(json.dumps(SchemaLoader.to_json(schema), indent=2))
