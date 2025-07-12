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
    comment: str | None = None


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
        # Trim whitespace so extraneous newlines don't break authentication
        db_url = (db_url or os.getenv("DATABASE_URL", "")).strip()
        if not db_url:
            raise ValueError("Provide DATABASE_URL env var or param")

        log.info("Loading schema from %s", db_url)
        eng = create_engine(db_url, pool_pre_ping=True)
        insp = inspect(eng)

        schema: Dict[str, TableInfo] = {}
        for tbl in insp.get_table_names():
            t_comment = insp.get_table_comment(tbl).get("text")
            log.debug("Processing table %s comment=%r", tbl, t_comment)
            cols = []
            for c in insp.get_columns(tbl):
                cols.append(
                    ColumnInfo(
                        c["name"],
                        str(c["type"]),
                        c.get("comment"),
                    )
                )
            log.debug("Discovered columns for %s: %s", tbl, [c.name for c in cols])
            pk_cols = insp.get_pk_constraint(tbl).get("constrained_columns")
            pk = pk_cols[0] if pk_cols else None
            schema[tbl] = TableInfo(tbl, cols, pk, t_comment)
        log.info("Loaded %d tables", len(schema))

        return schema

    @staticmethod
    def to_json(schema: Dict[str, TableInfo], max_tables: int | None = None) -> dict:
        """Convert ``schema`` to a JSON-serialisable dict.

        Args:
            schema: Mapping of table names to :class:`TableInfo`.
            max_tables: Optional limit on the number of tables to include.
        """

        items = list(schema.items())
        if max_tables is not None:
            import random

            items = random.sample(items, k=min(max_tables, len(items)))

        tables: Dict[str, dict] = {}
        for name, info in items:
            cols = {c.name: {"type": c.type_, "comment": c.comment} for c in info.columns}
            tables[name] = {
                "columns": cols,
                "primary_key": info.primary_key,
                "comment": info.comment,
            }
        return {"tables": tables}


if __name__ == "__main__":
    # quick manual test
    logging.basicConfig(level=logging.INFO)
    schema = SchemaLoader.load_schema()
    print(json.dumps(SchemaLoader.to_json(schema), indent=2))
