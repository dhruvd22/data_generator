"""Execute SQL and anonymise result rows.

The :class:`ResultWriter` fetches query results and replaces sensitive data with
deterministic fake values. This enables creating shareable datasets.

Example:
    >>> writer = ResultWriter()
    >>> rows = writer.fetch("SELECT id, email FROM users", n_rows=2)
    >>> writer.write_jsonl(rows, "out.jsonl")
"""

import os
import json
import tempfile
import datetime
from typing import Any, Dict, List

from sqlalchemy import create_engine, text
from faker import Faker

__all__ = ["ResultWriter"]


class ResultWriter:
    """Execute SQL and return fake rows for privacy-preserving datasets."""

    def __init__(self, db_url: str | None = None) -> None:
        """Initialise the writer.

        Args:
            db_url: Optional database URL else ``DATABASE_URL`` env variable.

        Raises:
            ValueError: If no database URL can be determined.
        """

        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not set")
        self.eng = create_engine(db_url, pool_pre_ping=True, connect_args={"sslmode": "require"})
        self.fake = Faker()

    def fetch(self, sql: str, n_rows: int = 5) -> List[Dict[str, Any]]:
        """Execute the query and return up to ``n_rows`` fake rows.

        Args:
            sql: SQL statement to execute.
            n_rows: Maximum number of rows to fetch.

        Returns:
            List of anonymised result rows.

        Raises:
            RuntimeError: If SQL execution fails.
        """
        from sqlalchemy.exc import SQLAlchemyError

        with self.eng.connect() as conn:
            try:
                res = conn.execute(text(sql))
                cols = list(res.keys())
                rows = res.fetchmany(n_rows)
            except SQLAlchemyError as err:  # pragma: no cover - depends on DB
                raise RuntimeError(f"SQL execution failed: {err}") from err

        data: List[Dict[str, Any]] = []
        for row in rows:
            record = {}
            for col, val in zip(cols, row):
                record[col] = self._fake_value(col, val)
            data.append(record)
        return data

    def write_jsonl(self, rows: List[Dict[str, Any]], path: str) -> None:
        """Atomically write the ``rows`` to ``path`` in JSONL format."""
        directory = os.path.dirname(path) or "."
        with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, suffix=".tmp") as tmp:
            for r in rows:
                tmp.write(json.dumps(r))
                tmp.write("\n")
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    def append_jsonl(self, row: Dict[str, Any], path: str) -> None:
        """Append ``row`` as JSON to ``path`` creating directories when needed."""
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row))
            fh.write("\n")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _fake_value(self, column: str, value: Any) -> Any:
        """Return a deterministic fake equivalent for ``value``."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, datetime.datetime):
            return self.fake.date_time_this_decade()
        if isinstance(value, datetime.date):
            return self.fake.date_this_decade()
        if isinstance(value, datetime.time):
            return self.fake.time()
        if isinstance(value, str):
            name = column.lower()
            if "email" in name:
                return self.fake.email()
            if "name" in name:
                return self.fake.name()
            if "date" in name:
                return str(self.fake.date())
            if "time" in name:
                return str(self.fake.time())
            return self.fake.word()
        # fallback for unsupported types
        try:
            return str(value)
        except Exception:
            return None
