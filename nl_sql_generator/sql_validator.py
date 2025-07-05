"""Lightweight validator: run EXPLAIN to catch syntax / unknown table errors."""

__all__ = ["SQLValidator"]
from sqlalchemy import text, create_engine
import os

class SQLValidator:
    """Run ``EXPLAIN`` to verify SQL without executing it."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not set")
        self.eng = create_engine(self.db_url, pool_pre_ping=True, connect_args={"sslmode": "require"})

    def check(self, sql: str) -> tuple[bool, str | None]:
        """Return (is_valid, error_msg). Does *not* execute the query."""
        try:
            with self.eng.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql}"))
            return True, None
        except Exception as err:  # broad on purpose
            return False, str(err)
