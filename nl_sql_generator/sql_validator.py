"""Lightweight validator: run ``EXPLAIN`` to catch SQL errors.

``SQLValidator`` does not execute queries; it merely issues ``EXPLAIN`` to the
configured PostgreSQL database to detect syntax or missing table issues.

Example:
    >>> v = SQLValidator()
    >>> v.check("SELECT 1")
    (True, None)
"""

__all__ = ["SQLValidator"]
from sqlalchemy import text, create_engine
import os


class SQLValidator:
    """Run ``EXPLAIN`` to verify SQL without executing it."""

    def __init__(self, db_url: str | None = None):
        """Initialise the validator.

        Args:
            db_url: Optional database URL, otherwise ``DATABASE_URL`` env is used.

        Raises:
            ValueError: If no database URL is provided.
        """

        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not set")
        self.eng = create_engine(
            self.db_url, pool_pre_ping=True, connect_args={"sslmode": "require"}
        )

    def check(self, sql: str) -> tuple[bool, str | None]:
        """Return ``(is_valid, error_msg)`` for the given SQL."""
        try:
            with self.eng.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql}"))
            return True, None
        except Exception as err:  # broad on purpose
            return False, str(err)
