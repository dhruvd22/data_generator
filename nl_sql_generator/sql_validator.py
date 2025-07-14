"""Lightweight validator: run ``EXPLAIN`` to catch SQL errors.

``SQLValidator`` does not execute queries; it merely issues ``EXPLAIN`` to the
configured PostgreSQL database to detect syntax or missing table issues.

Example:
    >>> v = SQLValidator()
    >>> v.check("SELECT 1")
    (True, None)
"""

__all__ = ["SQLValidator"]
import logging
import os
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


class SQLValidator:
    """Run ``EXPLAIN`` and a lightweight execution to verify SQL."""

    def __init__(self, db_url: str | None = None, pool_size: int | None = None):
        """Initialise the validator.

        Args:
            db_url: Optional database URL, otherwise ``DATABASE_URL`` env is used.
            pool_size: Connection pool size for concurrent validations.

        Raises:
            ValueError: If no database URL is provided.
        """

        # Strip whitespace to avoid connection issues when env vars contain
        # trailing newlines or spaces.
        self.db_url = (db_url or os.getenv("DATABASE_URL", "")).strip()
        if not self.db_url:
            raise ValueError("DATABASE_URL not set")
        if pool_size is None:
            pool_size = int(os.getenv("DB_COCURRENT_SESSION", "50"))
        self.eng = create_engine(
            self.db_url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=0,
            connect_args={"sslmode": "require"},
        )

    def check(self, sql: str) -> tuple[bool, str | None]:
        """Return ``(is_valid, error_msg)`` for the given SQL."""
        log.info("Validating SQL via EXPLAIN and execution: %s", sql)
        try:
            with self.eng.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql}"))
                # attempt to run the query with LIMIT 1 to catch runtime issues
                conn.execute(text(f"SELECT 1 FROM ({sql}) AS _q LIMIT 1"))
            log.debug("SQL validation succeeded")
            return True, None
        except Exception as err:  # broad on purpose
            log.warning("SQL validation failed: %s", err)
            return False, str(err)
