import os

__all__ = ["limit_pool_size"]


def limit_pool_size(workers: int, pool_size: int | None = None) -> int:
    """Return a DB pool size respecting the global session cap."""
    if pool_size is None:
        pool_size = int(os.getenv("DB_COCURRENT_SESSION", "50"))
    max_total = int(os.getenv("MAX_DB_CONCURRENT_LIMIT_ALL", "450"))
    if workers <= 0:
        return pool_size
    allowed = max_total // workers
    if allowed <= 0:
        allowed = 1
    return min(pool_size, allowed)
