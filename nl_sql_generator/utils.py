import os

__all__ = ["limit_pool_size", "pool_usage"]


def limit_pool_size(workers: int, pool_size: int | None = None, tasks: int = 1) -> int:
    """Return a DB pool size respecting the global session cap.

    ``workers`` denotes concurrent worker instances per task and ``tasks``
    indicates how many tasks run simultaneously.  The function ensures that
    ``workers * tasks * pool_size`` never exceeds
    ``MAX_DB_CONCURRENT_LIMIT_ALL`` by returning::

        min(pool_size, MAX_DB_CONCURRENT_LIMIT_ALL // (workers * tasks))
    """

    if pool_size is None:
        pool_size = int(os.getenv("DB_COCURRENT_SESSION", "50"))

    max_total = int(os.getenv("MAX_DB_CONCURRENT_LIMIT_ALL", "450"))

    total_workers = max(workers * max(tasks, 1), 1)
    allowed = max_total // total_workers

    if allowed <= 0:
        allowed = 1

    return min(pool_size, allowed)


def pool_usage(engine) -> tuple[int, int]:
    """Return ``(in_use, available)`` DB sessions for ``engine``'s pool.

    The helper attempts to query ``engine.pool`` for ``checkedout`` and ``size``
    information.  If either attribute is missing, ``(0, 0)`` is returned.
    """

    pool = getattr(engine, "pool", None)
    if not pool:
        return 0, 0

    try:
        in_use = int(pool.checkedout())
    except Exception:
        in_use = 0

    try:
        total = int(pool.size())
    except Exception:
        total = 0

    available = max(total - in_use, 0)
    return in_use, available
