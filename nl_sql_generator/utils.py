import os

__all__ = ["limit_pool_size"]


def limit_pool_size(
    workers: int, pool_size: int | None = None, tasks: int = 1
) -> int:
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
