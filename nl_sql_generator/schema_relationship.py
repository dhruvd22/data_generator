"""Infer table relationships using sample rows."""

from __future__ import annotations

from typing import Dict, List
import asyncio
import pandas as pd
from pandas.api import types as ptypes
from sqlalchemy import text
import logging

try:  # optional sempy/semopy support
    from semopy import Model
except Exception:  # pragma: no cover - optional dependency
    Model = None

from .schema_loader import TableInfo

log = logging.getLogger(__name__)


async def _fetch_rows(engine, table: str, n_rows: int) -> pd.DataFrame:
    """Return ``n_rows`` sample rows from ``table`` as a DataFrame."""

    log.info("Fetching %d rows from table %s", n_rows, table)

    def _run() -> pd.DataFrame:
        with engine.connect() as conn:
            res = conn.execute(text(f"SELECT * FROM {table} LIMIT {n_rows}"))
            cols = list(res.keys())
            rows = res.fetchall()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows])

    df = await asyncio.to_thread(_run)
    log.info("Fetched %d rows from table %s", len(df), table)
    return df


def _score_relation(series_a: pd.Series, series_b: pd.Series) -> float:
    """Return similarity score between two columns."""

    log.debug("Scoring relation between series of length %d and %d", len(series_a), len(series_b))
    df = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    if df.empty:
        return 0.0

    def _to_numeric(s: pd.Series) -> pd.Series:
        if ptypes.is_numeric_dtype(s) or ptypes.is_bool_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        return pd.Series(pd.factorize(s.astype(str))[0], index=s.index)

    df["a"] = _to_numeric(df["a"])
    df["b"] = _to_numeric(df["b"])

    # avoid runtime warnings when series are constant
    if df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return 0.0

    if Model is not None:
        try:  # pragma: no cover - best effort sempy usage
            m = Model("b ~ a")
            m.fit(df, quiet=True)
            params = m.parameters_dict
            val = params.get("l_b_a") or params.get("Beta") or 0.0
            return abs(float(val))
        except Exception:  # pragma: no cover - fallback
            pass

    corr = df["a"].corr(df["b"])
    score = 0.0 if pd.isna(corr) else abs(float(corr))
    log.debug("Correlation score: %.4f", score)
    return score


def _compatible_types(series_a: pd.Series, series_b: pd.Series) -> bool:
    """Return ``True`` if series have comparable data types."""
    if ptypes.is_numeric_dtype(series_a) and ptypes.is_numeric_dtype(series_b):
        return True
    if ptypes.is_bool_dtype(series_a) and ptypes.is_bool_dtype(series_b):
        return True
    if ptypes.is_datetime64_any_dtype(series_a) and ptypes.is_datetime64_any_dtype(series_b):
        return True
    if ptypes.is_string_dtype(series_a) and ptypes.is_string_dtype(series_b):
        return True
    return False


def _analyze_pair(t1: str, df1: pd.DataFrame, t2: str, df2: pd.DataFrame) -> List[Dict[str, str]]:
    """Return relationship records for ``t1`` and ``t2``."""
    log.info("Analyzing table pair %s <-> %s", t1, t2)
    relations: List[Dict[str, str]] = []
    for c1 in df1.columns:
        for c2 in df2.columns:
            s1 = df1[c1]
            s2 = df2[c2]
            if not _compatible_types(s1, s2):
                continue
            score = _score_relation(s1, s2)
            vals1 = set(s1.dropna())
            vals2 = set(s2.dropna())
            overlap = vals1 & vals2
            ov_count = len(overlap)
            ratio1 = ov_count / len(vals1) if vals1 else 0.0
            ratio2 = ov_count / len(vals2) if vals2 else 0.0
            log.debug(
                "Pair %s.%s vs %s.%s score=%.4f overlap=%d ratio1=%.2f ratio2=%.2f",
                t1,
                c1,
                t2,
                c2,
                score,
                ov_count,
                ratio1,
                ratio2,
            )
            if ov_count >= 2 and (score >= 0.9 or (ratio1 >= 0.5 and ratio2 >= 0.5)):
                relations.append(
                    {
                        "question": f"How is {t1}.{c1} related to {t2}.{c2}?",
                        "relationship": f"{t1}.{c1} -> {t2}.{c2}",
                    }
                )
    log.info("Found %d relationships between %s and %s", len(relations), t1, t2)
    return relations


async def discover_relationships(
    schema: Dict[str, TableInfo],
    engine,
    n_rows: int = 5,
    parallelism: int = 4,
) -> List[Dict[str, str]]:
    """Return relationship pairs extracted from the database."""
    log.info(
        "Discovering relationships using %d rows per table and parallelism=%d",
        n_rows,
        parallelism,
    )
    tables = list(schema.keys())
    log.info("Fetching sample rows for %d tables", len(tables))
    data = {tbl: await _fetch_rows(engine, tbl, n_rows) for tbl in tables}

    results: List[Dict[str, str]] = []
    tasks = []
    for i, t1 in enumerate(tables):
        for t2 in tables[i + 1 :]:
            df1 = data[t1]
            df2 = data[t2]
            log.info("Scheduling analysis for %s <-> %s", t1, t2)
            tasks.append(asyncio.to_thread(_analyze_pair, t1, df1, t2, df2))
            if len(tasks) >= parallelism:
                n = len(tasks)
                for rels in await asyncio.gather(*tasks):
                    results.extend(rels)
                log.info("Processed %d table pairs", n)
                tasks = []
    if tasks:
        n = len(tasks)
        for rels in await asyncio.gather(*tasks):
            results.extend(rels)
        log.info("Processed final %d table pairs", n)
    log.info("Discovered %d relationships", len(results))
    return results
