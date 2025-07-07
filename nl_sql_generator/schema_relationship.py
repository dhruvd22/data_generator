"""Infer table relationships using sample rows."""

from __future__ import annotations

from typing import Dict, List
import asyncio
import pandas as pd
from pandas.api import types as ptypes
from sqlalchemy import text

try:  # optional sempy/semopy support
    from semopy import Model
except Exception:  # pragma: no cover - optional dependency
    Model = None

from .schema_loader import TableInfo


async def _fetch_rows(engine, table: str, n_rows: int) -> pd.DataFrame:
    """Return ``n_rows`` sample rows from ``table`` as a DataFrame."""

    def _run() -> pd.DataFrame:
        with engine.connect() as conn:
            res = conn.execute(text(f"SELECT * FROM {table} LIMIT {n_rows}"))
            cols = list(res.keys())
            rows = res.fetchall()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows])

    return await asyncio.to_thread(_run)


def _score_relation(series_a: pd.Series, series_b: pd.Series) -> float:
    """Return similarity score between two columns."""
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
    return 0.0 if pd.isna(corr) else abs(float(corr))


def _analyze_pair(t1: str, df1: pd.DataFrame, t2: str, df2: pd.DataFrame) -> List[Dict[str, str]]:
    """Return relationship records for ``t1`` and ``t2``."""
    relations: List[Dict[str, str]] = []
    for c1 in df1.columns:
        for c2 in df2.columns:
            score = _score_relation(df1[c1], df2[c2])
            # also consider overlapping values for categorical columns
            overlap = len(set(df1[c1].dropna()) & set(df2[c2].dropna()))
            if score > 0.8 or overlap > 0:
                relations.append(
                    {
                        "question": f"How is {t1}.{c1} related to {t2}.{c2}?",
                        "relationship": f"{t1}.{c1} -> {t2}.{c2}",
                    }
                )
    return relations


async def discover_relationships(
    schema: Dict[str, TableInfo],
    engine,
    n_rows: int = 5,
    parallelism: int = 4,
) -> List[Dict[str, str]]:
    """Return relationship pairs extracted from the database."""
    tables = list(schema.keys())
    data = {tbl: await _fetch_rows(engine, tbl, n_rows) for tbl in tables}

    results: List[Dict[str, str]] = []
    tasks = []
    for i, t1 in enumerate(tables):
        for t2 in tables[i + 1 :]:
            df1 = data[t1]
            df2 = data[t2]
            tasks.append(asyncio.to_thread(_analyze_pair, t1, df1, t2, df2))
            if len(tasks) >= parallelism:
                for rels in await asyncio.gather(*tasks):
                    results.extend(rels)
                tasks = []
    if tasks:
        for rels in await asyncio.gather(*tasks):
            results.extend(rels)
    return results
