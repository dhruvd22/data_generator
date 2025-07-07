"""Infer table relationships via explicit constraints and heuristics."""

from __future__ import annotations

from typing import Dict, List, Any
import asyncio
import logging
import os
from difflib import SequenceMatcher

import numpy as np
from sqlalchemy import inspect, text

try:  # optional openai support
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None

from .schema_loader import TableInfo

__all__ = ["discover_relationships"]

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _same_type(t1: str, t2: str) -> bool:
    def _base(t: str) -> str:
        return t.split("(")[0].lower()

    return _base(t1) == _base(t2)


async def _comment_similarity(c1: str | None, c2: str | None) -> float:
    """Return cosine similarity between ``c1`` and ``c2`` comments."""
    if not c1 or not c2:
        return 0.0
    if openai is None or os.getenv("OPENAI_API_KEY") is None:
        return 0.0

    def _embed() -> tuple[np.ndarray, np.ndarray]:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small", input=[c1, c2])
        e1 = np.array(resp.data[0].embedding, dtype=float)
        e2 = np.array(resp.data[1].embedding, dtype=float)
        return e1, e2

    try:
        emb1, emb2 = await asyncio.to_thread(_embed)
    except Exception as err:  # pragma: no cover - network failures
        log.warning("Embedding failed: %s", err)
        return 0.0

    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(emb1, emb2) / denom)


async def _values_contained(
    engine, t_from: str, c_from: str, t_to: str, c_to: str, limit: int
) -> bool:
    """Return ``True`` if sampled values from ``t_from.c_from`` mostly appear in ``t_to.c_to``."""

    def _run() -> bool:
        with engine.connect() as conn:
            q1 = text(
                f"SELECT DISTINCT {c_from} FROM {t_from} "
                f"WHERE {c_from} IS NOT NULL LIMIT {limit}"
            )
            vals_a = {row[0] for row in conn.execute(q1)}
            if not vals_a:
                return False
            q2 = text(f"SELECT DISTINCT {c_to} FROM {t_to} WHERE {c_to} IS NOT NULL")
            vals_b = {row[0] for row in conn.execute(q2)}
        if not vals_a:
            return False
        matched = len([v for v in vals_a if v in vals_b])
        return matched / len(vals_a) >= 0.95

    try:
        return await asyncio.to_thread(_run)
    except Exception as err:  # pragma: no cover - DB errors
        log.warning("Value check failed: %s", err)
        return False


# ----------------------------------------------------------------------
# main logic
# ----------------------------------------------------------------------


async def discover_relationships(
    schema: Dict[str, TableInfo], engine, sample_limit: int = 5000
) -> List[Dict[str, Any]]:
    """Return discovered relationships sorted by confidence."""

    insp = inspect(engine)
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # ----------------------- step 1: explicit FKs ---------------------
    for table in schema:
        for fk in insp.get_foreign_keys(table):
            rt = fk.get("referred_table")
            lcols = fk.get("constrained_columns") or []
            rcols = fk.get("referred_columns") or []
            if not rt:
                continue
            for lc, rc in zip(lcols, rcols):
                rel = f"{table}.{lc} -> {rt}.{rc}"
                if rel in seen:
                    continue
                seen.add(rel)
                results.append(
                    {
                        "question": f"How is {table}.{lc} related to {rt}.{rc}?",
                        "relationship": rel,
                        "confidence": 1.0,
                    }
                )

    # gather PK/unique info for heuristic checks
    pk_unique: Dict[str, set[str]] = {}
    for tbl in schema:
        pkc = insp.get_pk_constraint(tbl).get("constrained_columns") or []
        uniq = []
        for uc in insp.get_unique_constraints(tbl):
            uniq.extend(uc.get("column_names", []))
        for idx in insp.get_indexes(tbl):
            if idx.get("unique"):
                uniq.extend(idx.get("column_names", []))
        pk_unique[tbl] = set(pkc + uniq)

    # mapping of column info
    type_map: Dict[tuple[str, str], str] = {}
    comment_map: Dict[tuple[str, str], str | None] = {}
    for t, info in schema.items():
        for col in info.columns:
            key = (t, col.name)
            type_map[key] = col.type_
            comment_map[key] = col.comment

    # --------------------- steps 2-4 heuristics -----------------------
    for ftbl, finfo in schema.items():
        for fcol in finfo.columns:
            ftype = type_map[(ftbl, fcol.name)]
            fcomment = comment_map[(ftbl, fcol.name)]
            for rtbl, _rinfo in schema.items():
                if ftbl == rtbl:
                    continue
                for rcol in _rinfo.columns:
                    if rcol.name not in pk_unique.get(rtbl, set()):
                        continue
                    rtype = type_map[(rtbl, rcol.name)]
                    rcomment = comment_map[(rtbl, rcol.name)]
                    if not _same_type(ftype, rtype):
                        continue
                    sim = max(
                        _name_similarity(fcol.name, rcol.name),
                        _name_similarity(fcol.name, f"{rtbl}_{rcol.name}"),
                        _name_similarity(fcol.name, f"{rtbl.rstrip('s')}_{rcol.name}"),
                    )
                    if sim < 0.8:
                        continue
                    conf = 0.75
                    com_sim = await _comment_similarity(fcomment, rcomment)
                    if com_sim >= 0.83:
                        conf += 0.1
                    if await _values_contained(
                        engine, ftbl, fcol.name, rtbl, rcol.name, sample_limit
                    ):
                        conf += 0.1
                    if conf >= 0.75:
                        rel = f"{ftbl}.{fcol.name} -> {rtbl}.{rcol.name}"
                        if rel in seen:
                            continue
                        seen.add(rel)
                        results.append(
                            {
                                "question": f"How is {ftbl}.{fcol.name} related to {rtbl}.{rcol.name}?",
                                "relationship": rel,
                                "confidence": round(conf, 2),
                            }
                        )

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results
