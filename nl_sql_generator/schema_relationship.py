"""Infer table relationships via explicit constraints and heuristics."""

from __future__ import annotations

from typing import Dict, List, Any
import asyncio
import logging
import os
import json
import re
from difflib import SequenceMatcher

import numpy as np
from sqlalchemy import inspect, text

from .sql_validator import SQLValidator

try:  # optional openai support
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None

from .schema_loader import TableInfo, SchemaLoader
from .openai_responses import acomplete, _load_budget

__all__ = ["discover_relationships"]

log = logging.getLogger(__name__)

# minimum heuristic votes required for GPT confirmation
THRESHOLD = 4

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


async def _combined_comment_similarity(
    c1: str | None, t1: str | None, c2: str | None, t2: str | None
) -> float:
    """Return max similarity using column comments alone or combined with table comments."""

    base = await _comment_similarity(c1, c2)
    joined1 = " ".join(x for x in [c1, t1] if x)
    joined2 = " ".join(x for x in [c2, t2] if x)
    combined = await _comment_similarity(joined1 or None, joined2 or None)
    return max(base, combined)


async def _value_overlap(
    engine, t_from: str, c_from: str, t_to: str, c_to: str, limit: int
) -> float:
    """Return the ratio of values from ``t_from.c_from`` found in ``t_to.c_to``."""

    def _run() -> float:
        log.debug(
            "Checking value overlap %s.%s -> %s.%s (limit=%d)",
            t_from,
            c_from,
            t_to,
            c_to,
            limit,
        )
        with engine.connect() as conn:
            q1 = text(
                f"SELECT DISTINCT {c_from} FROM {t_from} "
                f"WHERE {c_from} IS NOT NULL LIMIT {limit}"
            )
            vals_a = {row[0] for row in conn.execute(q1)}
            if not vals_a:
                return 0.0
            q2 = text(f"SELECT DISTINCT {c_to} FROM {t_to} WHERE {c_to} IS NOT NULL")
            vals_b = {row[0] for row in conn.execute(q2)}
        if not vals_a:
            return 0.0
        matched = len([v for v in vals_a if v in vals_b])
        return matched / len(vals_a)

    try:
        return await asyncio.to_thread(_run)
    except Exception as err:  # pragma: no cover - DB errors
        log.warning("Value overlap check failed: %s", err)
        return 0.0


async def _values_contained(
    engine, t_from: str, c_from: str, t_to: str, c_to: str, limit: int
) -> bool:
    """Return ``True`` if sampled values from ``t_from.c_from`` mostly appear in ``t_to.c_to``."""
    log.debug("Checking values contained %s.%s -> %s.%s", t_from, c_from, t_to, c_to)
    ratio = await _value_overlap(engine, t_from, c_from, t_to, c_to, limit)
    return ratio >= 0.95


async def _distinct_ratio(engine, table: str, column: str) -> float:
    """Return ``COUNT(DISTINCT column) / COUNT(*)`` for ``table``."""

    def _run() -> float:
        log.debug(
            "Checking distinct ratio for %s.%s",
            table,
            column,
        )
        with engine.connect() as conn:
            q = text(
                f"SELECT COUNT(DISTINCT {column})::FLOAT / NULLIF(COUNT(*), 0) "
                f"FROM {table}"
            )
            val = conn.execute(q).scalar()
        return float(val or 0.0)

    try:
        return await asyncio.to_thread(_run)
    except Exception as err:  # pragma: no cover - DB errors
        log.warning("Distinct ratio failed: %s", err)
        return 1.0


async def _no_orphans(
    engine, child: str, child_col: str, parent: str, parent_pk: str
) -> bool:
    """Return ``True`` if every ``child.child_col`` value exists in ``parent``."""

    def _run() -> bool:
        log.debug(
            "Checking orphan rows from %s.%s -> %s.%s",
            child,
            child_col,
            parent,
            parent_pk,
        )
        with engine.connect() as conn:
            q = text(
                f"SELECT COUNT(*) = 0 AS ok FROM {child} c "
                f"LEFT JOIN {parent} p ON c.{child_col} = p.{parent_pk} "
                f"WHERE c.{child_col} IS NOT NULL AND p.{parent_pk} IS NULL"
            )
            return bool(conn.execute(q).scalar())

    try:
        return await asyncio.to_thread(_run)
    except Exception as err:  # pragma: no cover - DB errors
        log.warning("Orphan check failed: %s", err)
        return False


async def _gpt_second_opinion(
    child: str, col: str, parent: str, pk: str, score: int
) -> str:
    """Ask GPT to confirm a relationship and return its verdict."""

    budget = _load_budget()
    if openai is None or os.getenv("OPENAI_API_KEY") is None or budget <= 0:
        return "yes"

    system = "You are a database expert. Reply with 'yes', 'no' or 'unsure'."
    user = (
        f"Child table: {child} column: {col}. Parent table: {parent} pk: {pk}. "
        f"Heuristic vote: {score}/6. Is this a foreign key relationship?"
    )

    try:
        response = await acomplete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
    except Exception as err:  # pragma: no cover - network failures
        log.warning("GPT opinion failed: %s", err)
        return "unsure"

    verdict = response.strip().lower()
    return verdict if verdict in {"yes", "no", "unsure"} else "unsure"


async def _gpt_relationship_sqls(
    schema: Dict[str, TableInfo], engine, n_rows: int = 2
) -> List[str]:
    """Return candidate relationship SQLs suggested by GPT."""
    if openai is None or os.getenv("OPENAI_API_KEY") is None:
        log.info("Skipping GPT relationship discovery")
        return []

    def _sample() -> Dict[str, list]:
        data: Dict[str, list] = {}
        with engine.connect() as conn:
            for t in schema:
                try:
                    q = text(f"SELECT * FROM {t} LIMIT {n_rows}")
                    rows = conn.execute(q).fetchall()
                    data[t] = [
                        dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
                        for r in rows
                    ]
                except Exception as err:  # pragma: no cover - DB errors
                    log.warning("Sample rows failed for %s: %s", t, err)
        return data

    try:
        rows = await asyncio.to_thread(_sample)
    except Exception as err:  # pragma: no cover - DB errors
        log.warning("Fetching sample rows failed: %s", err)
        rows = {}

    schema_json = SchemaLoader.to_json(schema)
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a database expert. Use the schema and sample rows to propose SQL "
                "queries confirming foreign key relationships. "
                "Return one SQL per line in the form: SELECT 1 FROM child a JOIN parent b ON a.col = b.pk LIMIT 1"
            ),
        },
        {
            "role": "user",
            "content": f"SCHEMA_JSON:\n{json.dumps(schema_json, indent=2, default=str)}\n\nSAMPLE_ROWS:\n{json.dumps(rows, indent=2, default=str)}",
        },
    ]

    log.info("Requesting GPT relationship suggestions")
    try:
        response_text = await acomplete(prompt)
    except Exception as err:  # pragma: no cover - network failures
        log.warning("GPT relationship discovery failed: %s", err)
        return []

    sqls: List[str] = []
    for line in response_text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip("`")
        if not line:
            continue
        line = line.rstrip(";").strip()
        if line.lower().startswith("select"):
            sqls.append(line)

    log.info("GPT suggested %d candidate SQLs", len(sqls))
    return sqls


# ----------------------------------------------------------------------
# main logic
# ----------------------------------------------------------------------


async def discover_relationships(
    schema: Dict[str, TableInfo],
    engine,
    sample_limit: int = 5000,
    parallelism: int = 4,
) -> List[Dict[str, Any]]:
    """Return discovered relationships sorted by confidence."""
    log.info("Starting relationship discovery with sample_limit=%d", sample_limit)

    insp = inspect(engine)
    validator = None
    if getattr(engine, "url", None):
        # ``str(engine.url)`` hides the password which breaks authentication.
        # Use ``render_as_string`` to obtain the full connection string.
        url = engine.url
        if hasattr(url, "render_as_string"):
            db_url = url.render_as_string(hide_password=False)
        else:  # pragma: no cover - fallback for custom URL objects
            db_url = str(url)
        validator = SQLValidator(db_url)
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # ----------------------- step 1: explicit FKs ---------------------
    log.info("Checking explicit foreign keys")
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
                log.debug("Found FK %s", rel)
                results.append(
                    {
                        "question": f"How is {table}.{lc} related to {rt}.{rc}?",
                        "answer": rel,
                        "confidence": 1.0,
                    }
                )
                rev_rel = f"{rt}.{rc} -> {table}.{lc}"
                if rev_rel not in seen:
                    seen.add(rev_rel)
                    results.append(
                        {
                            "question": f"How is {rt}.{rc} related to {table}.{lc}?",
                            "answer": rev_rel,
                            "confidence": 1.0,
                        }
                    )

    # gather column info

    # mapping of column info
    type_map: Dict[tuple[str, str], str] = {}
    comment_map: Dict[tuple[str, str], str | None] = {}
    for t, info in schema.items():
        for col in info.columns:
            key = (t, col.name)
            type_map[key] = col.type_
            comment_map[key] = col.comment

    log.info(
        "Discovering relationships using heuristics with parallelism=%d",
        parallelism,
    )
    sem = asyncio.Semaphore(max(1, parallelism))
    lock = asyncio.Lock()

    async def _process(ftbl: str, fcol, rtbl: str, pk: str) -> None:
        ftype = type_map[(ftbl, fcol.name)]
        fcomment = (comment_map[(ftbl, fcol.name)] or "").lower()
        rtype = type_map[(rtbl, pk)]

        base_rtbl = rtbl.lower().rstrip("s")
        s1 = (
            max(
                _name_similarity(fcol.name, pk),
                _name_similarity(fcol.name, f"{rtbl}_{pk}"),
                _name_similarity(fcol.name, f"{base_rtbl}_{pk}"),
                _name_similarity(fcol.name, rtbl),
                _name_similarity(fcol.name, base_rtbl),
            )
            >= 0.8
            or base_rtbl in fcol.name.lower()
        )
        s2 = _same_type(ftype, rtype)
        mention_target = base_rtbl
        s3 = mention_target in fcomment

        async with sem:
            s4_task = asyncio.create_task(
                _values_contained(engine, ftbl, fcol.name, rtbl, pk, sample_limit)
            )
            ratio_task = asyncio.create_task(_distinct_ratio(engine, ftbl, fcol.name))
            s4, ratio = await asyncio.gather(s4_task, ratio_task)
        s5 = ratio < 0.8
        s6 = fcol.name.lower() != "id"

        score = sum([s1, s2, s3, s4, s5, s6])
        log.debug(
            "%s.%s -> %s.%s signals: %s score=%d",
            ftbl,
            fcol.name,
            rtbl,
            pk,
            [s1, s2, s3, s4, s5, s6],
            score,
        )

        if score < THRESHOLD:
            return

        gpt = await _gpt_second_opinion(ftbl, fcol.name, rtbl, pk, score)
        if gpt != "yes":
            return

        async with sem:
            ok = await _no_orphans(engine, ftbl, fcol.name, rtbl, pk)
        if not ok:
            return

        sql = f"SELECT 1 FROM {ftbl} a JOIN {rtbl} b ON a.{fcol.name} = b.{pk} LIMIT 1"
        validator_ok = True
        if validator:
            try:
                validator_ok, _ = await asyncio.to_thread(validator.check, sql)
            except Exception as err:  # pragma: no cover - unexpected errors
                log.exception("Validation check failed: %s", err)
                validator_ok = False
            else:
                log.info("Validation of relationship SQL '%s': %s", sql, validator_ok)
        else:
            log.info("Validation skipped for SQL '%s'", sql)
        if not validator_ok:
            return

        rel = f"{ftbl}.{fcol.name} -> {rtbl}.{pk}"
        async with lock:
            if rel not in seen:
                seen.add(rel)
                results.append(
                    {
                        "question": f"How is {ftbl}.{fcol.name} related to {rtbl}.{pk}?",
                        "answer": rel,
                        "confidence": score / 6,
                    }
                )
            rev_rel = f"{rtbl}.{pk} -> {ftbl}.{fcol.name}"
            if rev_rel not in seen:
                seen.add(rev_rel)
                results.append(
                    {
                        "question": f"How is {rtbl}.{pk} related to {ftbl}.{fcol.name}?",
                        "answer": rev_rel,
                        "confidence": score / 6,
                    }
                )

    tasks = []
    for ftbl, finfo in schema.items():
        for fcol in finfo.columns:
            for rtbl, rinfo in schema.items():
                if ftbl == rtbl:
                    continue
                pk = rinfo.primary_key
                if not pk:
                    continue
                tasks.append(_process(ftbl, fcol, rtbl, pk))

    await asyncio.gather(*tasks)

    # ----------------------- step 3: GPT suggestions -------------------
    log.info("Fetching additional relationships from GPT")
    gpt_sqls = await _gpt_relationship_sqls(schema, engine)
    pattern = re.compile(
        r"SELECT\s+1\s+FROM\s+(\w+)\s+a\s+JOIN\s+(\w+)\s+b\s+ON\s+a\.(\w+)\s*=\s*b\.(\w+)",
        re.IGNORECASE,
    )
    for sql in gpt_sqls:
        validator_ok = True
        if validator:
            try:
                validator_ok, _ = await asyncio.to_thread(validator.check, sql)
            except Exception as err:  # pragma: no cover - unexpected errors
                log.exception("Validation check failed: %s", err)
                validator_ok = False
            else:
                log.info(
                    "Validation of GPT relationship SQL '%s': %s", sql, validator_ok
                )
        else:
            log.info("Validation skipped for SQL '%s'", sql)
        if not validator_ok:
            continue
        m = pattern.search(sql)
        if not m:
            continue
        child, parent, col, pk = m.groups()
        rel = f"{child}.{col} -> {parent}.{pk}"
        if rel not in seen:
            seen.add(rel)
            results.append(
                {
                    "question": f"How is {child}.{col} related to {parent}.{pk}?",
                    "answer": rel,
                    "confidence": 0.5,
                }
            )
        rev_rel = f"{parent}.{pk} -> {child}.{col}"
        if rev_rel not in seen:
            seen.add(rev_rel)
            results.append(
                {
                    "question": f"How is {parent}.{pk} related to {child}.{col}?",
                    "answer": rev_rel,
                    "confidence": 0.5,
                }
            )

    results.sort(key=lambda r: r["confidence"], reverse=True)
    log.info("Discovered %d relationships", len(results))
    return results
