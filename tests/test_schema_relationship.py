import asyncio
import pandas as pd
import pytest
from nl_sql_generator.schema_relationship import _analyze_pair, discover_relationships
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


def test_analyze_pair_overlap():
    df1 = pd.DataFrame({"id": [1, 2, 3]})
    df2 = pd.DataFrame({"id": [2, 3, 4]})
    rels = _analyze_pair("a", df1, "b", df2, pk1="id", pk2="id")
    assert {r["relationship"] for r in rels} == {"a.id -> b.id"}


def test_analyze_pair_low_overlap():
    df1 = pd.DataFrame({"id": [1, 2, 3]})
    df2 = pd.DataFrame({"id": [3, 4, 5]})
    rels = _analyze_pair("a", df1, "b", df2, pk1="id", pk2="id")
    assert rels == []


def test_analyze_pair_type_mismatch():
    df1 = pd.DataFrame({"id": [1, 2, 3]})
    df2 = pd.DataFrame({"val": ["1", "2", "3"]})
    rels = _analyze_pair("a", df1, "b", df2, pk1="id", pk2=None)
    assert rels == []


def test_analyze_pair_pk_match():
    df1 = pd.DataFrame({"id": [1, 2, 3]})
    df2 = pd.DataFrame({"a_id": [1, 2, 4]})
    rels = _analyze_pair("a", df1, "b", df2, pk1="id", pk2=None)
    assert {r["relationship"] for r in rels} == {"a.id -> b.a_id"}


class DummyEngine:
    pass


def test_discover_relationships(monkeypatch):
    async def fake_fetch(engine, table, n_rows):
        if table == "t1":
            return pd.DataFrame({"id": [1, 2, 3]})
        return pd.DataFrame({"id": [1, 3, 5]})

    monkeypatch.setattr("nl_sql_generator.schema_relationship._fetch_rows", fake_fetch)
    schema = {
        "t1": TableInfo("t1", [ColumnInfo("id", "int")], "id"),
        "t2": TableInfo("t2", [ColumnInfo("id", "int")], "id"),
    }
    engine = DummyEngine()
    pairs = asyncio.run(discover_relationships(schema, engine, n_rows=3, parallelism=1))
    assert {p["relationship"] for p in pairs} == {"t1.id -> t2.id"}


def test_score_relation_uuid():
    import uuid
    from nl_sql_generator.schema_relationship import _score_relation

    s = pd.Series([uuid.uuid4() for _ in range(5)])
    assert _score_relation(s, s) == pytest.approx(1.0)
