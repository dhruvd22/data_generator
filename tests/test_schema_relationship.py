import asyncio
import pandas as pd
from nl_sql_generator.schema_relationship import _analyze_pair, discover_relationships
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


def test_analyze_pair_overlap():
    df1 = pd.DataFrame({"id": [1, 2, 3]})
    df2 = pd.DataFrame({"id": [2, 3, 4]})
    rels = _analyze_pair("a", df1, "b", df2)
    assert {r["relationship"] for r in rels} == {"a.id -> b.id"}


class DummyEngine:
    pass


def test_discover_relationships(monkeypatch):
    async def fake_fetch(engine, table, n_rows):
        if table == "t1":
            return pd.DataFrame({"id": [1, 2, 3]})
        return pd.DataFrame({"id": [1, 3, 5]})

    monkeypatch.setattr("nl_sql_generator.schema_relationship._fetch_rows", fake_fetch)
    schema = {
        "t1": TableInfo("t1", [ColumnInfo("id", "int")]),
        "t2": TableInfo("t2", [ColumnInfo("id", "int")]),
    }
    engine = DummyEngine()
    pairs = asyncio.run(discover_relationships(schema, engine, n_rows=3, parallelism=1))
    assert {p["relationship"] for p in pairs} == {"t1.id -> t2.id"}
