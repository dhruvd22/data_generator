import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.schema_relationship import discover_relationships
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


class DummyInspector:
    def __init__(self, fks=None):
        self._fks = fks or {}

    def get_foreign_keys(self, table):
        return self._fks.get(table, [])

    def get_pk_constraint(self, table):
        return {"constrained_columns": ["id"]}

    def get_unique_constraints(self, table):
        return []

    def get_indexes(self, table):
        return []


class DummyEngine:
    pass


async def _noop_comment_similarity(*args, **kwargs):
    return 0.0


async def _always_contained(*args, **kwargs):
    return True


def test_fk_relationship(monkeypatch):
    inspector = DummyInspector(
        {
            "a": [
                {"referred_table": "b", "constrained_columns": ["b_id"], "referred_columns": ["id"]}
            ]
        }
    )
    monkeypatch.setattr("nl_sql_generator.schema_relationship.inspect", lambda e: inspector)
    monkeypatch.setattr(
        "nl_sql_generator.schema_relationship._comment_similarity", _noop_comment_similarity
    )
    monkeypatch.setattr("nl_sql_generator.schema_relationship._values_contained", _always_contained)

    schema = {
        "a": TableInfo("a", [ColumnInfo("b_id", "int")], None),
        "b": TableInfo("b", [ColumnInfo("id", "int")], "id"),
    }
    rels = asyncio.run(discover_relationships(schema, DummyEngine()))
    assert rels[0]["relationship"] == "a.b_id -> b.id"
    assert rels[0]["confidence"] == 1.0


def test_heuristic_relationship(monkeypatch):
    inspector = DummyInspector()
    monkeypatch.setattr("nl_sql_generator.schema_relationship.inspect", lambda e: inspector)

    async def _sim(*args, **kwargs):
        return 0.9

    async def _val(*args, **kwargs):
        return True

    monkeypatch.setattr("nl_sql_generator.schema_relationship._comment_similarity", _sim)
    monkeypatch.setattr("nl_sql_generator.schema_relationship._values_contained", _val)

    schema = {
        "a": TableInfo("a", [ColumnInfo("b_id", "int", "ref")], None),
        "b": TableInfo("b", [ColumnInfo("id", "int", "pk")], "id"),
    }
    rels = asyncio.run(discover_relationships(schema, DummyEngine()))
    assert rels[0]["relationship"] == "a.b_id -> b.id"
    assert rels[0]["confidence"] > 0.9


def test_reject_low_similarity(monkeypatch):
    inspector = DummyInspector()
    monkeypatch.setattr("nl_sql_generator.schema_relationship.inspect", lambda e: inspector)
    monkeypatch.setattr(
        "nl_sql_generator.schema_relationship._comment_similarity", _noop_comment_similarity
    )
    monkeypatch.setattr("nl_sql_generator.schema_relationship._values_contained", _always_contained)
    schema = {
        "a": TableInfo("a", [ColumnInfo("foo", "int")], None),
        "b": TableInfo("b", [ColumnInfo("id", "int")], "id"),
    }
    rels = asyncio.run(discover_relationships(schema, DummyEngine()))
    assert rels == []
