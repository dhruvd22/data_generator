import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nl_sql_generator.schema_loader import SchemaLoader


class DummyInspector:
    def __init__(self):
        pass

    def get_table_names(self):
        return ["tbl"]

    def get_columns(self, table):
        return [{"name": "id", "type": "int", "comment": "pk"}]

    def get_pk_constraint(self, table):
        return {"constrained_columns": ["id"]}

    def get_table_comment(self, table):
        return {"text": "tbl comment"}

    def get_foreign_keys(self, table):
        return [
            {
                "referred_table": "parent",
                "constrained_columns": ["parent_id"],
                "referred_columns": ["id"],
            }
        ]


class DummyEngine:
    pass


def test_load_schema_with_comments(monkeypatch):
    inspector = DummyInspector()
    monkeypatch.setattr(
        "nl_sql_generator.schema_loader.create_engine", lambda u, pool_pre_ping=True: DummyEngine()
    )
    monkeypatch.setattr("nl_sql_generator.schema_loader.inspect", lambda e: inspector)

    os.environ["DATABASE_URL"] = "pg://x"
    schema = SchemaLoader.load_schema()
    info = schema["tbl"]
    assert info.comment == "tbl comment"
    assert info.columns[0].comment == "pk"
    assert info.foreign_keys[0]["referred_table"] == "parent"
