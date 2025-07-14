import asyncio

from nl_sql_generator.worker_agent import _parse_pairs, WorkerAgent
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


class DummyClient:
    def __init__(self):
        self.calls = []

    async def acomplete(self, messages, return_message=False, model=None):
        self.calls.append(messages)
        text = "\n".join(
            [f'{{"question": "Q{len(self.calls)}-{i}", "answer": "A{i}"}}' for i in range(2)]
        )
        if return_message:
            return {"role": "assistant", "content": text}
        return text


def test_parse_pairs_basic():
    text = '{"question": "Q1", "answer": "A1"}\n{"question": "Q2", "answer": "A2"}'
    pairs = _parse_pairs(text)
    assert pairs == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


def test_parse_pairs_with_noise():
    text = """Here you go:
```json
{"question": "Q1", "answer": "A1"}
{"question": "Q2", "answer": "A2"}
```
- invalid
"""
    pairs = _parse_pairs(text)
    assert pairs == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


def test_generate_multiple_requests():
    client = DummyClient()
    schema = {
        "t": TableInfo("t", [ColumnInfo("id", "int")])
    }
    agent = WorkerAgent(schema, {"api_answer_count": 2}, lambda: None, 1, client)
    pairs = asyncio.run(agent.generate(3))

    assert len(pairs) == 3
    # Two API calls expected since api_answer_count=2 and k=3
    assert len(client.calls) == 2
    # Schema JSON should only appear once across the chat history
    call2 = client.calls[1]
    schema_mentions = sum(
        "SCHEMA_JSON" in m.get("content", "") for m in call2
    )
    assert schema_mentions == 1
