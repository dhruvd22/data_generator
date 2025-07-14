import asyncio

from nl_sql_generator.worker_agent import _parse_pairs, WorkerAgent
from nl_sql_generator.schema_loader import TableInfo, ColumnInfo


class DummyClient:
    def __init__(self):
        self.calls = []

    async def acomplete(self, messages, return_message=False, model=None):
        self.calls.append(messages)
        text = "\n".join(
            [
                f'{{"question": "Q{len(self.calls)}-{i}", "answer": "A{i}"}}'
                for i in range(2)
            ]
        )
        if return_message:
            return {"role": "assistant", "content": text}
        return text


class SinglePairClient(DummyClient):
    async def acomplete(self, messages, return_message=False, model=None):
        self.calls.append(messages)
        text = "\n".join(
            [
                f'{{"question": "Q{len(self.calls)}-{i}", "answer": "A{i}"}}'
                for i in range(1)
            ]
        )
        if return_message:
            return {"role": "assistant", "content": text}
        return text


class EmptyClient(DummyClient):
    async def acomplete(self, messages, return_message=False, model=None):
        self.calls.append(messages)
        if return_message:
            return {"role": "assistant", "content": ""}
        return ""


class FullBatchClient(DummyClient):
    def __init__(self, n):
        super().__init__()
        self.n = n

    async def acomplete(self, messages, return_message=False, model=None):
        self.calls.append(messages)
        text = "\n".join(
            [f'{{"question": "Q0-{i}", "answer": "A{i}"}}' for i in range(self.n)]
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


def test_parse_pairs_json_array():
    text = """[
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"}
    ]"""
    pairs = _parse_pairs(text)
    assert pairs == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


def test_parse_pairs_with_preamble():
    text = """Sure, here are some pairs:
```json
[
  {"question": "Q1", "answer": "A1"},
  {"question": "Q2", "answer": "A2"}
]
```"""
    pairs = _parse_pairs(text)
    assert pairs == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


def test_generate_multiple_requests():
    client = DummyClient()
    schema = {"t": TableInfo("t", [ColumnInfo("id", "int")])}
    agent = WorkerAgent(schema, {"api_answer_count": 2}, lambda: None, 1, client)
    pairs = asyncio.run(agent.generate(3))

    assert len(pairs) == 3
    # Two API calls expected since api_answer_count=2 and k=3
    assert len(client.calls) == 2
    # Schema JSON should only appear once across the chat history
    call2 = client.calls[1]
    schema_mentions = sum("SCHEMA_JSON" in m.get("content", "") for m in call2)
    assert schema_mentions == 1


def test_schema_included_once_per_call():
    client = SinglePairClient()
    schema = {"t": TableInfo("t", [ColumnInfo("id", "int")])}
    agent = WorkerAgent(schema, {"api_answer_count": 1}, lambda: None, 1, client)
    pairs = asyncio.run(agent.generate(3))

    assert len(pairs) == 3
    assert len(client.calls) == 3
    for call in client.calls:
        schema_mentions = sum("SCHEMA_JSON" in m.get("content", "") for m in call)
        assert schema_mentions == 1


def test_generate_respects_max_attempts():
    client = EmptyClient()
    schema = {"t": TableInfo("t", [ColumnInfo("id", "int")])}
    agent = WorkerAgent(
        schema,
        {"api_answer_count": 2, "max_attempts": 2},
        lambda: None,
        1,
        client,
    )
    pairs = asyncio.run(agent.generate(3))

    assert pairs == []
    assert len(client.calls) == 2


def test_stop_when_batch_met():
    k = 2
    client = FullBatchClient(k)
    schema = {"t": TableInfo("t", [ColumnInfo("id", "int")])}
    agent = WorkerAgent(schema, {"api_answer_count": k}, lambda: None, 1, client)
    pairs = asyncio.run(agent.generate(k))

    assert len(pairs) == k
    assert len(client.calls) == 1


def test_remaining_uses_pairs_received():
    client = SinglePairClient()
    schema = {"t": TableInfo("t", [ColumnInfo("id", "int")])}
    agent = WorkerAgent(schema, {"api_answer_count": 2}, lambda: None, 1, client)
    pairs = asyncio.run(agent.generate(3))

    assert len(pairs) == 3
    assert len(client.calls) >= 2
    # Second request should ask for only one additional pair
    assert "Generate 1 more" in client.calls[1][-2]["content"]
