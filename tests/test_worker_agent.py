from nl_sql_generator.worker_agent import _parse_pairs


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
