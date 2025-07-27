from __future__ import annotations

"""Helpers for converting generated JSONL datasets into model features."""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

__all__ = ["prepare_dataset"]


def prepare_dataset(
    jsonl_path: str,
    model_name: str,
    *,
    max_length: int = 512,
):
    """Return tokenized dataset and collator for training.

    This utility loads a dataset written by :class:`ResultWriter` and applies
    tokenization with truncation and padding so the trainer receives tensors of
    consistent shapes.

    Args:
        jsonl_path: Path to the generated ``dataset.jsonl`` file.
        model_name: Name of the HuggingFace model tokenizer.
        max_length: Optional maximum sequence length.

    Returns:
        A tuple of ``datasets.Dataset`` and :class:`DataCollatorWithPadding`.
    """

    ds = load_dataset("json", data_files=jsonl_path, split="train")
    tok = AutoTokenizer.from_pretrained(model_name)

    def _tok(example):
        q = example.get("question")
        sql = example.get("sql") or example.get("answer", "")
        model_inputs = tok(
            q,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = tok(
            sql,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(_tok, remove_columns=ds.column_names)
    collator = DataCollatorWithPadding(tok, padding=True, return_tensors="pt")
    return ds, collator

