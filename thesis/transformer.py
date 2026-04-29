"""Helpers for fine-tuning HuggingFace transformers on NoReC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import Dataset

from .data import LABEL_NAMES
from .evaluation import compute_metrics


def tokenize_dataframe(
    df: pd.DataFrame, tokenizer, max_length: int = 512
) -> Dataset:
    """Build an HF Dataset of tokenized inputs + labels from our pandas frame."""
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def hf_compute_metrics(eval_pred) -> dict:
    """Adapter: convert HF Trainer's EvalPrediction to a flat scalar metric dict."""
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(eval_pred.label_ids, preds, LABEL_NAMES)
    flat = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    for cls, vals in m["per_class"].items():
        flat[f"{cls}_f1"] = vals["f1"]
        flat[f"{cls}_precision"] = vals["precision"]
        flat[f"{cls}_recall"] = vals["recall"]
    return flat
