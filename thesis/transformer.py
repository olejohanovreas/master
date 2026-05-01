"""Helpers for fine-tuning HuggingFace transformers on NoReC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
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


def predict_chunked(
    texts: list[str],
    model,
    tokenizer,
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Chunk-and-pool inference over long documents.

    Each document is tokenized without truncation and split into overlapping
    windows of `max_length` tokens (with `stride` step). Logits are averaged
    across all windows of a document and argmax gives the per-document
    prediction. Documents shorter than `max_length` collapse to a single
    forward pass.
    """
    model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    predictions: list[int] = []
    for text in texts:
        encoded = tokenizer(
            text, return_tensors="pt", truncation=False, padding=False,
            add_special_tokens=True,
        )
        ids = encoded["input_ids"][0]
        n_tokens = ids.shape[0]

        if n_tokens <= max_length:
            chunks = [ids]
        else:
            chunks = []
            start = 0
            while start < n_tokens:
                chunks.append(ids[start : start + max_length])
                if start + max_length >= n_tokens:
                    break
                start += stride

        n_chunks = len(chunks)
        padded = torch.full((n_chunks, max_length), pad_id, dtype=torch.long)
        attn = torch.zeros((n_chunks, max_length), dtype=torch.long)
        for i, c in enumerate(chunks):
            L = c.shape[0]
            padded[i, :L] = c
            attn[i, :L] = 1

        padded = padded.to(device)
        attn = attn.to(device)

        chunk_logits = []
        for bs in range(0, n_chunks, batch_size):
            with torch.no_grad():
                out = model(
                    input_ids=padded[bs : bs + batch_size],
                    attention_mask=attn[bs : bs + batch_size],
                )
            chunk_logits.append(out.logits.float())
        logits = torch.cat(chunk_logits, dim=0)
        avg = logits.mean(dim=0)
        predictions.append(int(torch.argmax(avg).item()))

    return np.array(predictions)
