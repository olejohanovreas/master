"""Chunk-and-pool evaluation of a fine-tuned NB-BERT-base on NoReC test.

Loads a saved checkpoint (default: seed 42 final model produced by
run_finetune_nbbert.py), tokenises every test review without truncation,
splits long reviews into overlapping 512-token windows, and averages logits
across windows per document. The result is a like-for-like comparison with
the standard right-truncated 512-token pipeline, isolating the effect of
the truncation strategy on long-review performance.

Usage:
    uv run python scripts/eval_nbbert_chunked.py [--checkpoint <path>]
                                                 [--max_length 512]
                                                 [--stride 256]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from thesis.data import LABEL_NAMES, get_split, load_norec  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402
from thesis.transformer import predict_chunked  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "nbbert-base-seed42" / "final"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    if not args.checkpoint.exists():
        raise SystemExit(
            f"checkpoint dir not found: {args.checkpoint}\n"
            "Run scripts/run_finetune_nbbert.py first to produce it."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print("Loading NoReC test split...")
    df = load_norec()
    test_df = get_split(df, "test")
    print(f"  test size: {len(test_df)}")

    print(
        f"Running chunked inference  max_length={args.max_length} "
        f"stride={args.stride}"
    )
    t0 = time.time()
    preds = predict_chunked(
        test_df["text"].tolist(),
        model, tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"  inference time: {elapsed:.1f}s")

    y_true = test_df["label"].to_numpy()
    metrics = compute_metrics(y_true, preds, LABEL_NAMES)
    print(
        f"\nTest acc:    {metrics['accuracy']:.4f}\n"
        f"Macro-F1:    {metrics['macro_f1']:.4f}\n"
        f"Negative-F1: {metrics['per_class']['negative']['f1']:.4f}\n"
        f"Positive-F1: {metrics['per_class']['positive']['f1']:.4f}"
    )

    preds_path = RESULTS_DIR / "nbbert_preds_chunked.csv"
    pd.DataFrame(
        {"id": test_df["id"].to_numpy(), "label": y_true, "pred": preds}
    ).to_csv(preds_path, index=False)

    out = {
        "model": "NB-BERT-base (chunk-and-pool)",
        "config": {
            "checkpoint": str(args.checkpoint),
            "max_length": args.max_length,
            "stride": args.stride,
        },
        "elapsed_seconds": elapsed,
        "test": metrics,
    }
    out_path = RESULTS_DIR / "nbbert_chunked.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Wrote {preds_path}")


if __name__ == "__main__":
    main()
