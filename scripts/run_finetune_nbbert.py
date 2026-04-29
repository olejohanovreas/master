"""Fine-tune NB-BERT-base on NoReC for binary sentiment.

Selects the best checkpoint by dev macro-F1, evaluates it on test, and writes
predictions plus metrics to results/. Designed to run on the V100.

Usage:
    uv run python scripts/run_finetune_nbbert.py [--smoke_test] [--epochs 3]
                                                 [--batch_size 32] [--lr 2e-5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from thesis.data import LABEL_NAMES, get_split, load_norec  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402
from thesis.transformer import hf_compute_metrics, tokenize_dataframe  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints" / "nbbert-base"
MODEL_NAME = "NbAiLab/nb-bert-base"
MAX_LENGTH = 512
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Subsample heavily and run 1 epoch — for verifying the pipeline.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NoReC...")
    df = load_norec()
    train_df = get_split(df, "train")
    dev_df = get_split(df, "dev")
    test_df = get_split(df, "test")

    if args.smoke_test:
        print("[smoke_test] subsampling train=200 dev=100 test=100, epochs=1")
        train_df = train_df.sample(200, random_state=SEED).reset_index(drop=True)
        dev_df = dev_df.sample(100, random_state=SEED).reset_index(drop=True)
        test_df = test_df.sample(100, random_state=SEED).reset_index(drop=True)
        args.epochs = 1

    print(
        f"sizes: train={len(train_df):,} "
        f"dev={len(dev_df):,} test={len(test_df):,}"
    )

    print(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )

    print("Tokenizing...")
    train_ds = tokenize_dataframe(train_df, tokenizer, MAX_LENGTH)
    dev_ds = tokenize_dataframe(dev_df, tokenizer, MAX_LENGTH)
    test_ds = tokenize_dataframe(test_df, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics,
    )

    print("Training...")
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"  train wall time: {train_seconds:.1f}s")

    print("Eval on dev (best checkpoint, reloaded):")
    dev_metrics_flat = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    for k, v in dev_metrics_flat.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("Predict on test:")
    test_pred = trainer.predict(test_ds, metric_key_prefix="test")
    logits = (
        test_pred.predictions[0]
        if isinstance(test_pred.predictions, tuple)
        else test_pred.predictions
    )
    test_yhat = np.argmax(logits, axis=-1)
    test_y = np.array(test_df["label"])
    test_metrics = compute_metrics(test_y, test_yhat, LABEL_NAMES)

    preds_path = RESULTS_DIR / "nbbert_preds.csv"
    pd.DataFrame(
        {"id": test_df["id"].to_numpy(), "label": test_y, "pred": test_yhat}
    ).to_csv(preds_path, index=False)

    out = {
        "model": MODEL_NAME,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "smoke_test": args.smoke_test,
        },
        "train_seconds": train_seconds,
        "dev": {k: float(v) for k, v in dev_metrics_flat.items() if isinstance(v, (int, float))},
        "test": test_metrics,
    }
    out_path = RESULTS_DIR / "nbbert.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Wrote {preds_path}")
    print("\n=== Summary ===")
    print(
        f"test acc: {test_metrics['accuracy']:.4f}  "
        f"test macro-F1: {test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
