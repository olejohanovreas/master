"""Zero-shot and few-shot prompting eval on NoReC test for instruction-tuned LLMs.

Runs each (model, regime) combo against the held-out test set, using greedy
decoding and a fixed prompt. Saves per-config predictions plus aggregate metrics.

Default model set is the Llama 3 Instruct family (small/mid/large within one
family). Llama models on HF Hub are gated; before running, log in once on the
V100 with `uv run hf auth login`.

Usage:
    uv run python scripts/run_llm.py [--smoke_test] [--batch_size 8]
                                     [--models meta-llama/Llama-3.2-1B-Instruct ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from thesis.data import LABEL_NAMES, get_split, load_norec  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402
from thesis.llm import (  # noqa: E402
    FewShotExample,
    build_messages,
    parse_response,
    select_few_shot_examples,
    truncate_text_to_tokens,
)

RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
MAX_REVIEW_TOKENS = 1500  # cap each review BEFORE chat-template assembly
MAX_INPUT_TOKENS = 4096   # safety budget for full prompt; never truncate the assistant marker
MAX_NEW_TOKENS = 4
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Subsample test heavily to verify the pipeline.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip any (model, regime) whose predictions CSV already exists.",
    )
    return p.parse_args()


def predictions_path(model_name: str, regime: str) -> Path:
    short = model_name.split("/")[-1]
    return RESULTS_DIR / f"llm_preds__{short}__{regime}.csv"


def write_summary(
    runs: list[dict], few_shot: list[FewShotExample], args: argparse.Namespace
) -> None:
    """Atomically write the aggregate llm.json from current runs (called after each run)."""
    out = {
        "few_shot_examples": [asdict(ex) for ex in few_shot],
        "config": {
            "max_review_tokens": MAX_REVIEW_TOKENS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": args.batch_size,
            "decoding": "greedy",
            "smoke_test": args.smoke_test,
            "seed": SEED,
        },
        "runs": runs,
    }
    out_path = RESULTS_DIR / "llm.json"
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    tmp.replace(out_path)


def generate_predictions(
    texts: list[str],
    few_shot: list[FewShotExample] | None,
    tokenizer,
    model,
    batch_size: int,
) -> tuple[list[int | None], list[str]]:
    """Return (parsed_predictions, raw_responses) for each input text."""
    parsed: list[int | None] = []
    raws: list[str] = []
    device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        truncated = [
            truncate_text_to_tokens(t, tokenizer, MAX_REVIEW_TOKENS)
            for t in batch_texts
        ]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(t, few_shot),
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in truncated
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for d in decoded:
            raws.append(d)
            parsed.append(parse_response(d))

        if (start // batch_size) % 25 == 0:
            done = start + len(batch_texts)
            print(f"    progress: {done}/{len(texts)}", flush=True)
    return parsed, raws


def evaluate_model_regime(
    model_name: str,
    regime: str,
    test_df: pd.DataFrame,
    few_shot: list[FewShotExample],
    tokenizer,
    model,
    batch_size: int,
) -> dict:
    """Run one (model, regime) configuration and return a result record."""
    examples = few_shot if regime == "few-shot" else None
    texts = test_df["text"].tolist()
    y_true = test_df["label"].to_numpy()

    print(f"  [{regime}] generating on {len(texts)} examples...", flush=True)
    t0 = time.time()
    parsed, raws = generate_predictions(
        texts, examples, tokenizer, model, batch_size
    )
    elapsed = time.time() - t0

    n_unparseable = sum(1 for p in parsed if p is None)
    fallback = int(np.bincount(y_true).argmax())  # majority class as fallback
    y_pred = np.array([p if p is not None else fallback for p in parsed])
    metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)

    print(
        f"  [{regime}] acc={metrics['accuracy']:.4f} "
        f"macro-F1={metrics['macro_f1']:.4f} "
        f"unparseable={n_unparseable}/{len(parsed)} "
        f"({elapsed:.0f}s)",
        flush=True,
    )

    preds_df = pd.DataFrame(
        {
            "id": test_df["id"].to_numpy(),
            "label": y_true,
            "pred": y_pred,
            "raw": raws,
            "parsed": [p is not None for p in parsed],
        }
    )
    preds_df.to_csv(predictions_path(model_name, regime), index=False)

    return {
        "model": model_name,
        "regime": regime,
        "elapsed_seconds": elapsed,
        "n_examples": len(parsed),
        "n_unparseable": n_unparseable,
        "fallback_label_for_unparseable": LABEL_NAMES[fallback],
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NoReC...")
    df = load_norec()
    train_df = get_split(df, "train")
    test_df = get_split(df, "test")
    if args.smoke_test:
        test_df = test_df.sample(50, random_state=SEED).reset_index(drop=True)
        print(f"[smoke_test] test subsampled to {len(test_df)}")

    few_shot = select_few_shot_examples(train_df, n_per_class=2, seed=SEED)
    print(f"Few-shot examples ({len(few_shot)}):")
    for ex in few_shot:
        wc = len(ex.text.split())
        print(f"  id={ex.review_id} label={ex.label_name} ({wc} words)")

    runs: list[dict] = []
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        regimes_to_run = [
            r for r in ("zero-shot", "few-shot")
            if not (args.skip_existing and predictions_path(model_name, r).exists())
        ]
        if not regimes_to_run:
            print(f"  skipping (predictions exist for both regimes)")
            continue
        if args.skip_existing and len(regimes_to_run) < 2:
            skipped = set(("zero-shot", "few-shot")) - set(regimes_to_run)
            print(f"  skipping existing: {sorted(skipped)}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        for regime in regimes_to_run:
            run = evaluate_model_regime(
                model_name, regime, test_df, few_shot, tokenizer, model,
                args.batch_size,
            )
            runs.append(run)
            write_summary(runs, few_shot, args)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    seen = {(r["model"], r["regime"]) for r in runs}
    for model_name in args.models:
        for regime in ("zero-shot", "few-shot"):
            if (model_name, regime) in seen:
                continue
            path = predictions_path(model_name, regime)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            y_true = df["label"].to_numpy()
            y_pred = df["pred"].to_numpy()
            metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)
            fallback = int(np.bincount(y_true).argmax())
            runs.append(
                {
                    "model": model_name,
                    "regime": regime,
                    "n_examples": len(df),
                    "n_unparseable": int((~df["parsed"]).sum()),
                    "fallback_label_for_unparseable": LABEL_NAMES[fallback],
                    "metrics": metrics,
                    "loaded_from_csv": True,
                }
            )

    write_summary(runs, few_shot, args)
    print(f"\nWrote {RESULTS_DIR / 'llm.json'}")

    print("\n=== Summary ===")
    print(f"{'model':<40} {'regime':<11} {'acc':<8} {'macro-F1':<10} {'unparse':<8}")
    for r in runs:
        short = r["model"].split("/")[-1]
        print(
            f"{short:<40} {r['regime']:<11} "
            f"{r['metrics']['accuracy']:.4f}  "
            f"{r['metrics']['macro_f1']:.4f}    "
            f"{r['n_unparseable']:<8}"
        )


if __name__ == "__main__":
    main()
