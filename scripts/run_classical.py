"""Train and evaluate the classical TF-IDF baselines on NoReC.

For each (classifier, class_weight) config, sweeps C on dev and reports the
best model on test. Also prints a majority-class baseline as a sanity floor.

Outputs:
    results/classical.json           — full per-run metrics
    results/classical_preds__*.csv   — per-config test predictions for error analysis

Usage: uv run python scripts/run_classical.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thesis.classical import make_pipeline  # noqa: E402
from thesis.data import LABEL_NAMES, get_split, load_norec  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
C_GRID = [0.1, 1.0, 10.0]
CLASSIFIERS = ["logreg", "svm"]
BALANCED_OPTIONS = [False, True]


def majority_baseline(y_train, y_test):
    maj = int(np.bincount(y_train).argmax())
    y_pred = np.full_like(y_test, maj)
    return compute_metrics(y_test, y_pred, LABEL_NAMES)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NoReC...")
    df = load_norec()
    train, dev, test = (get_split(df, s) for s in ("train", "dev", "test"))
    Xtr, ytr = train["text"].tolist(), train["label"].to_numpy()
    Xdv, ydv = dev["text"].tolist(), dev["label"].to_numpy()
    Xte, yte = test["text"].tolist(), test["label"].to_numpy()
    print(f"  train={len(Xtr):,}  dev={len(Xdv):,}  test={len(Xte):,}\n")

    print("Majority-class baseline (predict 'positive' on test):")
    maj = majority_baseline(ytr, yte)
    print(f"  acc={maj['accuracy']:.4f}  macro-F1={maj['macro_f1']:.4f}\n")

    runs = [{"model": "majority", "balanced": None, "best_C": None, "test": maj}]

    for clf_name in CLASSIFIERS:
        for balanced in BALANCED_OPTIONS:
            tag = f"{clf_name} balanced={balanced}"
            print(f"=== {tag} ===")
            best = None
            sweep: list[dict] = []
            for C in C_GRID:
                t0 = time.time()
                pipe = make_pipeline(clf_name, C, balanced)
                pipe.fit(Xtr, ytr)
                fit_s = time.time() - t0
                yhat_dev = pipe.predict(Xdv)
                dev_metrics = compute_metrics(ydv, yhat_dev, LABEL_NAMES)
                print(
                    f"  C={C:>5}  fit={fit_s:5.1f}s  "
                    f"dev acc={dev_metrics['accuracy']:.4f}  "
                    f"dev macro-F1={dev_metrics['macro_f1']:.4f}"
                )
                sweep.append(
                    {
                        "C": C,
                        "fit_seconds": fit_s,
                        "dev_accuracy": dev_metrics["accuracy"],
                        "dev_macro_f1": dev_metrics["macro_f1"],
                    }
                )
                if best is None or dev_metrics["macro_f1"] > best["dev"]["macro_f1"]:
                    best = {
                        "C": C,
                        "fit_s": fit_s,
                        "dev": dev_metrics,
                        "pipe": pipe,
                    }

            t0 = time.time()
            yhat_test = best["pipe"].predict(Xte)
            pred_s = time.time() - t0
            test_metrics = compute_metrics(yte, yhat_test, LABEL_NAMES)
            print(
                f"  --> chosen C={best['C']}  "
                f"test acc={test_metrics['accuracy']:.4f}  "
                f"test macro-F1={test_metrics['macro_f1']:.4f}\n"
            )

            preds_df = pd.DataFrame(
                {"id": test["id"].to_numpy(), "label": yte, "pred": yhat_test}
            )
            preds_path = (
                RESULTS_DIR
                / f"classical_preds__{clf_name}__balanced={balanced}.csv"
            )
            preds_df.to_csv(preds_path, index=False)

            runs.append(
                {
                    "model": clf_name,
                    "balanced": balanced,
                    "best_C": best["C"],
                    "fit_seconds": best["fit_s"],
                    "predict_seconds_test": pred_s,
                    "dev_sweep": sweep,
                    "dev": best["dev"],
                    "test": test_metrics,
                }
            )

    out_path = RESULTS_DIR / "classical.json"
    with out_path.open("w") as f:
        json.dump(runs, f, indent=2)
    print(f"Wrote {out_path}\n")

    print("=== Summary ===")
    header = f"{'model':<10} {'balanced':<9} {'C':<5} {'test acc':<10} {'test F1':<10}"
    print(header)
    print("-" * len(header))
    for r in runs:
        bal = "" if r["balanced"] is None else str(r["balanced"])
        C = "" if r["best_C"] is None else str(r["best_C"])
        print(
            f"{r['model']:<10} {bal:<9} {C:<5} "
            f"{r['test']['accuracy']:<10.4f} {r['test']['macro_f1']:<10.4f}"
        )


if __name__ == "__main__":
    main()
