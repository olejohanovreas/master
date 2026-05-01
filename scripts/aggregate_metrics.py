"""Build a single test-set metrics table from all experiment outputs.

Reads classical.json, every nbbert_seed*.json, nbbert_chunked.json (if
present), and every llm_preds__*__*__s*_p*.csv prediction file, aggregates
seeds with mean and std, and writes a unified CSV + Markdown summary that
can be cited directly in the thesis. Computing LLM metrics from CSVs (not
from the per-seed summary JSONs) is robust to the run_llm.py summary file
being overwritten when seeds were run with subsets of --models.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(REPO_ROOT))
from thesis.data import LABEL_NAMES  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402

LLM_PREDS_RE = re.compile(
    r"^llm_preds__(?P<model>[^_]+(?:-[^_]+)*)__"
    r"(?P<regime>[^_]+(?:-[^_]+)*)__s(?P<seed>\d+)_p(?P<prompt>\w+)\.csv$"
)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def fmt_mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return f"{mean:.{digits}f}"
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(var)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def collect_classical(rows: list[dict]) -> None:
    path = RESULTS_DIR / "classical.json"
    if not path.exists():
        return
    with path.open() as f:
        runs = json.load(f)
    for r in runs:
        if r["model"] == "majority":
            family, model, config = "baseline", "majority", "always positive"
        else:
            family = "classical"
            model = "Logistic Regression" if r["model"] == "logreg" else "Linear SVM"
            config = (
                f"class_weight={'balanced' if r['balanced'] else 'none'}, "
                f"C={r['best_C']}"
            )
        m = r["test"]
        rows.append(
            {
                "family": family, "model": model, "config": config,
                "n_seeds": 1,
                "accuracy": fmt_mean_std([m["accuracy"]]),
                "macro_f1": fmt_mean_std([m["macro_f1"]]),
                "neg_f1": fmt_mean_std([m["per_class"]["negative"]["f1"]]),
                "pos_f1": fmt_mean_std([m["per_class"]["positive"]["f1"]]),
                "n_unparseable": "",
            }
        )


def collect_nbbert(rows: list[dict]) -> None:
    seed_files = sorted(RESULTS_DIR.glob("nbbert_seed*.json"))
    if not seed_files:
        return
    summaries: list[dict] = []
    for p in seed_files:
        with p.open() as f:
            summaries.append(json.load(f))
    cfg = summaries[0]["config"]
    accs = [s["test"]["accuracy"] for s in summaries]
    f1s = [s["test"]["macro_f1"] for s in summaries]
    neg = [s["test"]["per_class"]["negative"]["f1"] for s in summaries]
    pos = [s["test"]["per_class"]["positive"]["f1"] for s in summaries]
    rows.append(
        {
            "family": "transformer",
            "model": "NB-BERT-base",
            "config": (
                f"{cfg['epochs']} ep, lr={cfg['lr']}, batch={cfg['batch_size']}, "
                f"max_len={cfg['max_length']}"
            ),
            "n_seeds": len(summaries),
            "accuracy": fmt_mean_std(accs),
            "macro_f1": fmt_mean_std(f1s),
            "neg_f1": fmt_mean_std(neg),
            "pos_f1": fmt_mean_std(pos),
            "n_unparseable": "",
        }
    )


def collect_chunked(rows: list[dict]) -> None:
    path = RESULTS_DIR / "nbbert_chunked.json"
    if not path.exists():
        return
    with path.open() as f:
        s = json.load(f)
    m = s["test"]
    cfg = s["config"]
    rows.append(
        {
            "family": "transformer",
            "model": "NB-BERT-base + chunk-and-pool",
            "config": f"max_len={cfg['max_length']}, stride={cfg['stride']}",
            "n_seeds": 1,
            "accuracy": fmt_mean_std([m["accuracy"]]),
            "macro_f1": fmt_mean_std([m["macro_f1"]]),
            "neg_f1": fmt_mean_std([m["per_class"]["negative"]["f1"]]),
            "pos_f1": fmt_mean_std([m["per_class"]["positive"]["f1"]]),
            "n_unparseable": "",
        }
    )


def collect_llm(rows: list[dict]) -> None:
    """Read every llm_preds__*__*__s*_p*.csv and aggregate by (model, regime, prompt)."""
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for p in sorted(RESULTS_DIR.glob("llm_preds__*__s*_p*.csv")):
        m = LLM_PREDS_RE.match(p.name)
        if not m:
            continue
        df = pd.read_csv(p)
        y_true = df["label"].to_numpy()
        y_pred = df["pred"].to_numpy()
        metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)
        n_unparse = int((~df["parsed"]).sum()) if "parsed" in df.columns else 0
        key = (m["model"], m["regime"], m["prompt"])
        grouped.setdefault(key, []).append(
            {
                "seed": int(m["seed"]),
                "metrics": metrics,
                "n_unparseable": n_unparse,
            }
        )

    for (model_short, regime, prompt), runs_for_key in grouped.items():
        accs = [r["metrics"]["accuracy"] for r in runs_for_key]
        f1s = [r["metrics"]["macro_f1"] for r in runs_for_key]
        neg = [r["metrics"]["per_class"]["negative"]["f1"] for r in runs_for_key]
        pos = [r["metrics"]["per_class"]["positive"]["f1"] for r in runs_for_key]
        unp = [r["n_unparseable"] for r in runs_for_key]
        config_str = regime if prompt == "default" else f"{regime}, prompt={prompt}"
        rows.append(
            {
                "family": f"LLM ({regime})",
                "model": model_short,
                "config": config_str,
                "n_seeds": len(runs_for_key),
                "accuracy": fmt_mean_std(accs),
                "macro_f1": fmt_mean_std(f1s),
                "neg_f1": fmt_mean_std(neg),
                "pos_f1": fmt_mean_std(pos),
                "n_unparseable": (
                    f"{sum(unp) / len(unp):.0f}" if len(unp) > 1 else f"{unp[0]}"
                ),
            }
        )


def main() -> None:
    rows: list[dict] = []
    collect_classical(rows)
    collect_nbbert(rows)
    collect_chunked(rows)
    collect_llm(rows)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "summary_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_md = RESULTS_DIR / "summary_table.md"
    out_md.write_text(
        "# NoReC binary sentiment - test-set results (mean ± std across seeds)\n\n"
        + df_to_markdown(df) + "\n"
    )
    print(f"Wrote {out_md}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
