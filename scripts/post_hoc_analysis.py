"""Post-hoc analyses on existing predictions.

Computes:
  1) McNemar's test for NB-BERT (seed=42) vs Llama-3.1-8B 4-shot (seed=42)
  2) Majority-vote ensemble of best classical + NB-BERT + Llama-3.1-8B 4-shot
  3) Diagnostic for Llama-3.2-1B zero-shot collapse: how much of the "97% negative"
     output is the literal token vs. unparseable fallback?

All inputs are existing prediction CSVs in results/. No re-runs required.

Usage: uv run python scripts/post_hoc_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_aligned(*paths):
    dfs = [pd.read_csv(p).sort_values("id").reset_index(drop=True) for p in paths]
    ref = dfs[0]["id"].to_numpy()
    for d in dfs[1:]:
        assert np.array_equal(d["id"].to_numpy(), ref), "id ordering mismatch"
    return dfs


def mcnemar(y_true, pred_a, pred_b):
    """Returns (b, c, statistic, p_two_tailed) for paired predictions."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(((correct_a) & (~correct_b)).sum())
    c = int(((~correct_a) & (correct_b)).sum())
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / n
    chi2_p = 1.0 - stats.chi2.cdf(chi2, df=1)
    exact_p = stats.binomtest(min(b, c), n=n, p=0.5).pvalue
    return b, c, chi2, exact_p


def majority_vote(preds):
    """preds: list of 1-D arrays of equal length, each in {0,1}.
    Returns array of element-wise majority."""
    stack = np.vstack(preds)
    return (stack.sum(axis=0) > (len(preds) / 2)).astype(int)


def main():
    out = {}

    # === McNemar ===
    cls, nb, llm = load_aligned(
        RESULTS / "classical_preds__logreg__balanced=True.csv",
        RESULTS / "nbbert_preds_seed42.csv",
        RESULTS / "llm_preds__Llama-3.1-8B-Instruct__few-shot__s42_pdefault.csv",
    )
    y = nb["label"].to_numpy()
    p_cls = cls["pred"].to_numpy()
    p_nb = nb["pred"].to_numpy()
    p_llm = llm["pred"].to_numpy()

    b, c, chi2, p = mcnemar(y, p_nb, p_llm)
    out["mcnemar_nbbert_vs_llm8b"] = {
        "n_test": int(len(y)),
        "nbbert_correct_only": b,
        "llm_correct_only": c,
        "both_correct": int(((p_nb == y) & (p_llm == y)).sum()),
        "both_wrong": int(((p_nb != y) & (p_llm != y)).sum()),
        "chi2_statistic": chi2,
        "exact_two_tailed_p": p,
        "note": "NB-BERT is seed=42; LLM is Llama-3.1-8B 4-shot seed=42 default prompt.",
    }

    # === Ensemble ===
    ensemble_pred = majority_vote([p_cls, p_nb, p_llm])
    ens_acc = float((ensemble_pred == y).mean())
    ens_macro_f1 = float(f1_score(y, ensemble_pred, average="macro"))
    ens_per_class_f1 = f1_score(y, ensemble_pred, average=None).tolist()
    indiv = {
        "classical_lr_balanced": {
            "acc": float((p_cls == y).mean()),
            "macro_f1": float(f1_score(y, p_cls, average="macro")),
        },
        "nbbert_seed42": {
            "acc": float((p_nb == y).mean()),
            "macro_f1": float(f1_score(y, p_nb, average="macro")),
        },
        "llm_8b_4shot_seed42_default": {
            "acc": float((p_llm == y).mean()),
            "macro_f1": float(f1_score(y, p_llm, average="macro")),
        },
    }
    out["ensemble"] = {
        "members": list(indiv.keys()),
        "accuracy": ens_acc,
        "macro_f1": ens_macro_f1,
        "per_class_f1_neg_pos": ens_per_class_f1,
        "individual_members_at_seed42": indiv,
    }

    # === 1B zero-shot diagnostic ===
    df1b = pd.read_csv(
        RESULTS / "llm_preds__Llama-3.2-1B-Instruct__zero-shot__s42_pdefault.csv"
    )
    df1b_neg = df1b[df1b["pred"] == 0]
    parseable_neg = int(df1b_neg["parsed"].sum())
    unparseable = (~df1b["parsed"]).sum()
    n_total = int(len(df1b))
    n_neg = int((df1b["pred"] == 0).sum())
    n_pos = int((df1b["pred"] == 1).sum())
    out["llama_1b_zero_shot_diagnostic"] = {
        "n_test": n_total,
        "n_predicted_negative": n_neg,
        "frac_predicted_negative": n_neg / n_total,
        "n_predicted_positive": n_pos,
        "n_unparseable_total": int(unparseable),
        "of_negative_preds_n_parseable": parseable_neg,
        "of_negative_preds_n_unparseable_fallback": n_neg - parseable_neg,
        "note": (
            "All unparseable responses default to majority class (positive=1) per "
            "thesis convention, so any 'negative' prediction is a parseable model "
            "output — not a fallback. The output column shows what the model "
            "actually produced."
        ),
    }

    # raw distribution of negative-predicted outputs (top k)
    raw_counts = df1b_neg["raw"].astype(str).value_counts().head(10).to_dict()
    out["llama_1b_zero_shot_diagnostic"]["top_negative_raw_outputs"] = raw_counts

    out_path = RESULTS / "post_hoc.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)

    print(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
