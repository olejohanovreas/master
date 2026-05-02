"""Per-source and per-year breakdowns of test-set performance.

Joins the test-set predictions of the four headline configurations against
NoReC metadata and reports macro-F1 per (source) and per (year) group, with
support counts so small bins can be flagged.

Outputs:
    results/per_source.csv
    results/per_year.csv
    results/figures/per_source_heatmap.png
    results/figures/per_year.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from thesis.data import LABEL_NAMES, load_norec  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

HEADLINE = [
    ("classical", "logreg+balanced", "classical_preds__logreg__balanced=True.csv"),
    ("transformer", "NB-BERT-base", "nbbert_preds_seed42.csv"),
    (
        "LLM 0-shot",
        "Llama-3.1-8B 0-shot",
        "llm_preds__Llama-3.1-8B-Instruct__zero-shot__s42_pdefault.csv",
    ),
    (
        "LLM 4-shot",
        "Llama-3.1-8B 4-shot",
        "llm_preds__Llama-3.1-8B-Instruct__few-shot__s42_pdefault.csv",
    ),
]


def load_preds(csv_name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / csv_name, dtype={"id": str})
    df["id"] = df["id"].str.zfill(6)
    return df


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_all = load_norec()
    df_test = df_all[df_all["split"] == "test"][
        ["id", "source", "year", "category"]
    ].reset_index(drop=True)
    df_test["id"] = df_test["id"].astype(str).str.zfill(6)

    # ----- per-source -----
    src_rows: list[dict] = []
    for _family, label, csv_name in HEADLINE:
        df = load_preds(csv_name).merge(df_test, on="id", how="left")
        for src, sub in df.groupby("source", observed=True):
            metrics = compute_metrics(
                sub["label"].to_numpy(), sub["pred"].to_numpy(), LABEL_NAMES
            )
            src_rows.append(
                {
                    "model": label, "source": src, "n": len(sub),
                    "accuracy": round(metrics["accuracy"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                }
            )
    src_df = pd.DataFrame(src_rows)
    src_df.to_csv(RESULTS_DIR / "per_source.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'per_source.csv'}")

    pivot = src_df.pivot(index="source", columns="model", values="macro_f1")
    pivot = pivot.reindex(
        src_df.groupby("source", observed=True)["n"].first().sort_values(
            ascending=False
        ).index
    )
    pivot = pivot[[label for _, label, _ in HEADLINE]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    supports = src_df.groupby("source", observed=True)["n"].first()
    ax.set_yticklabels([f"{s} (n={supports[s]})" for s in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(
                j, i, "—" if pd.isna(v) else f"{v:.2f}",
                ha="center", va="center", fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="macro-F1")
    ax.set_title("Per-source macro-F1 (test)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_source_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'per_source_heatmap.png'}")

    # ----- per-year -----
    year_rows: list[dict] = []
    for _family, label, csv_name in HEADLINE:
        df = load_preds(csv_name).merge(df_test, on="id", how="left")
        for yr, sub in df.groupby("year", observed=True):
            if len(sub) < 30:
                continue
            metrics = compute_metrics(
                sub["label"].to_numpy(), sub["pred"].to_numpy(), LABEL_NAMES
            )
            year_rows.append(
                {
                    "model": label, "year": int(yr), "n": len(sub),
                    "accuracy": round(metrics["accuracy"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                }
            )
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(RESULTS_DIR / "per_year.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'per_year.csv'}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _family, label, _ in HEADLINE:
        sub = year_df[year_df["model"] == label].sort_values("year")
        ax.plot(sub["year"], sub["macro_f1"], marker="o", label=label)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("macro-F1")
    ax.set_title("Per-year macro-F1 on NoReC test (years with n >= 30)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_year.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'per_year.png'}")


if __name__ == "__main__":
    main()
