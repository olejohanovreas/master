"""Error analysis across model families.

Produces confusion matrices, per-category and per-length breakdowns, and a
small selection of qualitative misclassifications, all from the saved
prediction CSVs joined against NoReC metadata.

Outputs (under results/):
    figures/confusion_matrices.png
    figures/per_category_heatmap.png
    figures/per_length.png
    per_category.csv
    per_length.csv
    error_samples.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from thesis.data import LABEL_NAMES, load_norec  # noqa: E402
from thesis.evaluation import compute_metrics  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# (family, label, preds-csv-filename)
HEADLINE = [
    ("classical", "logreg+balanced", "classical_preds__logreg__balanced=True.csv"),
    ("transformer", "NB-BERT-base", "nbbert_preds.csv"),
    (
        "LLM 0-shot",
        "Llama-3.1-8B 0-shot",
        "llm_preds__Llama-3.1-8B-Instruct__zero-shot.csv",
    ),
    (
        "LLM 4-shot",
        "Llama-3.1-8B 4-shot",
        "llm_preds__Llama-3.1-8B-Instruct__few-shot.csv",
    ),
]


def load_preds(csv_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / csv_name
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = df["id"].str.zfill(6)
    return df


def plot_confusion(y_true, y_pred, title: str, ax) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    ax.set_title(title, fontsize=11)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NoReC metadata...")
    df_all = load_norec()
    df_test = df_all[df_all["split"] == "test"][
        ["id", "category", "language", "year", "text"]
    ].reset_index(drop=True)
    df_test["id"] = df_test["id"].astype(str).str.zfill(6)
    df_test["n_words"] = df_test["text"].str.split().str.len()

    # ---------- Confusion matrices ----------
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(4 * len(HEADLINE), 4))
    for ax, (_family, label, csv_name) in zip(axes, HEADLINE):
        df = load_preds(csv_name)
        plot_confusion(df["label"].to_numpy(), df["pred"].to_numpy(), label, ax)
    fig.suptitle("Confusion matrices on NoReC test (counts and row-normalized rates)")
    fig.tight_layout()
    out = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------- Per-category ----------
    cat_rows: list[dict] = []
    for _family, label, csv_name in HEADLINE:
        df = load_preds(csv_name).merge(df_test, on="id", how="left")
        for cat, sub in df.groupby("category", observed=True):
            metrics = compute_metrics(
                sub["label"].to_numpy(), sub["pred"].to_numpy(), LABEL_NAMES
            )
            cat_rows.append(
                {
                    "model": label,
                    "category": cat,
                    "n": len(sub),
                    "accuracy": round(metrics["accuracy"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                }
            )
    cat_df = pd.DataFrame(cat_rows)
    out_csv = RESULTS_DIR / "per_category.csv"
    cat_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    pivot = cat_df.pivot(index="category", columns="model", values="macro_f1")
    pivot = pivot.reindex(
        cat_df.groupby("category", observed=True)["n"].first().sort_values(
            ascending=False
        ).index
    )
    pivot = pivot[[label for _, label, _ in HEADLINE]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(
                j,
                i,
                "—" if pd.isna(v) else f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="macro-F1")
    ax.set_title("Per-category macro-F1 (test)")
    fig.tight_layout()
    out = FIGURES_DIR / "per_category_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------- Per-length ----------
    bins = [0, 100, 200, 300, 500, 1000, 10_000]
    bin_labels = ["0-100", "100-200", "200-300", "300-500", "500-1000", "1000+"]
    df_test["length_bin"] = pd.cut(
        df_test["n_words"], bins=bins, labels=bin_labels, right=False
    )

    len_rows: list[dict] = []
    for _family, label, csv_name in HEADLINE:
        df = load_preds(csv_name).merge(df_test, on="id", how="left")
        for bin_name, sub in df.groupby("length_bin", observed=True):
            if len(sub) == 0:
                continue
            metrics = compute_metrics(
                sub["label"].to_numpy(), sub["pred"].to_numpy(), LABEL_NAMES
            )
            len_rows.append(
                {
                    "model": label,
                    "length_bin": str(bin_name),
                    "n": len(sub),
                    "accuracy": round(metrics["accuracy"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                }
            )
    len_df = pd.DataFrame(len_rows)
    out_csv = RESULTS_DIR / "per_length.csv"
    len_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for _family, label, _ in HEADLINE:
        sub = (
            len_df[len_df["model"] == label]
            .set_index("length_bin")
            .reindex(bin_labels)
        )
        ax.plot(sub.index, sub["macro_f1"], marker="o", label=label)
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("macro-F1")
    ax.set_title("Per-length macro-F1 on NoReC test")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "per_length.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # ---------- Qualitative samples ----------
    bert = load_preds("nbbert_preds.csv")
    llm = load_preds("llm_preds__Llama-3.1-8B-Instruct__few-shot.csv")
    bert.rename(columns={"pred": "bert_pred"}, inplace=True)
    llm.rename(columns={"pred": "llm_pred", "raw": "llm_raw"}, inplace=True)
    merged = (
        bert[["id", "label", "bert_pred"]]
        .merge(llm[["id", "llm_pred", "llm_raw"]], on="id")
        .merge(df_test[["id", "category", "n_words", "text"]], on="id")
    )
    merged["bert_correct"] = merged["bert_pred"] == merged["label"]
    merged["llm_correct"] = merged["llm_pred"] == merged["label"]

    bert_only = merged[merged["bert_correct"] & ~merged["llm_correct"]]
    llm_only = merged[~merged["bert_correct"] & merged["llm_correct"]]
    both_wrong = merged[~merged["bert_correct"] & ~merged["llm_correct"]]

    def fmt_row(r) -> str:
        excerpt = r["text"][:400].replace("\n", " ").strip()
        if len(r["text"]) > 400:
            excerpt += "..."
        return (
            f"- **id={r['id']}** ({r['category']}, {r['n_words']} words) "
            f"true={LABEL_NAMES[r['label']]}, "
            f"BERT={LABEL_NAMES[r['bert_pred']]}, "
            f"LLM raw={r['llm_raw']!r}\n  > {excerpt}"
        )

    rng = np.random.default_rng(42)
    out_md = RESULTS_DIR / "error_samples.md"
    with out_md.open("w") as f:
        f.write("# Qualitative misclassification samples\n\n")
        f.write(f"NB-BERT correct & 8B 4-shot wrong: {len(bert_only)} / {len(merged)}\n")
        f.write(f"NB-BERT wrong & 8B 4-shot correct: {len(llm_only)} / {len(merged)}\n")
        f.write(f"Both wrong: {len(both_wrong)} / {len(merged)}\n\n")
        for header, sub in [
            ("## NB-BERT right, 8B 4-shot wrong", bert_only),
            ("## 8B 4-shot right, NB-BERT wrong", llm_only),
            ("## Both wrong", both_wrong),
        ]:
            f.write(header + "\n\n")
            picks = sub.iloc[rng.choice(len(sub), size=min(5, len(sub)), replace=False)]
            for _, r in picks.iterrows():
                f.write(fmt_row(r) + "\n\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
