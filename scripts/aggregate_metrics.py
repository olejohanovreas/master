"""Build a single test-set metrics table from all experiment outputs.

Reads results/{classical.json, nbbert.json, llm.json} and writes a unified
CSV + Markdown summary that can be cited directly in the thesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    rows: list[dict] = []

    with (RESULTS_DIR / "classical.json").open() as f:
        classical = json.load(f)
    for r in classical:
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
                "family": family,
                "model": model,
                "config": config,
                "accuracy": round(m["accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "neg_f1": round(m["per_class"]["negative"]["f1"], 4),
                "pos_f1": round(m["per_class"]["positive"]["f1"], 4),
                "n_unparseable": "",
            }
        )

    with (RESULTS_DIR / "nbbert.json").open() as f:
        nb = json.load(f)
    m = nb["test"]
    rows.append(
        {
            "family": "transformer",
            "model": "NB-BERT-base",
            "config": (
                f"3 ep, lr={nb['config']['lr']}, batch={nb['config']['batch_size']}, "
                f"max_len={nb['config']['max_length']}"
            ),
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "neg_f1": round(m["per_class"]["negative"]["f1"], 4),
            "pos_f1": round(m["per_class"]["positive"]["f1"], 4),
            "n_unparseable": "",
        }
    )

    with (RESULTS_DIR / "llm.json").open() as f:
        llm = json.load(f)
    for r in llm["runs"]:
        m = r["metrics"]
        short = r["model"].split("/")[-1]
        rows.append(
            {
                "family": f"LLM ({r['regime']})",
                "model": short,
                "config": r["regime"],
                "accuracy": round(m["accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "neg_f1": round(m["per_class"]["negative"]["f1"], 4),
                "pos_f1": round(m["per_class"]["positive"]["f1"], 4),
                "n_unparseable": r.get("n_unparseable", 0),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["family", "macro_f1"], ascending=[True, False]).reset_index(
        drop=True
    )

    out_csv = RESULTS_DIR / "summary_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_md = RESULTS_DIR / "summary_table.md"
    out_md.write_text(
        "# NoReC binary sentiment — test-set results\n\n" + df_to_markdown(df) + "\n"
    )
    print(f"Wrote {out_md}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
