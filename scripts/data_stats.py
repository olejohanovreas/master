"""Print descriptive statistics about NoReC after binary label mapping.

Usage: uv run python scripts/data_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thesis.data import LABEL_NAMES, SPLITS, load_norec  # noqa: E402


def _word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    print("Loading NoReC...")
    df = load_norec()
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].map(_word_count)

    print(f"\nTotal reviews: {len(df):,}\n")

    print("Per-split counts:")
    print(df.groupby("split", observed=True).size().reindex(SPLITS).to_string())

    print("\nRating distribution per split (rows = split, cols = rating 1-6):")
    print(
        pd.crosstab(df["split"], df["rating"]).reindex(SPLITS).to_string()
    )

    print("\nBinary class balance per split (label 0=neg, 1=pos):")
    bal = pd.crosstab(df["split"], df["label"]).reindex(SPLITS)
    bal.columns = [LABEL_NAMES[c] for c in bal.columns]
    bal["pos_share"] = (bal["positive"] / bal.sum(axis=1)).round(3)
    print(bal.to_string())

    print("\nCategory distribution (top 10 across full corpus):")
    print(df["category"].value_counts().head(10).to_string())

    print("\nLanguage distribution:")
    print(df["language"].value_counts().to_string())

    print("\nText length (whitespace-split words) per split:")
    length_stats = (
        df.groupby("split", observed=True)["n_words"]
        .agg(["mean", "median", "min", "max", "std"])
        .reindex(SPLITS)
        .round(1)
    )
    print(length_stats.to_string())


if __name__ == "__main__":
    main()
