"""NoReC loader.

Loads the document-level Norwegian Review Corpus (https://github.com/ltgoslo/norec)
from a local clone, applies the binary label mapping
{1,2,3} -> negative (0), {4,5,6} -> positive (1), and exposes train/dev/test splits
as pandas DataFrames.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOREC_ROOT = REPO_ROOT / "data" / "norec" / "data"

RATING_TO_LABEL = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
LABEL_NAMES = ["negative", "positive"]
SPLITS = ("train", "dev", "test")


def load_norec(root: Path | str | None = None) -> pd.DataFrame:
    """Load every NoReC review into a single DataFrame.

    Columns: id, split, rating, label, category, language, source, year, title, text.
    """
    root = Path(root) if root else DEFAULT_NOREC_ROOT
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected NoReC metadata at {metadata_path}. "
            "Did you `git clone https://github.com/ltgoslo/norec.git data/norec`?"
        )

    with metadata_path.open(encoding="utf-8") as f:
        metadata: dict[str, dict] = json.load(f)

    rows = []
    for rid, m in metadata.items():
        split = m["split"]
        text_path = root / split / f"{rid}.txt"
        rows.append(
            {
                "id": rid,
                "split": split,
                "rating": m["rating"],
                "label": RATING_TO_LABEL[m["rating"]],
                "category": m.get("category"),
                "language": m.get("language"),
                "source": m.get("source"),
                "year": m.get("year"),
                "title": m.get("title"),
                "text": text_path.read_text(encoding="utf-8"),
            }
        )
    return pd.DataFrame(rows)


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return df.loc[df["split"] == split].reset_index(drop=True)
