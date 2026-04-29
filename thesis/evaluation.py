"""Classification metrics shared across all model families."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, label_names: Sequence[str]
) -> dict:
    """Return a JSON-serializable dict of standard classification metrics.

    Includes accuracy, macro-F1 (primary), per-class precision/recall/F1/support,
    and the raw confusion matrix.
    """
    n_classes = len(label_names)
    labels = list(range(n_classes))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    p = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    sup = np.bincount(y_true, minlength=n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            label_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(sup[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
    }
