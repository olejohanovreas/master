"""TF-IDF + linear classifier pipelines for the classical baseline."""

from __future__ import annotations

from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ClassifierName = Literal["logreg", "svm"]


def make_vectorizer() -> TfidfVectorizer:
    """Default TF-IDF for Norwegian: word unigrams + bigrams, sublinear TF.

    `strip_accents` is left at None to preserve æ/ø/å.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
    )


def make_classifier(name: ClassifierName, C: float, balanced: bool):
    cw = "balanced" if balanced else None
    if name == "logreg":
        return LogisticRegression(C=C, max_iter=1000, class_weight=cw)
    if name == "svm":
        return LinearSVC(C=C, class_weight=cw, max_iter=2000)
    raise ValueError(f"unknown classifier {name!r}")


def make_pipeline(name: ClassifierName, C: float, balanced: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", make_vectorizer()),
            ("clf", make_classifier(name, C, balanced)),
        ]
    )
