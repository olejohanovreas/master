"""Prompting helpers for instruction-tuned LLMs on NoReC binary sentiment."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .data import LABEL_NAMES

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Norwegian reviews. "
    "Read the review and respond with exactly one word: "
    "either 'positive' or 'negative'. Do not explain."
)


def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Trim text so that it tokenizes to at most max_tokens.

    Used to bound review length before building chat messages, so the chat
    template's assistant-turn marker is never lost to right-side truncation.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


@dataclass(frozen=True)
class FewShotExample:
    review_id: str
    text: str
    label_name: str  # "positive" or "negative"


def build_messages(
    review: str, few_shot: list[FewShotExample] | None = None
) -> list[dict[str, str]]:
    """Build a chat-style message list for an instruction-tuned model."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for ex in few_shot:
            messages.append({"role": "user", "content": f"Review:\n{ex.text}"})
            messages.append({"role": "assistant", "content": ex.label_name})
    messages.append({"role": "user", "content": f"Review:\n{review}"})
    return messages


_FIRST_WORD_RE = re.compile(r"[A-Za-zÆØÅæøå]+")


def parse_response(text: str) -> int | None:
    """Map a model response to {0=negative, 1=positive}. Returns None if unparseable.

    Looks at the first alphabetic token and matches case-insensitively against
    English ('positive'/'negative') and Norwegian ('positiv(t)'/'negativ(t)').
    """
    if not text:
        return None
    m = _FIRST_WORD_RE.search(text)
    if not m:
        return None
    word = m.group(0).lower()
    if word.startswith("pos"):
        return 1
    if word.startswith("neg"):
        return 0
    return None


def select_few_shot_examples(
    train_df: pd.DataFrame,
    n_per_class: int = 2,
    seed: int = 42,
    min_words: int = 50,
    max_words: int = 200,
) -> list[FewShotExample]:
    """Sample few-shot demonstrations from train, balanced across classes.

    Restricts to mid-length reviews so the prompt stays compact and easy to read.
    Returns examples interleaved (neg, pos, neg, pos, ...) to avoid ordering bias.
    """
    word_counts = train_df["text"].str.split().str.len()
    pool = train_df[(word_counts >= min_words) & (word_counts <= max_words)]

    by_class: list[list[FewShotExample]] = []
    for label in (0, 1):
        sub = pool[pool["label"] == label].sample(
            n_per_class, random_state=seed + label
        )
        by_class.append(
            [
                FewShotExample(row["id"], row["text"], LABEL_NAMES[label])
                for _, row in sub.iterrows()
            ]
        )
    interleaved: list[FewShotExample] = []
    for i in range(n_per_class):
        for cls in by_class:
            interleaved.append(cls[i])
    return interleaved
