"""Prompting helpers for instruction-tuned LLMs on NoReC binary sentiment."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .data import LABEL_NAMES

SYSTEM_PROMPTS: dict[str, str] = {
    "default": (
        "You are a sentiment classifier for Norwegian reviews. "
        "Read the review and respond with exactly one word: "
        "either 'positive' or 'negative'. Do not explain."
    ),
    "terse": (
        "Classify the sentiment of the following Norwegian review. "
        "Reply with one word: positive or negative."
    ),
    "norwegian": (
        "Du er en sentimentklassifiserer for norske anmeldelser. "
        "Les anmeldelsen og svar med nøyaktig ett ord: "
        "enten 'positive' eller 'negative'. Ikke forklar."
    ),
}


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
    review: str,
    few_shot: list[FewShotExample] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build a chat-style message list for an instruction-tuned model.

    `system_prompt` defaults to SYSTEM_PROMPTS["default"]; pass an alternative
    string to swap in a different prompt without changing call sites.
    """
    sp = system_prompt if system_prompt is not None else SYSTEM_PROMPTS["default"]
    messages: list[dict[str, str]] = [{"role": "system", "content": sp}]
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
    Different `seed` values yield different demonstration sets, which is the
    primary source of variance for multi-seed few-shot evaluation.
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


def shuffle_few_shot(
    examples: list[FewShotExample], shuffle_seed: int
) -> list[FewShotExample]:
    """Return a permuted copy of `examples` using a deterministic seed.

    Used by the demonstration-order ablation: the same four demonstrations are
    presented to the model in different orders to test whether the prompted-
    classifier result is robust to ordering, independent of demo content.
    """
    import random

    rng = random.Random(shuffle_seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled
