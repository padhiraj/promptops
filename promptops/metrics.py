"""Deterministic, dependency-free evaluation metrics.

These are lexical heuristics. They are fast, free and reproducible, which makes them suitable for a CI
regression gate. For a final quality read-out, spot-check with an LLM judge or a human reviewer.
"""
import re

from .textutil import content_tokens, split_sentences

REFUSAL_PATTERNS = [
    "don't know", "do not know", "i cannot find", "i can't find", "not mentioned in", "no information",
    "not in the provided", "unable to answer", "can't help with that", "cannot help with that",
    "i can only answer questions about", "not covered in the provided",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u2019", "'").lower()).strip()


def is_refusal(text: str) -> bool:
    t = _norm(text)
    return any(p in t for p in REFUSAL_PATTERNS)


def key_fact_hit(answer: str, fact: str) -> bool:
    """`fact` may list alternatives separated by '|'."""
    a = _norm(answer)
    return any(alt.strip().lower() in a for alt in fact.split("|"))


def correctness(answer: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 1.0
    return sum(key_fact_hit(answer, f) for f in key_facts) / len(key_facts)


def faithfulness(answer: str, context: str, threshold: float = 0.7) -> float:
    """Share of answer sentences whose content words are mostly present in the retrieved context.

    A refusal makes no claims, so it is treated as fully faithful. An empty answer scores 0.
    """
    if not answer.strip():
        return 0.0
    if is_refusal(answer):
        return 1.0
    ctx = set(content_tokens(context))
    sents = [content_tokens(s) for s in split_sentences(answer)]
    sents = [s for s in sents if s]
    if not sents:
        return 0.0
    supported = sum((sum(t in ctx for t in s) / len(s)) >= threshold for s in sents)
    return supported / len(sents)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs, p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]
