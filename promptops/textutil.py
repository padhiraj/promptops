"""Small text helpers shared by retrieval, metrics and the mock LLM."""
import re

STOP = set(
    """a an the is are was were be been being of to in on at for from by with and or if as it its this that these
    those what which who whom how when where why do does did can could should would may might must will shall i you we
    they he she them our your their my me us there here than then so any all each per about into over under up down out
    off just also very more most such own same too s t get give tell much many long often big""".split()
)

_CITATION = re.compile(r"\[[^\]]*#\d+\]")


def stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > 4 and tok.endswith(suf):
            tok = tok[: -len(suf)]
            break
    if len(tok) > 3 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


def content_tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(t) for t in toks if t not in STOP and (len(t) > 2 or t.isdigit())]


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text)


def split_sentences(text: str) -> list[str]:
    text = strip_citations(text)
    parts = re.split(r"\n+|(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = re.sub(r"^[\s\-\*\u2022\d\.\)]+", "", p).strip()
        if p:
            out.append(p)
    return out
