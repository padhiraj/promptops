"""Lightweight input and output guardrails (pattern based, deliberately simple and inspectable)."""
import re
from typing import Optional

from .config import CANARY, REFUSAL

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"(reveal|print|show|repeat|output|display).{0,40}(system prompt|hidden instructions|internal id|your instructions)",
    r"you are now\b",
    r"\bDAN\b",
    r"jailbreak",
]
PII_PATTERNS = [
    r"(phone number|home address|personal (phone|email|number)|email address of).{0,60}(customer|employee|employees|staff)",
    r"(customer|employee|employees|staff).{0,60}(phone number|home address|personal (phone|email|number))",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_PII = [re.compile(p, re.I) for p in PII_PATTERNS]


def check_input(question: str) -> Optional[str]:
    """Return a reason string if the question should be blocked, else None."""
    for rx in _INJ:
        if rx.search(question):
            return "prompt_injection"
    for rx in _PII:
        if rx.search(question):
            return "personal_data"
    return None


def check_output(answer: str) -> tuple[str, list[str]]:
    """Redact answers that leak the internal canary."""
    if CANARY.lower() in answer.lower():
        return REFUSAL, ["output_blocked:canary_leak"]
    return answer, []
