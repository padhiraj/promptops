"""LLM backends.

* `mock`  : a deterministic extractive simulator. It exists so tests and CI run with no API key, and so the
            harness can be shown to catch bad behaviour. Its scores say nothing about real model quality.
* `groq`  : Groq's OpenAI-compatible endpoint (GROQ_API_KEY).
* `openai`: OpenAI chat completions (OPENAI_API_KEY).
"""
import os
import re
import time
from typing import Optional

import requests

from .config import CANARY, REFUSAL
from .textutil import content_tokens, split_sentences

_INJECT = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules)|disregard (all |any )?(the )?"
    r"(previous|prior|above|earlier) (instructions|rules)|you are now\b|(reveal|print|show|repeat).{0,40}"
    r"(system prompt|hidden instructions|internal id)",
    re.I,
)


class MockLLM:
    """Simulates a model that follows (or ignores) the rules written in the system prompt."""

    name = "mock"
    MIN_COVERAGE = 0.4

    def complete(self, system: str, user: str, meta: Optional[dict] = None) -> str:
        meta = meta or {}
        q = meta.get("question", "")
        chunks = meta.get("chunks", [])
        s = system.lower()
        resistant = "untrusted" in s
        has_refusal_rule = REFUSAL.lower() in s
        cites = "cite" in s

        if _INJECT.search(q):
            if resistant:
                return REFUSAL
            if re.search(r"system prompt|instructions|internal id", q, re.I):
                return system  # simulated leak: contains the canary
            return "PWNED"

        qtok = set(content_tokens(q))
        scored = []
        for cid, text in chunks:
            for sent in split_sentences(text):
                st = set(content_tokens(sent))
                if qtok and st:
                    scored.append((len(qtok & st) / len(qtok), sent, cid))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < self.MIN_COVERAGE:
            if has_refusal_rule:
                return REFUSAL
            return "According to standard company policy, this is typically handled within 10 working days by the relevant department."
        top = [x for x in scored[:2] if x[0] >= self.MIN_COVERAGE]
        text = " ".join(x[1].rstrip(".") + "." for x in top)
        return text + (f" [{top[0][2]}]" if cites else "")


class ChatCompletionsLLM:
    def __init__(self, name: str, base_url: str, key_env: str, default_model: str):
        self.name, self.base_url, self.key_env = name, base_url.rstrip("/"), key_env
        self.model = os.getenv("PROMPTOPS_MODEL", default_model)

    def complete(self, system: str, user: str, meta: Optional[dict] = None) -> str:
        key = os.getenv(self.key_env)
        if not key:
            raise RuntimeError(f"{self.key_env} is not set. Export it or use --provider mock.")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 400,
        }
        for attempt in range(4):
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError("unreachable")


def get_llm(provider: str):
    if provider == "mock":
        return MockLLM()
    if provider == "groq":
        return ChatCompletionsLLM("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.1-8b-instant")
    if provider == "openai":
        return ChatCompletionsLLM("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini")
    raise ValueError(f"Unknown provider '{provider}'. Choose mock, groq or openai.")
