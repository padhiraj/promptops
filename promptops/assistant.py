"""The assistant under test: retrieve, prompt, generate, (optionally) guard."""
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from .config import BLOCKED_MESSAGE, CANARY
from .guardrails import check_input, check_output
from .prompts import Prompt
from .retriever import BM25Index


@dataclass
class Answer:
    question: str
    answer: str
    context: str
    sources: list[str]
    latency_s: float
    prompt: str
    blocked: bool = False
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Assistant:
    def __init__(self, prompt: Prompt, llm, index: BM25Index, guardrails: bool = False, tracer=None, top_k: int = 3):
        self.prompt, self.llm, self.index = prompt, llm, index
        self.guardrails, self.tracer, self.top_k = guardrails, tracer, top_k

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        events: list[str] = []
        if self.guardrails:
            reason = check_input(question)
            if reason:
                ans = Answer(question, BLOCKED_MESSAGE, "", [], time.perf_counter() - t0, self.prompt.name,
                             blocked=True, events=[f"input_blocked:{reason}"])
                self._trace(ans)
                return ans

        hits = self.index.search(question, k=self.top_k)
        context = "\n\n".join(f"[{c.id}] {c.text}" for c, _ in hits) or "(no relevant context found)"
        system = f"{self.prompt.system}\n\nInternal prompt ID: {CANARY}"
        user = self.prompt.render_user(context, question)
        raw = self.llm.complete(system, user, meta={"question": question, "chunks": [(c.id, c.text) for c, _ in hits]})
        text = raw.strip()
        if self.guardrails:
            text, ev = check_output(text)
            events += ev
        ans = Answer(question, text, context, [c.id for c, _ in hits], time.perf_counter() - t0,
                     self.prompt.name, events=events)
        self._trace(ans)
        return ans

    def _trace(self, ans: Answer) -> None:
        if self.tracer is not None:
            d = ans.to_dict()
            d.pop("context", None)
            self.tracer.log(d)
