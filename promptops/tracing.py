"""Tracing: every call is appended to a local JSONL file. Langfuse is used too when keys are configured.

The Langfuse path is best effort and wrapped in try/except so a tracing problem never breaks an eval run.
It needs `pip install langfuse` plus LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, and has not been exercised
without real keys.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

from .config import RESULTS_DIR


class Tracer:
    def __init__(self, path: Optional[Path] = None, use_langfuse: bool = True):
        self.path = Path(path) if path else RESULTS_DIR / "traces.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lf = None
        if use_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse  # type: ignore
                self.lf = Langfuse()
            except Exception as exc:  # pragma: no cover
                print(f"[tracing] Langfuse disabled: {exc}")

    def log(self, record: dict) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.lf is not None:  # pragma: no cover
            try:
                self.lf.trace(
                    name="promptops.ask",
                    input=record.get("question"),
                    output=record.get("answer"),
                    metadata={k: v for k, v in record.items() if k not in ("question", "answer")},
                )
            except Exception as exc:
                print(f"[tracing] Langfuse log failed: {exc}")

    def flush(self) -> None:
        if self.lf is not None:  # pragma: no cover
            try:
                self.lf.flush()
            except Exception:
                pass
