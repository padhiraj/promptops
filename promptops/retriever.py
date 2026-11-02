"""Pure-Python BM25 retriever over markdown SOP documents (one chunk per `##` section)."""
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DOCS_DIR
from .textutil import content_tokens


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    text: str


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        title, section, body, n = path.stem, "", [], 0
        sections: list[tuple[str, list[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if body:
                    sections.append((section, body))
                section, body = line[3:].strip(), []
            elif line.strip():
                body.append(line.strip())
        if body:
            sections.append((section, body))
        for section, body in sections:
            n += 1
            chunks.append(Chunk(f"{path.stem}#{n}", path.stem, title, f"{section}\n" + "\n".join(body)))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(content_tokens(f"{c.title} {c.text}")) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = content_tokens(query)
        scored = []
        for i, d in enumerate(self.docs):
            score = 0.0
            for t in q:
                f = d.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                score += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def default_index() -> BM25Index:
    return BM25Index(load_chunks())
