"""
Policy retrieval with BM25 over Markdown policy files.

Each "## <ID> <title>" section in data/policies/*.md becomes one chunk, so the
policy agent can cite exact section IDs. Pure Python and offline; swap in an
embedding index later (see README roadmap) and compare both in the evals.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from cc_copilot.rag.arabic_text import tokenize

_SECTION = re.compile(r"^##\s+(?P<id>[A-Z]+-\d+)\s+(?P<title>.+)$", re.MULTILINE)


@dataclass(frozen=True)
class PolicyChunk:
    policy_id: str
    title: str
    text: str
    source: str


@dataclass
class SearchHit:
    chunk: PolicyChunk
    score: float


def load_policy_chunks(policy_dir: Path) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for path in sorted(policy_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        sections = list(_SECTION.finditer(content))
        for i, m in enumerate(sections):
            end = sections[i + 1].start() if i + 1 < len(sections) else len(content)
            body = content[m.end() : end].strip()
            chunks.append(PolicyChunk(m["id"], m["title"].strip(), body, path.name))
    return chunks


@dataclass
class BM25Retriever:
    chunks: list[PolicyChunk]
    k1: float = 1.5
    b: float = 0.75
    _doc_tokens: list[list[str]] = field(init=False, repr=False)
    _idf: dict[str, float] = field(init=False, repr=False)
    _avg_len: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._doc_tokens = [tokenize(f"{c.title} {c.text}") for c in self.chunks]
        n = len(self._doc_tokens)
        df = Counter(t for doc in self._doc_tokens for t in set(doc))
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._avg_len = sum(map(len, self._doc_tokens)) / n if n else 0.0

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        query_tokens = tokenize(query)
        hits: list[SearchHit] = []
        for chunk, tokens in zip(self.chunks, self._doc_tokens):
            tf = Counter(tokens)
            score = 0.0
            for t in query_tokens:
                if t not in tf:
                    continue
                freq = tf[t]
                norm = 1 - self.b + self.b * len(tokens) / self._avg_len
                score += self._idf[t] * freq * (self.k1 + 1) / (freq + self.k1 * norm)
            if score > 0:
                hits.append(SearchHit(chunk, score))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]
