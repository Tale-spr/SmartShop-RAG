import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = (text or "").lower()
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9\.\-]+|[一-鿿]+", normalized):
        chunk = match.group(0)
        if re.fullmatch(r"[a-z0-9\.\-]+", chunk):
            tokens.append(chunk)
            continue
        chars = [char for char in chunk if char.strip()]
        tokens.extend(chars)
        tokens.extend("".join(chars[index:index + 2]) for index in range(len(chars) - 1))
    return tokens


@dataclass
class BM25Match:
    document: Document
    score: float
    rank: int


class BM25Index:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_terms = [tokenize_for_bm25(doc.page_content) for doc in self.documents]
        self.doc_term_freqs = [Counter(terms) for terms in self.doc_terms]
        self.doc_lengths = [len(terms) for terms in self.doc_terms]
        self.avgdl = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.doc_freqs = Counter()
        for term_freq in self.doc_term_freqs:
            for term in term_freq:
                self.doc_freqs[term] += 1

    def search(self, query: str, *, top_k: int = 5) -> list[BM25Match]:
        query_terms = tokenize_for_bm25(query)
        if not query_terms or not self.documents:
            return []

        scored: list[tuple[float, int]] = []
        total_docs = len(self.documents)
        for index, term_freq in enumerate(self.doc_term_freqs):
            score = 0.0
            doc_length = self.doc_lengths[index] or 1
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq <= 0:
                    continue
                doc_freq = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / (self.avgdl or 1))
                score += idf * numerator / denominator
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [BM25Match(document=self.documents[index], score=score, rank=rank) for rank, (score, index) in enumerate(scored[:top_k], start=1)]
