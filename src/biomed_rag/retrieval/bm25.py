"""BM25 keyword retrieval over chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from biomed_rag.retrieval.tokenizer import tokenize


@dataclass
class SearchResult:
    """One retrieved chunk with its relevance score."""

    chunk_id: str
    pmid: str
    score: float
    text: str


class BM25Retriever:
    """Keyword search: index chunks once, then rank them for any query."""

    def __init__(self, chunk_ids: list[str], pmids: list[str], texts: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.pmids = pmids
        self.texts = texts
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])

    @classmethod
    def from_parquet(cls, path: str | Path) -> BM25Retriever:
        """Build the index from a chunks Parquet file."""
        df = pd.read_parquet(path)
        return cls(df["chunk_id"].tolist(), df["pmid"].tolist(), df["text"].tolist())

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return the top-k chunks for the query, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(self.chunk_ids[i], self.pmids[i], float(scores[i]), self.texts[i])
            for i in top
            if scores[i] > 0
        ]
