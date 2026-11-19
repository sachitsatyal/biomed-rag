"""Reranking: a cross-encoder rereads (question, chunk) pairs and reorders the candidates."""

from __future__ import annotations

from typing import Protocol

from sentence_transformers import CrossEncoder

from biomed_rag.retrieval.base import Retriever, SearchResult

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PairScorer(Protocol):
    """Anything that scores (query, text) pairs, like a CrossEncoder."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class RerankingRetriever:
    """Fetch candidates with a fast retriever, then reorder them with a slower, smarter model."""

    def __init__(
        self,
        retriever: Retriever,
        model_name: str = DEFAULT_RERANKER,
        candidates: int = 30,
        model: PairScorer | None = None,
    ) -> None:
        self.retriever = retriever
        self.model_name = model_name
        self.candidates = candidates
        self.model = model if model is not None else CrossEncoder(model_name)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        candidates = self.retriever.search(query, k=self.candidates)
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            SearchResult(
                candidates[i].chunk_id, candidates[i].pmid, float(scores[i]), candidates[i].text
            )
            for i in order
        ]
