"""Hybrid retrieval: fuse several rankings with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from biomed_rag.retrieval.base import Retriever, SearchResult


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], k: int = 10, rrf_k: int = 60
) -> list[SearchResult]:
    """Combine ranked lists using only ranks: score = sum of 1 / (rrf_k + rank)."""
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(result.chunk_id, result)
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [
        SearchResult(cid, first_seen[cid].pmid, scores[cid], first_seen[cid].text) for cid in top
    ]


class HybridRetriever:
    """Ask every retriever for candidates, then fuse their rankings with RRF."""

    def __init__(self, retrievers: list[Retriever], candidates: int = 50, rrf_k: int = 60) -> None:
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        rankings = [r.search(query, k=self.candidates) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=k, rrf_k=self.rrf_k)
