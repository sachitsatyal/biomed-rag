import pytest

from biomed_rag.retrieval.base import SearchResult
from biomed_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def ranking(*chunk_ids: str) -> list[SearchResult]:
    return [SearchResult(cid, cid.split("-")[0], 0.0, f"text {cid}") for cid in chunk_ids]


class FakeRetriever:
    """Returns a fixed ranking and remembers the k it was asked for."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.asked_k = None

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        self.asked_k = k
        return self.results[:k]


def test_chunk_found_by_both_retrievers_wins():
    fused = reciprocal_rank_fusion([ranking("a-0", "b-0", "c-0"), ranking("d-0", "b-0", "e-0")])
    assert fused[0].chunk_id == "b-0"


def test_score_is_sum_of_reciprocal_ranks():
    fused = reciprocal_rank_fusion([ranking("a-0", "b-0"), ranking("b-0")], rrf_k=60)
    scores = {r.chunk_id: r.score for r in fused}
    assert scores["b-0"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a-0"] == pytest.approx(1 / 61)


def test_k_limits_results():
    assert len(reciprocal_rank_fusion([ranking("a-0", "b-0", "c-0")], k=2)) == 2


def test_empty_ranking_falls_back_to_the_other():
    fused = reciprocal_rank_fusion([[], ranking("a-0", "b-0")])
    assert [r.chunk_id for r in fused] == ["a-0", "b-0"]


def test_hybrid_asks_each_retriever_for_candidates():
    bm25, dense = FakeRetriever(ranking("a-0", "b-0")), FakeRetriever(ranking("b-0", "c-0"))
    results = HybridRetriever([bm25, dense], candidates=50).search("query", k=3)
    assert bm25.asked_k == dense.asked_k == 50
    assert results[0].chunk_id == "b-0"
    assert results[0].pmid == "b"
