import pytest

from biomed_rag.evaluation.metrics import ranked_pmids, recall_at_k, reciprocal_rank
from biomed_rag.retrieval.base import SearchResult


def test_ranked_pmids_keeps_first_occurrence_only():
    results = [SearchResult(c, c.split("-")[0], 0.0, "") for c in ["1-0", "1-1", "2-0", "1-2"]]
    assert ranked_pmids(results) == ["1", "2"]


def test_recall_counts_relevant_papers_in_top_k():
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d", "z"}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d", "z"}, k=4) == pytest.approx(2 / 3)


def test_recall_needs_relevant_papers():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=10)


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == pytest.approx(1 / 2)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
