from biomed_rag.retrieval.base import SearchResult
from biomed_rag.retrieval.rerank import RerankingRetriever


class FakeRetriever:
    """Returns chunks in a fixed (deliberately poor) order."""

    def __init__(self, texts: list[str]) -> None:
        self.results = [SearchResult(f"{i}-0", str(i), 0.0, t) for i, t in enumerate(texts)]
        self.asked_k = None

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        self.asked_k = k
        return self.results[:k]


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the text. No download."""

    def predict(self, pairs):
        return [sum(w in text.lower() for w in query.lower().split()) for query, text in pairs]


def make(texts):
    base = FakeRetriever(texts)
    return base, RerankingRetriever(base, candidates=30, model=FakeCrossEncoder())


def test_reranker_moves_best_match_to_top():
    _, rr = make(["sleep study", "trem2 microglia", "trem2 microglia variants"])
    assert [r.pmid for r in rr.search("trem2 microglia variants", k=3)] == ["2", "1", "0"]


def test_reranker_asks_base_for_candidates_and_returns_k():
    base, rr = make(["a", "b", "c", "d"])
    assert len(rr.search("a", k=2)) == 2
    assert base.asked_k == 30


def test_reranker_handles_no_candidates():
    _, rr = make([])
    assert rr.search("anything") == []
