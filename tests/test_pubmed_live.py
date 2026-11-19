from biomed_rag.data.pubmed import Article
from biomed_rag.retrieval.pubmed_live import PubMedLiveRetriever


class FakeClient:
    def __init__(self) -> None:
        self.query = None

    def search(self, query, max_results):
        self.query = query
        return ["2", "1"][:max_results]

    def fetch(self, pmids):
        return [Article(p, f"Title {p}", f"Abstract {p}", "J", 2025, None, None) for p in pmids]


def test_live_results_keep_pubmed_order_and_carry_pmids():
    client = FakeClient()
    results = PubMedLiveRetriever(client).search("p-tau217 accuracy", k=2)
    assert [r.pmid for r in results] == ["2", "1"]
    assert results[0].text == "Title 2\n\nAbstract 2"
    assert results[0].score > results[1].score
    assert "hasabstract" in client.query
