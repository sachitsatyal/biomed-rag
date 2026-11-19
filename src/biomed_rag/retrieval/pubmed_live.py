"""Live PubMed search as a retriever: for questions the local corpus cannot answer."""

from __future__ import annotations

from typing import Any

from biomed_rag.retrieval.base import SearchResult


class PubMedLiveRetriever:
    """Search PubMed right now and return abstracts in the same shape as the other retrievers."""

    def __init__(self, client: Any) -> None:
        self.client = client  # a PubMedClient (search + fetch)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        pmids = self.client.search(f"({query}) AND hasabstract", k)
        if not pmids:
            return []
        articles = {a.pmid: a for a in self.client.fetch(pmids)}
        results = []
        for rank, pmid in enumerate(pmids, start=1):
            article = articles.get(pmid)
            if article is not None:
                text = f"{article.title}\n\n{article.abstract}"
                results.append(SearchResult(f"{pmid}-live", pmid, 1.0 / rank, text))
        return results
