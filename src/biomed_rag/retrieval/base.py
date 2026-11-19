"""Types shared by all retrievers (BM25, dense, hybrid)."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """One retrieved chunk with its relevance score."""

    chunk_id: str
    pmid: str
    score: float
    text: str
