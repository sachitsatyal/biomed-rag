"""Types shared by all retrievers (BM25, dense, hybrid)."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    """One retrieved chunk with its relevance score."""

    chunk_id: str
    pmid: str
    score: float
    text: str


class Retriever(Protocol):
    """Anything with this search method counts as a retriever (BM25, dense, hybrid)."""

    def search(self, query: str, k: int = 10) -> list[SearchResult]: ...
