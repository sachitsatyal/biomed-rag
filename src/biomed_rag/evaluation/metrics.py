"""Retrieval metrics, measured at the paper (PMID) level."""

from biomed_rag.retrieval.base import SearchResult


def ranked_pmids(results: list[SearchResult]) -> list[str]:
    """Papers in ranked order, each paper once (several chunks can share a PMID)."""
    seen: dict[str, None] = {}
    for result in results:
        seen.setdefault(result.pmid, None)
    return list(seen)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant papers that appear in the top k."""
    if not relevant:
        raise ValueError("a question needs at least one relevant paper")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant paper; 0 if none was found."""
    for position, pmid in enumerate(ranked, start=1):
        if pmid in relevant:
            return 1.0 / position
    return 0.0


def judged_at_k(ranked: list[str], judged: set[str], k: int) -> float:
    """Share of the top k that has a relevance label. Low = the score is unreliable."""
    top = ranked[:k]
    return sum(pmid in judged for pmid in top) / len(top) if top else 0.0
