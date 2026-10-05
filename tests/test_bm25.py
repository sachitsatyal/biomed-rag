from biomed_rag.retrieval.bm25 import BM25Retriever


def make_retriever() -> BM25Retriever:
    return BM25Retriever(
        chunk_ids=["1-0", "2-0", "3-0"],
        pmids=["1", "2", "3"],
        texts=[
            "TREM2 variants alter microglia function.",
            "Lecanemab reduced amyloid in a clinical trial.",
            "Sleep quality and cognitive decline in older adults.",
        ],
    )


def test_top_result_matches_rare_term():
    assert make_retriever().search("TREM2", k=3)[0].pmid == "1"


def test_results_sorted_by_score():
    results = make_retriever().search("amyloid clinical trial microglia", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_no_matching_words_returns_nothing():
    assert make_retriever().search("zebrafish", k=3) == []


def test_k_limits_number_of_results():
    assert len(make_retriever().search("amyloid microglia sleep", k=2)) == 2
