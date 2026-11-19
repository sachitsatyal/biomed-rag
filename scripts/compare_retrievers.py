"""Compare BM25, dense and hybrid results for one query, side by side.

Usage:  uv run python scripts/compare_retrievers.py "TREM2 microglia"
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunks", default="data/processed/chunks_abstract.parquet")
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the cached model, never go online
    from biomed_rag.retrieval.bm25 import BM25Retriever
    from biomed_rag.retrieval.dense import DenseRetriever
    from biomed_rag.retrieval.hybrid import HybridRetriever

    bm25 = BM25Retriever.from_parquet(args.chunks)
    dense = DenseRetriever.load(args.index)
    hybrid = HybridRetriever([bm25, dense])

    for name, retriever in [("BM25", bm25), ("Dense", dense), ("Hybrid", hybrid)]:
        print(f"\n== {name} ==")
        for r in retriever.search(args.query, k=args.k):
            title = r.text.split("\n")[0][:70]
            print(f"{r.score:8.3f}  {r.pmid:>9}  {title}")


if __name__ == "__main__":
    main()
