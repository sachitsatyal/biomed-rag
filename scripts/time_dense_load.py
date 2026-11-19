"""Show where dense-retriever startup time goes: imports, loading, and queries.

Usage:  uv run python scripts/time_dense_load.py
"""

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    parser.add_argument("--query", default="anti-amyloid antibody trial results")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the cached model, never go online

    t0 = time.perf_counter()
    from biomed_rag.retrieval.dense import DenseRetriever  # imports torch: the heavy part

    t1 = time.perf_counter()
    retriever = DenseRetriever.load(args.index)
    t2 = time.perf_counter()
    retriever.search(args.query, k=5)
    t3 = time.perf_counter()
    retriever.search(args.query, k=5)
    t4 = time.perf_counter()

    print(f"import libraries : {t1 - t0:7.1f} s")
    print(f"load model+index : {t2 - t1:7.1f} s")
    print(f"first query      : {(t3 - t2) * 1000:7.0f} ms")
    print(f"next query       : {(t4 - t3) * 1000:7.0f} ms")


if __name__ == "__main__":
    main()
