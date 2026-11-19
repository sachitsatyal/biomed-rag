"""Embed chunks and save a FAISS index to disk.

Usage:  uv run python scripts/build_dense_index.py
"""

import argparse

from biomed_rag.retrieval.dense import DEFAULT_MODEL, DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/processed/chunks_abstract.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/indexes/dense_bge-small_abstract")
    args = parser.parse_args()

    retriever = DenseRetriever.from_parquet(args.chunks, model_name=args.model)
    retriever.save(args.out)
    print(f"Saved {retriever.index.ntotal} vectors (dim {retriever.index.d}) to {args.out}")


if __name__ == "__main__":
    main()
