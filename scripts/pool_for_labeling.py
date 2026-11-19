"""Pool the top papers from every retriever into a CSV for relevance labeling.

Usage:  uv run python scripts/pool_for_labeling.py
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="golden/questions_v1.jsonl")
    parser.add_argument("--out", default="golden/labels_v1.csv")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--chunks", default="data/processed/chunks_abstract.parquet")
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    args = parser.parse_args()

    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite your labels.")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.evaluation.metrics import ranked_pmids
    from biomed_rag.retrieval.bm25 import BM25Retriever
    from biomed_rag.retrieval.dense import DenseRetriever
    from biomed_rag.retrieval.hybrid import HybridRetriever

    bm25 = BM25Retriever.from_parquet(args.chunks)
    dense = DenseRetriever.load(args.index)
    systems = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}
    texts = dict(zip(bm25.pmids, bm25.texts, strict=True))

    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    rows = []
    for q in questions:
        found_by: dict[str, list[str]] = {}
        for name, retriever in systems.items():
            for pmid in ranked_pmids(retriever.search(q["question"], k=50))[: args.depth]:
                found_by.setdefault(pmid, []).append(name)
        for pmid, names in found_by.items():
            title, _, abstract = texts[pmid].partition("\n\n")
            rows.append(
                {
                    "qid": q["qid"],
                    "question": q["question"],
                    "pmid": pmid,
                    "title": title,
                    "abstract_start": abstract[:300],
                    "found_by": "+".join(names),
                    "relevant": "",
                }
            )

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} rows for {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
