"""Score BM25, dense and hybrid retrieval against the labeled golden set.

Usage:  uv run python scripts/evaluate_retrieval.py
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="golden/questions_v1.jsonl")
    parser.add_argument("--labels", default="golden/labels_v1.csv")
    parser.add_argument("--chunks", default="data/processed/chunks_abstract.parquet")
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    parser.add_argument("--out", default="reports/retrieval_eval_v1.json")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.evaluation.metrics import ranked_pmids, recall_at_k, reciprocal_rank
    from biomed_rag.retrieval.bm25 import BM25Retriever
    from biomed_rag.retrieval.dense import DenseRetriever
    from biomed_rag.retrieval.hybrid import HybridRetriever

    labels = pd.read_csv(args.labels, dtype=str, encoding="utf-8-sig")
    relevant = labels[labels["relevant"] == "1"].groupby("qid")["pmid"].apply(set).to_dict()
    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    questions = [q for q in questions if q["qid"] in relevant]

    bm25 = BM25Retriever.from_parquet(args.chunks)
    dense = DenseRetriever.load(args.index)
    systems = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}

    rows = []
    for name, retriever in systems.items():
        for q in questions:
            ranked = ranked_pmids(retriever.search(q["question"], k=50))
            rel = relevant[q["qid"]]
            rows.append(
                {
                    "system": name,
                    "qid": q["qid"],
                    "type": q["type"],
                    "recall@5": recall_at_k(ranked, rel, 5),
                    "recall@10": recall_at_k(ranked, rel, 10),
                    "mrr": reciprocal_rank(ranked, rel),
                }
            )

    df = pd.DataFrame(rows)
    metrics = ["recall@5", "recall@10", "mrr"]
    overall = df.groupby("system", sort=False)[metrics].mean()
    by_type = df.pivot_table(index="type", columns="system", values="recall@10", sort=False)

    print(f"\n{len(questions)} questions, pooled relevance labels\n")
    print(overall.round(3).to_string())
    print("\nRecall@10 by question type\n")
    print(by_type[list(systems)].round(3).to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_questions": len(questions),
        "labels": args.labels,
        "chunks": args.chunks,
        "dense_index": args.index,
        "overall": overall.round(4).to_dict(orient="index"),
        "recall@10_by_type": by_type.round(4).to_dict(orient="index"),
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
