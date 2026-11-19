"""Score retrieval systems against the labeled golden set.

Usage:  uv run python scripts/evaluate_retrieval.py
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="golden/questions_v1.jsonl")
    parser.add_argument("--labels", default="golden/labels_v3.csv")
    parser.add_argument("--chunks", default="data/processed/chunks_abstract.parquet")
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    parser.add_argument("--reranker", default=None, help="cross-encoder model name")
    parser.add_argument("--out", default="reports/retrieval_eval_v3.json")
    parser.add_argument("--unjudged-out", default="golden/unjudged_v2.csv")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.evaluation.metrics import (
        judged_at_k,
        ranked_pmids,
        recall_at_k,
        reciprocal_rank,
    )
    from biomed_rag.retrieval.bm25 import BM25Retriever
    from biomed_rag.retrieval.dense import DenseRetriever
    from biomed_rag.retrieval.hybrid import HybridRetriever
    from biomed_rag.retrieval.rerank import DEFAULT_RERANKER, RerankingRetriever

    labels = pd.read_csv(args.labels, dtype=str, encoding="utf-8-sig")
    relevant = labels[labels["relevant"] == "1"].groupby("qid")["pmid"].apply(set).to_dict()
    judged = labels.groupby("qid")["pmid"].apply(set).to_dict()
    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    questions = [q for q in questions if q["qid"] in relevant]

    bm25 = BM25Retriever.from_parquet(args.chunks)
    dense = DenseRetriever.load(args.index)
    hybrid = HybridRetriever([bm25, dense])
    reranker_name = args.reranker or DEFAULT_RERANKER
    dense_rr = RerankingRetriever(dense, model_name=reranker_name)
    hybrid_rr = RerankingRetriever(hybrid, model=dense_rr.model, model_name=reranker_name)
    systems = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "dense+rerank": dense_rr,
        "hybrid+rerank": hybrid_rr,
    }

    texts = dict(zip(bm25.pmids, bm25.texts, strict=True))
    rows, unjudged = [], {}
    for name, retriever in systems.items():
        for q in questions:
            start = time.perf_counter()
            ranked = ranked_pmids(retriever.search(q["question"], k=50))
            elapsed_ms = (time.perf_counter() - start) * 1000
            rel = relevant[q["qid"]]
            for pmid in ranked[:10]:
                if pmid not in judged[q["qid"]]:
                    unjudged.setdefault((q["qid"], pmid), (q["question"], []))[1].append(name)
            rows.append(
                {
                    "system": name,
                    "qid": q["qid"],
                    "type": q["type"],
                    "recall@5": recall_at_k(ranked, rel, 5),
                    "recall@10": recall_at_k(ranked, rel, 10),
                    "mrr": reciprocal_rank(ranked, rel),
                    "judged@10": judged_at_k(ranked, judged[q["qid"]], 10),
                    "ms_per_query": elapsed_ms,
                }
            )

    df = pd.DataFrame(rows)
    metrics = ["recall@5", "recall@10", "mrr", "judged@10", "ms_per_query"]
    overall = df.groupby("system", sort=False)[metrics].mean()
    by_type = df.pivot_table(index="type", columns="system", values="recall@10", sort=False)

    print(f"\n{len(questions)} questions, pooled relevance labels, reranker: {reranker_name}\n")
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
        "reranker": reranker_name,
        "overall": overall.round(4).to_dict(orient="index"),
        "recall@10_by_type": by_type.round(4).to_dict(orient="index"),
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")

    if unjudged:
        new_rows = []
        for (qid, pmid), (question, names) in sorted(unjudged.items()):
            title, _, abstract = texts[pmid].partition("\n\n")
            new_rows.append(
                {
                    "qid": qid,
                    "question": question,
                    "pmid": pmid,
                    "title": title,
                    "abstract_start": abstract[:300],
                    "found_by": ", ".join(names),
                    "relevant": "",
                }
            )
        pd.DataFrame(new_rows).to_csv(args.unjudged_out, index=False, encoding="utf-8-sig")
        print(f"{len(new_rows)} unlabeled papers in some top 10 -> {args.unjudged_out}")


if __name__ == "__main__":
    main()
