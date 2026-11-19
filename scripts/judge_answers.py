"""Judge saved answers: citation validity, cited-paper relevance and groundedness.

Usage:  uv run python scripts/judge_answers.py
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="reports/answers_v1.jsonl")
    parser.add_argument("--labels", default="golden/labels_v3.csv")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-flash",
        help="use a different model from the answerer to reduce self-bias",
    )
    parser.add_argument("--out", default="reports/answer_eval_v1.json")
    parser.add_argument("--claims-out", default="reports/answer_judgements_v1.jsonl")
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    from biomed_rag.evaluation.answer_metrics import cited_relevance, judge_claim
    from biomed_rag.generation.llm import RetryingLLM, make_llm

    judge = RetryingLLM(make_llm(args.provider, args.judge_model, args.reasoning_effort))
    labels = pd.read_csv(args.labels, dtype=str, encoding="utf-8-sig")
    relevant = labels[labels["relevant"] == "1"].groupby("qid")["pmid"].apply(set).to_dict()
    judged = labels.groupby("qid")["pmid"].apply(set).to_dict()
    lines = Path(args.answers).read_text(encoding="utf-8").splitlines()
    answers = [json.loads(line) for line in lines if line.strip()]

    rows, claim_rows = [], []
    for a in answers:
        if "error" in a:
            rows.append({"qid": a["qid"], "type": a["type"], "error": True})
            continue
        passages = {e["pmid"]: e["text"] for e in a["evidence"]}
        cited = [p for c in a["claims"] for p in c["pmids"]]
        rel_rate, unlabeled = cited_relevance(
            cited, relevant.get(a["qid"], set()), judged.get(a["qid"], set())
        )
        verdicts = []
        for c in a["claims"]:
            cited_passages = {p: passages[p] for p in c["pmids"] if p in passages}
            if not cited_passages:
                verdict, reason = "unsupported", "no valid citation"
            else:
                try:
                    j = judge_claim(judge, c["text"], cited_passages)
                    verdict, reason = j.verdict, j.reason
                except Exception as error:
                    verdict, reason = "judge_error", str(error)[:200]
                time.sleep(args.sleep)
            verdicts.append(verdict)
            claim_rows.append(
                {
                    "qid": a["qid"],
                    "claim": c["text"],
                    "pmids": c["pmids"],
                    "verdict": verdict,
                    "reason": reason,
                }
            )
        scored = [v for v in verdicts if v != "judge_error"]
        rows.append(
            {
                "qid": a["qid"],
                "type": a["type"],
                "error": False,
                "abstained": a["abstained"],
                "n_claims": len(a["claims"]),
                "invalid_citations": len(a["invalid_citations"]),
                "uncited_claims": a["uncited_claims"],
                "cited_relevant_rate": rel_rate,
                "unlabeled_citations": unlabeled,
                "supported": sum(v == "supported" for v in scored) / len(scored)
                if scored
                else math.nan,
                "supported_or_partial": (
                    sum(v in ("supported", "partial") for v in scored) / len(scored)
                    if scored
                    else math.nan
                ),
                "seconds": a["seconds"],
            }
        )
        print(f"{a['qid']}: {verdicts}")

    df = pd.DataFrame(rows)
    ok = df[~df["error"]]
    summary = {
        "n_questions": len(df),
        "errors": int(df["error"].sum()),
        "abstention_rate": float(ok["abstained"].mean()),
        "mean_claims": float(ok["n_claims"].mean()),
        "invalid_citation_total": int(ok["invalid_citations"].sum()),
        "uncited_claim_total": int(ok["uncited_claims"].sum()),
        "cited_relevant_rate": float(ok["cited_relevant_rate"].mean()),
        "unlabeled_citation_total": int(ok["unlabeled_citations"].sum()),
        "claims_supported": float(ok["supported"].mean()),
        "claims_supported_or_partial": float(ok["supported_or_partial"].mean()),
        "mean_seconds_per_answer": float(ok["seconds"].mean()),
        "judge_model": args.judge_model,
        "labels": args.labels,
    }
    by_type = ok.groupby("type")[["supported", "cited_relevant_rate"]].mean().round(3)

    print(
        "\n"
        + "\n".join(
            f"{k:>30}: {v:.3f}" if isinstance(v, float) else f"{k:>30}: {v}"
            for k, v in summary.items()
        )
    )
    print("\nBy question type\n" + by_type.to_string())
    Path(args.out).write_text(json.dumps(summary, indent=2))
    with Path(args.claims_out).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in claim_rows)
    print(f"\nSaved {args.out} and {args.claims_out}")


if __name__ == "__main__":
    main()
