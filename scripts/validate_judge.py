"""Check the LLM judge with negative controls: claims paired with unrelated passages.

A judge that calls these "supported" is too lenient, and its groundedness scores can't be trusted.

Usage:  uv run python scripts/validate_judge.py --judge-model gemini-3.1-flash-lite
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="reports/answers_v1.jsonl")
    parser.add_argument("--positives", default="reports/answer_judgements_v1.jsonl")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--out", default="reports/judge_validation_v1.json")
    parser.add_argument("--details-out", default="reports/judge_negatives_v1.jsonl")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    from biomed_rag.evaluation.answer_metrics import judge_claim, make_negative_controls
    from biomed_rag.generation.llm import make_llm

    judge = make_llm(args.provider, args.judge_model, args.reasoning_effort)
    lines = Path(args.answers).read_text(encoding="utf-8").splitlines()
    answers = [json.loads(line) for line in lines if line.strip()]
    controls = make_negative_controls(answers, seed=args.seed)
    print(f"{len(controls)} negative controls (seed {args.seed})")

    details = []
    for i, c in enumerate(controls, start=1):
        try:
            verdict = judge_claim(judge, c["claim"], {c["pmid"]: c["text"]}).verdict
        except Exception as error:
            verdict = f"error: {type(error).__name__}"
        details.append(c | {"verdict": verdict})
        print(f"[{i}/{len(controls)}] {c['qid']} vs {c['donor_qid']}: {verdict}")
        time.sleep(args.sleep)

    neg = Counter(d["verdict"] for d in details if not d["verdict"].startswith("error"))
    pos_lines = Path(args.positives).read_text(encoding="utf-8").splitlines()
    pos = Counter(json.loads(line)["verdict"] for line in pos_lines if line.strip())
    n_neg, n_pos = sum(neg.values()), sum(pos.values())
    specificity = neg["unsupported"] / n_neg if n_neg else float("nan")
    sensitivity = pos["supported"] / n_pos if n_pos else float("nan")
    summary = {
        "judge_model": args.judge_model,
        "negatives_judged": n_neg,
        "negatives": dict(neg),
        "positives": dict(pos),
        "specificity (negatives called unsupported)": round(specificity, 3),
        "sensitivity (real claims called supported)": round(sensitivity, 3),
        "balanced_accuracy": round((specificity + sensitivity) / 2, 3),
    }
    print("\n" + json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps(summary, indent=2))
    with Path(args.details_out).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in details)
    print(f"\nSaved {args.out} and {args.details_out}")


if __name__ == "__main__":
    main()
