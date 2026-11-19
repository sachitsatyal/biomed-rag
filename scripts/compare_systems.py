"""Compare answer systems (e.g. plain RAG vs agent) side by side.

Usage:  uv run python scripts/compare_systems.py \
            --run rag=reports/rag_answers.jsonl:reports/rag_eval.json \
            --run agent=reports/agent_answers.jsonl:reports/agent_eval.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def answer_stats(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ok = [r for r in records if "error" not in r]
    answerable = [r for r in ok if r["type"] != "unanswerable"]
    unanswerable = [r for r in ok if r["type"] == "unanswerable"]

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else float("nan")

    return {
        "answered OK / total": f"{len(ok)} / {len(records)}",
        "abstained on answerable (lower is better)": mean(r["abstained"] for r in answerable),
        "abstained on unanswerable (higher is better)": mean(r["abstained"] for r in unanswerable),
        "claims per answer": mean(len(r["claims"]) for r in answerable),
        "passages used as evidence": mean(len(r["evidence"]) for r in ok),
        "search rounds": mean(r.get("rounds", 1) for r in ok),
        "used live PubMed": mean("pubmed" in r.get("tools", []) for r in ok),
        "tokens per question": mean(r.get("tokens", 0) for r in ok),
        "seconds per question": mean(r["seconds"] for r in ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="name=answers.jsonl:eval.json (repeat for each system)",
    )
    parser.add_argument("--out", default="reports/system_comparison_v1.csv")
    args = parser.parse_args()

    columns = {}
    for spec in args.run:
        name, paths = spec.split("=", 1)
        answers, evaluation = paths.split(":", 1) if ":" in paths else (paths, None)
        stats = answer_stats(Path(answers))
        if evaluation and Path(evaluation).exists():
            ev = json.loads(Path(evaluation).read_text())
            stats |= {
                "claims judged supported": ev["claims_supported"],
                "cited papers labeled relevant": ev["cited_relevant_rate"],
                "invalid citations (total)": ev["invalid_citation_total"],
                "uncited claims (total)": ev["uncited_claim_total"],
            }
        columns[name] = stats

    table = pd.DataFrame(columns)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    table.to_csv(args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
