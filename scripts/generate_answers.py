"""Answer every golden question and save the answers (generation only, no judging).

Usage:  uv run python scripts/generate_answers.py
"""

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="golden/questions_v1.jsonl")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    parser.add_argument("--out", default="reports/answers_v1.jsonl")
    parser.add_argument(
        "--sleep", type=float, default=5.0, help="pause between calls (rate limits)"
    )
    parser.add_argument("--reasoning-effort", default="low", help="none, low, medium or high")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.generation.answer import AnswerGenerator
    from biomed_rag.generation.llm import make_llm
    from biomed_rag.retrieval.dense import DenseRetriever

    llm = make_llm(args.provider, args.model, args.reasoning_effort)
    generator = AnswerGenerator(DenseRetriever.load(args.index), llm, k=args.k)
    lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: keep finished answers and only redo missing or failed questions.
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "error" not in record:
                done[record["qid"]] = line
    out.write_text("".join(line + "\n" for line in done.values()), encoding="utf-8")
    if done:
        print(f"Resuming: {len(done)} answers already saved")

    with out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            if q["qid"] in done:
                continue
            start = time.perf_counter()
            record = {
                "qid": q["qid"],
                "type": q["type"],
                "question": q["question"],
                "provider": args.provider,
                "model": llm.model,
            }
            try:
                result = generator.answer(q["question"])
                record |= {
                    "abstained": result.answer.abstained,
                    "abstain_reason": result.answer.abstain_reason,
                    "claims": [c.model_dump() for c in result.answer.claims],
                    "evidence": [{"pmid": r.pmid, "text": r.text} for r in result.evidence],
                    "invalid_citations": result.invalid_citations,
                    "uncited_claims": result.uncited_claims,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            except Exception as error:  # keep going; record the failure
                record["error"] = f"{type(error).__name__}: {error}"
            record["seconds"] = round(time.perf_counter() - start, 2)
            f.write(json.dumps(record) + "\n")
            f.flush()  # saved immediately, so Ctrl+C never loses finished answers
            status = record.get("error") or f"{len(record.get('claims', []))} claims"
            print(f"[{i}/{len(questions)}] {q['qid']}  {status}  ({record['seconds']} s)")
            time.sleep(args.sleep)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
