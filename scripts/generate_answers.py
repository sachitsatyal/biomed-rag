"""Answer every golden question with plain RAG or the agent, and save the answers.

Usage:  uv run python scripts/generate_answers.py --system rag   --out reports/rag_answers.jsonl
        uv run python scripts/generate_answers.py --system agent --out reports/agent_answers.jsonl
"""

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

DEFAULT_QUESTIONS = ["golden/questions_v1.jsonl", "golden/questions_unanswerable_v1.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["rag", "agent"], default="rag")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default="low", help="none, low, medium or high")
    parser.add_argument("--k", type=int, default=8, help="passages for plain RAG")
    parser.add_argument("--max-rounds", type=int, default=3, help="agent search rounds")
    parser.add_argument("--no-live", action="store_true", help="agent without live PubMed")
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    parser.add_argument("--out", default="reports/answers_v1.jsonl")
    parser.add_argument("--sleep", type=float, default=5.0, help="pause between questions")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.agent.graph import build_agent, run_agent
    from biomed_rag.data.pubmed import PubMedClient
    from biomed_rag.generation.answer import AnswerGenerator
    from biomed_rag.generation.llm import RetryingLLM, make_llm
    from biomed_rag.retrieval.dense import DenseRetriever
    from biomed_rag.retrieval.pubmed_live import PubMedLiveRetriever

    llm = RetryingLLM(make_llm(args.provider, args.model, args.reasoning_effort))
    retriever = DenseRetriever.load(args.index)
    if args.system == "rag":
        generator = AnswerGenerator(retriever, llm, k=args.k)
    else:
        live = None
        if not args.no_live:
            client = PubMedClient(os.environ["NCBI_API_KEY"], os.environ["NCBI_EMAIL"])
            live = PubMedLiveRetriever(client)
        agent = build_agent(retriever, llm, live_retriever=live, max_rounds=args.max_rounds)

    questions = []
    for path in args.questions:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        questions += [json.loads(line) for line in lines if line.strip()]

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
                "system": args.system,
                "provider": args.provider,
                "model": llm.model,
                "reasoning_effort": args.reasoning_effort,
                "date": date.today().isoformat(),
            }
            try:
                if args.system == "rag":
                    result = generator.answer(q["question"])
                    tokens = result.input_tokens + result.output_tokens
                    extra = {"rounds": 1, "tools": ["corpus"]}
                else:
                    state = run_agent(agent, q["question"])
                    result, tokens = state["result"], state["tokens"]
                    retrieves = [s for s in state["trace"] if s["step"] == "retrieve"]
                    extra = {
                        "rounds": len(retrieves),
                        "tools": [s["tool"] for s in retrieves],
                        "live_pmids": [
                            r.pmid for r in result.evidence if r.chunk_id.endswith("-live")
                        ],
                        "trace": state["trace"],
                    }
                record |= {
                    "abstained": result.answer.abstained,
                    "abstain_reason": result.answer.abstain_reason,
                    "claims": [c.model_dump() for c in result.answer.claims],
                    "evidence": [{"pmid": r.pmid, "text": r.text} for r in result.evidence],
                    "invalid_citations": result.invalid_citations,
                    "uncited_claims": result.uncited_claims,
                    "tokens": tokens,
                    **extra,
                }
            except Exception as error:  # keep going; record the failure
                record["error"] = f"{type(error).__name__}: {error}"
            record["seconds"] = round(time.perf_counter() - start, 2)
            f.write(json.dumps(record) + "\n")
            f.flush()  # saved immediately, so Ctrl+C never loses finished answers
            status = record.get("error") or (
                "ABSTAINED" if record["abstained"] else f"{len(record['claims'])} claims"
            )
            tools = "+".join(record.get("tools", []))
            print(f"[{i}/{len(questions)}] {q['qid']}  {status}  {tools}  ({record['seconds']} s)")
            time.sleep(args.sleep)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
