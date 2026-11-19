"""Ask the research agent a question and show every step it takes.

Usage:  uv run python scripts/ask_agent.py "Do anti-amyloid antibodies slow cognitive decline?"
"""

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.agent.graph import build_agent, run_agent
    from biomed_rag.generation.llm import RetryingLLM, make_llm
    from biomed_rag.retrieval.dense import DenseRetriever

    llm = RetryingLLM(make_llm(args.provider, args.model, args.reasoning_effort))
    agent = build_agent(DenseRetriever.load(args.index), llm, max_rounds=args.max_rounds)

    start = time.perf_counter()
    state = run_agent(agent, args.question)
    seconds = time.perf_counter() - start

    print(f"\nQ: {args.question}   [{args.provider}: {llm.model}]\n\nTRACE")
    for step in state["trace"]:
        details = {k: v for k, v in step.items() if k != "step"}
        print(f"  {step['step']:<8} {details}")

    result = state["result"]
    print("\nANSWER")
    if result.answer.abstained:
        print(f"  ABSTAINED: {result.answer.abstain_reason}")
    for claim in result.answer.claims:
        print(f"  - {claim.text} [{', '.join(claim.pmids) or 'NO CITATION'}]")
    print(f"\nInvalid citations: {result.invalid_citations or 'none'}")
    print(f"Tokens: {state['tokens']}   Retries: {llm.retries}   Time: {seconds:.1f} s")


if __name__ == "__main__":
    main()
