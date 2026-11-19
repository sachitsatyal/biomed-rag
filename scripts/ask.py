"""Ask one question and get a cited answer.

Usage:  uv run python scripts/ask.py "Does lecanemab slow cognitive decline?"
        uv run python scripts/ask.py "..." --provider ollama
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--provider", default="gemini", help="gemini, ollama or anthropic")
    parser.add_argument("--model", default=None, help="override the provider's default model")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--reasoning-effort", default="low", help="none, low, medium or high")
    parser.add_argument("--index", default="data/indexes/dense_bge-small_abstract")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.generation.answer import AnswerGenerator
    from biomed_rag.generation.llm import make_llm
    from biomed_rag.retrieval.dense import DenseRetriever

    llm = make_llm(args.provider, args.model, args.reasoning_effort)
    generator = AnswerGenerator(DenseRetriever.load(args.index), llm, k=args.k)
    result = generator.answer(args.question)

    print(f"\nQ: {result.question}   [{args.provider}: {llm.model}]\n")
    if result.answer.abstained:
        print(f"ABSTAINED: {result.answer.abstain_reason}")
    for claim in result.answer.claims:
        print(f"- {claim.text} [{', '.join(claim.pmids) or 'NO CITATION'}]")
    print(f"\nInvalid citations: {result.invalid_citations or 'none'}")
    print(f"Uncited claims: {result.uncited_claims}")
    print("\nEvidence retrieved:")
    for r in result.evidence:
        print(f"  {r.pmid}  {r.text.split(chr(10))[0][:80]}")
    print(f"\nTokens: {result.input_tokens} in / {result.output_tokens} out")


if __name__ == "__main__":
    main()
