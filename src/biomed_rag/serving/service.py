"""The service layer: request/response schemas and the logic behind the API.

Kept free of FastAPI so it can be tested (and reused) on its own.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from biomed_rag.agent.graph import run_agent


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    mode: Literal["rag", "agent"] = "agent"


class ClaimOut(BaseModel):
    text: str
    pmids: list[str]


class EvidenceOut(BaseModel):
    pmid: str
    title: str
    url: str
    source: Literal["corpus", "pubmed"]


class AskResponse(BaseModel):
    question: str
    mode: str
    abstained: bool
    abstain_reason: str | None
    claims: list[ClaimOut]
    evidence: list[EvidenceOut]  # only the papers the answer actually cites
    invalid_citations: list[str]
    tokens: int
    seconds: float
    trace: list[dict] = []


@dataclass
class Components:
    """Everything loaded once at startup and shared by all requests."""

    generator: Any  # AnswerGenerator (plain RAG)
    agent: Any  # compiled LangGraph agent
    model: str


def answer_question(components: Components, request: AskRequest) -> AskResponse:
    start = time.perf_counter()
    if request.mode == "rag":
        result = components.generator.answer(request.question)
        tokens, trace = result.input_tokens + result.output_tokens, []
    else:
        state = run_agent(components.agent, request.question)
        result, tokens, trace = state["result"], state["tokens"], state["trace"]

    cited = {pmid for claim in result.answer.claims for pmid in claim.pmids}
    evidence = [
        EvidenceOut(
            pmid=r.pmid,
            title=r.text.split("\n")[0],
            url=f"https://pubmed.ncbi.nlm.nih.gov/{r.pmid}/",
            source="pubmed" if r.chunk_id.endswith("-live") else "corpus",
        )
        for r in result.evidence
        if r.pmid in cited
    ]
    return AskResponse(
        question=request.question,
        mode=request.mode,
        abstained=result.answer.abstained,
        abstain_reason=result.answer.abstain_reason,
        claims=[ClaimOut(text=c.text, pmids=c.pmids) for c in result.answer.claims],
        evidence=evidence,
        invalid_citations=result.invalid_citations,
        tokens=tokens,
        seconds=round(time.perf_counter() - start, 2),
        trace=trace,
    )


def load_components() -> Components:
    """Build the real pipeline from environment settings (called once, at startup)."""
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from biomed_rag.agent.graph import build_agent
    from biomed_rag.data.pubmed import PubMedClient
    from biomed_rag.generation.answer import AnswerGenerator
    from biomed_rag.generation.llm import RetryingLLM, make_llm
    from biomed_rag.retrieval.dense import DenseRetriever
    from biomed_rag.retrieval.pubmed_live import PubMedLiveRetriever

    llm = RetryingLLM(
        make_llm(
            os.environ.get("LLM_PROVIDER", "gemini"),
            os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite"),
            os.environ.get("LLM_REASONING_EFFORT", "low"),
        )
    )
    retriever = DenseRetriever.load(
        os.environ.get("DENSE_INDEX", "data/indexes/dense_bge-small_abstract")
    )
    live = None
    if os.environ.get("ENABLE_LIVE_PUBMED", "true").lower() == "true" and os.environ.get(
        "NCBI_API_KEY"
    ):
        client = PubMedClient(os.environ["NCBI_API_KEY"], os.environ.get("NCBI_EMAIL", ""))
        live = PubMedLiveRetriever(client)
    return Components(
        generator=AnswerGenerator(retriever, llm),
        agent=build_agent(retriever, llm, live_retriever=live),
        model=llm.model,
    )
