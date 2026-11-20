"""Fake pipeline pieces shared by the serving tests (no model, no network)."""

from biomed_rag.generation.answer import CitedAnswer, Claim, GeneratedAnswer
from biomed_rag.retrieval.base import SearchResult
from biomed_rag.serving.service import Components

EVIDENCE = [
    SearchResult("111-0", "111", 0.9, "Lecanemab slowed decline.\n\nAbstract."),
    SearchResult("222-live", "222", 0.8, "Doubts about anti-amyloid drugs.\n\nAbstract."),
    SearchResult("333-0", "333", 0.7, "Unrelated paper.\n\nAbstract."),
]


def make_result(question: str) -> GeneratedAnswer:
    answer = CitedAnswer(
        abstained=False,
        claims=[
            Claim(text="It slowed decline.", pmids=["111"]),
            Claim(text="Benefit is debated.", pmids=["222"]),
        ],
    )
    return GeneratedAnswer(question, answer, EVIDENCE, input_tokens=100, output_tokens=20)


class FakeGenerator:
    def answer(self, question):
        if "fail" in question:
            raise RuntimeError("LLM down")
        return make_result(question)


class FakeAgent:
    def invoke(self, state):
        trace = [{"step": "plan", "queries": ["q"]}, {"step": "answer", "claims": 2}]
        return {"result": make_result(state["question"]), "tokens": 500, "trace": trace}


def fake_components() -> Components:
    return Components(generator=FakeGenerator(), agent=FakeAgent(), model="fake-model")
