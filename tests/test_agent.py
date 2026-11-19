import json

from biomed_rag.agent.graph import GRADE_PROMPT, PLAN_PROMPT, build_agent, run_agent
from biomed_rag.generation.llm import LLMReply
from biomed_rag.retrieval.base import SearchResult


class FakeRetriever:
    """Returns different chunks for different queries, and records every query."""

    def __init__(self) -> None:
        self.queries = []

    def search(self, query, k=10):
        self.queries.append(query)
        tag = "trial" if "trial" in query else "mouse" if "mouse" in query else "general"
        return [SearchResult(f"{tag}{i}-0", f"{tag}{i}", 1.0, f"{tag} paper {i}") for i in range(k)]


class ScriptedLLM:
    """Answers each kind of prompt with the next scripted reply for that kind."""

    def __init__(self, plan, grades, answer) -> None:
        self.plan, self.grades, self.answer = plan, list(grades), answer
        self.calls = []

    def complete(self, system, user):
        if system == PLAN_PROMPT:
            self.calls.append("plan")
            return LLMReply(json.dumps(self.plan), 10, 5)
        if system == GRADE_PROMPT:
            self.calls.append("grade")
            return LLMReply(json.dumps(self.grades.pop(0)), 10, 5)
        self.calls.append("answer")
        return LLMReply(json.dumps(self.answer), 10, 5)


ANSWER = {"abstained": False, "claims": [{"text": "x", "pmids": ["general0"]}]}


def test_sufficient_evidence_answers_after_one_round():
    llm = ScriptedLLM({"queries": ["general q"]}, [{"sufficient": True}], ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm), "q")
    assert llm.calls == ["plan", "grade", "answer"]
    assert [s["step"] for s in state["trace"]] == ["plan", "retrieve", "grade", "answer"]
    assert state["result"].invalid_citations == []
    assert state["tokens"] == 45


def test_insufficient_evidence_triggers_a_new_search():
    grades = [{"sufficient": False, "next_query": "phase 3 trial"}, {"sufficient": True}]
    retriever = FakeRetriever()
    llm = ScriptedLLM({"queries": ["general q", "mouse q"]}, grades, ANSWER)
    state = run_agent(build_agent(retriever, llm), "q")
    assert retriever.queries == ["general q", "mouse q", "phase 3 trial"]
    assert state["round"] == 2
    assert any(r.pmid.startswith("trial") for r in state["evidence"])


def test_round_limit_stops_the_loop():
    grades = [{"sufficient": False, "next_query": "trial"}] * 5
    llm = ScriptedLLM({"queries": ["general q"]}, grades, ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm, max_rounds=2), "q")
    assert llm.calls == ["plan", "grade", "grade", "answer"]
    assert state["round"] == 2


def test_evidence_is_deduplicated_and_capped():
    llm = ScriptedLLM({"queries": ["general a", "general b"]}, [{"sufficient": True}], ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm, per_query=5, max_evidence=8), "q")
    ids = [r.chunk_id for r in state["evidence"]]
    assert len(ids) == len(set(ids)) == 5  # both queries return the same 5 chunks


def test_unreadable_plan_falls_back_to_the_question():
    retriever = FakeRetriever()
    llm = ScriptedLLM("not a dict", [{"sufficient": True}], ANSWER)
    run_agent(build_agent(retriever, llm), "original question")
    assert retriever.queries == ["original question"]
