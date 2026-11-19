import json

from biomed_rag.agent.graph import GRADE_PROMPT, PLAN_PROMPT, build_agent, run_agent
from biomed_rag.generation.llm import LLMReply
from biomed_rag.retrieval.base import SearchResult


class FakeRetriever:
    """Returns different papers for different queries, and records every query."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.queries = []

    def search(self, query, k=10):
        self.queries.append(query)
        tag = "trial" if "trial" in query else "mouse" if "mouse" in query else "general"
        tag = self.prefix + tag
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
NOT_ENOUGH = {"sufficient": False, "next_query": "general again", "tool": "corpus"}


def steps(state):
    return [s["step"] for s in state["trace"]]


def test_sufficient_evidence_answers_after_one_round():
    llm = ScriptedLLM({"queries": ["general q"]}, [{"sufficient": True}], ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm), "q")
    assert steps(state) == ["plan", "retrieve", "grade", "answer"]
    assert state["result"].invalid_citations == []
    assert state["tokens"] == 45


def test_insufficient_evidence_triggers_a_new_search():
    grades = [{"sufficient": False, "next_query": "phase 3 trial"}, {"sufficient": True}]
    retriever = FakeRetriever()
    llm = ScriptedLLM({"queries": ["general q", "mouse q"]}, grades, ANSWER)
    state = run_agent(build_agent(retriever, llm), "q")
    assert retriever.queries == ["general q", "mouse q", "phase 3 trial"]
    assert any(r.pmid.startswith("trial") for r in state["evidence"])


def test_round_limit_stops_the_loop():
    grades = [
        {"sufficient": False, "next_query": "trial"},
        {"sufficient": False, "next_query": "mouse"},
        {"sufficient": False, "next_query": "x"},
    ]
    llm = ScriptedLLM({"queries": ["general q"]}, grades, ANSWER)
    run_agent(build_agent(FakeRetriever(), llm, max_rounds=2), "q")
    assert llm.calls == ["plan", "grade", "grade", "answer"]


def test_evidence_is_deduplicated_by_paper():
    llm = ScriptedLLM({"queries": ["general a", "general b"]}, [{"sufficient": True}], ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm, per_query=5), "q")
    pmids = [r.pmid for r in state["evidence"]]
    assert len(pmids) == len(set(pmids)) == 5  # both queries return the same 5 papers


def test_unreadable_plan_falls_back_to_the_question():
    retriever = FakeRetriever()
    llm = ScriptedLLM("not a dict", [{"sufficient": True}], ANSWER)
    run_agent(build_agent(retriever, llm), "original question")
    assert retriever.queries == ["original question"]


def test_no_new_evidence_without_live_tool_stops_early():
    llm = ScriptedLLM({"queries": ["general q"]}, [NOT_ENOUGH, NOT_ENOUGH, NOT_ENOUGH], ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm, max_rounds=3), "q")
    # round 2 finds nothing new, so the agent answers instead of trying a third time
    assert llm.calls == ["plan", "grade", "grade", "answer"]
    assert state["trace"][-2]["guardrail"] == "no new evidence found -> stopping"


def test_exhausted_corpus_switches_to_live_pubmed():
    live = FakeRetriever(prefix="live-")
    llm = ScriptedLLM(
        {"queries": ["general q"]}, [NOT_ENOUGH, NOT_ENOUGH, {"sufficient": True}], ANSWER
    )
    state = run_agent(build_agent(FakeRetriever(), llm, live_retriever=live), "q")
    tools = [s["tool"] for s in state["trace"] if s["step"] == "retrieve"]
    assert tools == ["corpus", "corpus", "pubmed"]
    assert any(r.pmid.startswith("live-") for r in state["evidence"])


def test_grader_can_choose_pubmed_directly():
    live = FakeRetriever(prefix="live-")
    grades = [
        {"sufficient": False, "next_query": "newest trial", "tool": "pubmed"},
        {"sufficient": True},
    ]
    llm = ScriptedLLM({"queries": ["general q"]}, grades, ANSWER)
    state = run_agent(build_agent(FakeRetriever(), llm, live_retriever=live), "q")
    assert live.queries == ["newest trial"]
    assert state["used_live"]


def test_pubmed_choice_without_live_tool_falls_back_to_corpus():
    grades = [{"sufficient": False, "next_query": "trial", "tool": "pubmed"}, {"sufficient": True}]
    retriever = FakeRetriever()
    llm = ScriptedLLM({"queries": ["general q"]}, grades, ANSWER)
    run_agent(build_agent(retriever, llm), "q")
    assert retriever.queries == ["general q", "trial"]
