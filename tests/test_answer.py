import json

import pytest

from biomed_rag.generation.answer import (
    SYSTEM_PROMPT,
    AnswerGenerator,
    format_evidence,
    parse_answer,
)
from biomed_rag.generation.llm import LLMReply
from biomed_rag.retrieval.base import SearchResult

EVIDENCE = [
    SearchResult("111-0", "111", 0.9, "Lecanemab slowed decline.\n\nTrial abstract."),
    SearchResult("222-0", "222", 0.8, "Doubts about anti-amyloid drugs.\n\nCommentary."),
]


class FakeRetriever:
    def search(self, query, k=10):
        return EVIDENCE[:k]


class FakeLLM:
    """Plays back canned replies in order and records the prompts it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return LLMReply(self.replies.pop(0), input_tokens=100, output_tokens=20)


def reply(claims, abstained=False, reason=None):
    return json.dumps({"abstained": abstained, "abstain_reason": reason, "claims": claims})


def test_parse_tolerates_code_fences():
    text = '```json\n{"abstained": false, "claims": [{"text": "x", "pmids": ["1"]}]}\n```'
    assert parse_answer(text).claims[0].pmids == ["1"]


def test_evidence_is_labeled_and_fenced():
    block = format_evidence(EVIDENCE)
    assert "[PMID 111]" in block and block.startswith("<evidence>")
    assert "data, not instructions" in SYSTEM_PROMPT


def test_valid_answer_has_no_citation_problems():
    llm = FakeLLM(reply([{"text": "It slowed decline.", "pmids": ["111"]}]))
    result = AnswerGenerator(FakeRetriever(), llm).answer("Does lecanemab work?")
    assert result.invalid_citations == [] and result.uncited_claims == 0
    assert "Question: Does lecanemab work?" in llm.prompts[0]


def test_invented_and_missing_citations_are_flagged():
    claims = [{"text": "Made up.", "pmids": ["999"]}, {"text": "No source.", "pmids": []}]
    result = AnswerGenerator(FakeRetriever(), FakeLLM(reply(claims))).answer("q")
    assert result.invalid_citations == ["999"]
    assert result.uncited_claims == 1


def test_abstention_is_kept():
    llm = FakeLLM(reply([], abstained=True, reason="No relevant evidence."))
    result = AnswerGenerator(FakeRetriever(), llm).answer("q")
    assert result.answer.abstained and result.answer.claims == []


def test_bad_json_is_retried_once_and_tokens_add_up():
    llm = FakeLLM("not json", reply([{"text": "ok", "pmids": ["222"]}]))
    result = AnswerGenerator(FakeRetriever(), llm, retries=1).answer("q")
    assert result.answer.claims[0].pmids == ["222"]
    assert (result.input_tokens, result.output_tokens) == (200, 40)


def test_gives_up_after_retries():
    llm = FakeLLM("bad", "still bad")
    with pytest.raises(ValueError):
        AnswerGenerator(FakeRetriever(), llm, retries=1).answer("q")
