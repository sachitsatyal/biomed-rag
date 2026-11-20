import pytest
from pydantic import ValidationError
from serving_fakes import fake_components

from biomed_rag.serving.service import AskRequest, answer_question


def test_rag_mode_returns_claims_and_only_cited_evidence():
    response = answer_question(fake_components(), AskRequest(question="Does it work?", mode="rag"))
    assert [c.pmids for c in response.claims] == [["111"], ["222"]]
    assert [e.pmid for e in response.evidence] == ["111", "222"]  # 333 was not cited
    assert response.evidence[0].url == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert response.tokens == 120 and response.trace == []


def test_agent_mode_returns_trace_and_marks_live_sources():
    response = answer_question(fake_components(), AskRequest(question="Does it work?"))
    assert response.mode == "agent" and response.tokens == 500
    assert [s["step"] for s in response.trace] == ["plan", "answer"]
    assert {e.pmid: e.source for e in response.evidence} == {"111": "corpus", "222": "pubmed"}


def test_request_validation_rejects_bad_input():
    with pytest.raises(ValidationError):
        AskRequest(question="hi")
    with pytest.raises(ValidationError):
        AskRequest(question="A long enough question?", mode="magic")
