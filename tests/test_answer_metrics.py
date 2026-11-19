import math

from biomed_rag.evaluation.answer_metrics import cited_relevance, judge_claim, parse_judgement
from biomed_rag.generation.llm import LLMReply


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.user = None

    def complete(self, system, user):
        self.user = user
        return LLMReply(self.text)


def test_parse_judgement_tolerates_fences():
    assert parse_judgement('```json\n{"verdict": "partial"}\n```').verdict == "partial"


def test_judge_sees_claim_and_cited_passages():
    llm = FakeLLM('{"verdict": "supported", "reason": "stated"}')
    verdict = judge_claim(llm, "Drug X slows decline.", {"111": "Drug X slowed decline."})
    assert verdict.verdict == "supported"
    assert "Claim: Drug X slows decline." in llm.user and "[PMID 111]" in llm.user


def test_cited_relevance_ignores_unlabeled_papers():
    rate, unlabeled = cited_relevance(["1", "2", "3"], relevant={"1"}, judged={"1", "2"})
    assert rate == 0.5 and unlabeled == 1


def test_cited_relevance_with_nothing_labeled_is_nan():
    rate, unlabeled = cited_relevance(["9"], relevant=set(), judged=set())
    assert math.isnan(rate) and unlabeled == 1
