from types import SimpleNamespace

import pytest

from biomed_rag.generation.llm import LLMReply, OpenAICompatibleLLM, make_llm


class FakeChatClient:
    """Mimics openai.OpenAI().chat.completions.create and records the request."""

    def __init__(self) -> None:
        self.request = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content='{"abstained": true}')
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_openai_compatible_llm_sends_system_and_user_and_counts_tokens():
    client = FakeChatClient()
    reply = OpenAICompatibleLLM("some-model", "http://x", "key", client=client).complete("S", "U")
    assert reply.text == '{"abstained": true}'
    assert (reply.input_tokens, reply.output_tokens) == (50, 5)
    assert [m["role"] for m in client.request["messages"]] == ["system", "user"]
    assert client.request["model"] == "some-model"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        make_llm("not-a-provider")


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        make_llm("gemini")


def test_reasoning_effort_is_sent_only_when_set():
    client = FakeChatClient()
    OpenAICompatibleLLM("m", "http://x", "k", client=client).complete("S", "U")
    assert "reasoning_effort" not in client.request
    OpenAICompatibleLLM("m", "http://x", "k", client=client, reasoning_effort="low").complete(
        "S", "U"
    )
    assert client.request["reasoning_effort"] == "low"


class RateLimitError(Exception):
    """Same class name as the SDK's, which is what RetryingLLM checks."""


class FlakyLLM:
    model = "flaky"

    def __init__(self, failures: int, error: type[Exception] = RateLimitError) -> None:
        self.failures = failures
        self.error = error

    def complete(self, system, user):
        if self.failures:
            self.failures -= 1
            raise self.error("busy")
        return LLMReply("ok")


def test_retrying_llm_waits_longer_each_time_then_succeeds():
    from biomed_rag.generation.llm import RetryingLLM

    waits = []
    llm = RetryingLLM(FlakyLLM(failures=2), base_delay=10, sleep=waits.append)
    assert llm.complete("S", "U").text == "ok"
    assert llm.retries == 2 and 10 <= waits[0] < waits[1]
    assert llm.model == "flaky"


def test_retrying_llm_does_not_retry_real_bugs():
    from biomed_rag.generation.llm import RetryingLLM

    llm = RetryingLLM(FlakyLLM(failures=1, error=ValueError), sleep=lambda s: None)
    with pytest.raises(ValueError):
        llm.complete("S", "U")


def test_retrying_llm_gives_up_after_max_attempts():
    from biomed_rag.generation.llm import RetryingLLM

    llm = RetryingLLM(FlakyLLM(failures=9), max_attempts=3, sleep=lambda s: None)
    with pytest.raises(RateLimitError):
        llm.complete("S", "U")
