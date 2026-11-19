"""A thin, swappable interface to large language models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMReply:
    """The model's text plus token counts, so every call's cost can be tracked."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    """Anything that turns a system prompt and a user message into a reply."""

    def complete(self, system: str, user: str) -> LLMReply: ...


# Many providers speak the same "OpenAI-compatible" API; only the address and key differ.
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model": "qwen2.5:7b",
    },
}


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat API: Gemini, Ollama, Groq, OpenAI and others."""

    def __init__(self, model: str, base_url: str, api_key: str, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI  # imported here so tests never need the SDK

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model

    def complete(self, system: str, user: str) -> LLMReply:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        return LLMReply(
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


class AnthropicLLM:
    """Claude via the Anthropic API. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 4000) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMReply:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMReply(text, response.usage.input_tokens, response.usage.output_tokens)


def make_llm(provider: str, model: str | None = None) -> LLM:
    """Build an LLM by provider name, e.g. make_llm("gemini") or make_llm("ollama")."""
    if provider == "anthropic":
        return AnthropicLLM(model or "claude-sonnet-5-5")
    if provider not in PROVIDERS:
        known = ", ".join([*PROVIDERS, "anthropic"])
        raise ValueError(f"unknown provider {provider!r}; choose one of: {known}")
    preset = PROVIDERS[provider]
    key_env = preset["key_env"]
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if not api_key:
        raise ValueError(f"set {key_env} in your .env file")
    return OpenAICompatibleLLM(model or preset["model"], preset["base_url"], api_key)
