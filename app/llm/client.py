from __future__ import annotations   # MUST be first statement

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx
from mlflow.entities import SpanType

from dotenv import load_dotenv

from app import observability

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Connection settings for an OpenAI-compatible chat-completions server.

    The defaults are Groq's (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL), read at
    instantiation so a late load_dotenv() is still honoured. Other providers
    build an LLMConfig from their own env vars -- see VLLMClient / LlamaCppClient.
    """

    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "30")))

    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class Completion:
    text: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    """Every LLM provider must implement this. The agent loop
    (app/agent/loop.py) only ever talks to this interface."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> Completion: ...

    @abstractmethod
    def stream(self, messages: list[dict[str, Any]], **overrides: Any) -> Iterator[str]: ...

    @abstractmethod
    def stream_measured(
        self, messages: list[dict[str, Any]], **overrides: Any
    ) -> tuple[Completion, list[str]]: ...

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseLLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _OpenAICompatLLMClient(BaseLLMClient):
    """Shared httpx transport for any server speaking the OpenAI
    /chat/completions protocol (Groq, vLLM, llama.cpp, ...). Providers
    subclass this and only decide which env vars fill the LLMConfig.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        log.info("model=%s base_url=%s", self.config.model, self.config.base_url)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self._http = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout_s,
        )

    def close(self) -> None:
        self._http.close()

    @observability.trace(name="llm.complete", span_type=SpanType.LLM)
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> Completion:

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **overrides,
        }
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        response = self._http.post("/chat/completions", json=payload)
        total_ms = (time.perf_counter() - start) * 1000

        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        # reasoning models put output in 'reasoning' when content is empty
        content = message.get("content") or ""
        if not content.strip():
            content = message.get("reasoning") or message.get("reasoning_content") or ""

        return Completion(
            text=content,
            total_ms=total_ms,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        **overrides: Any,
    ) -> Iterator[str]:

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            **overrides,
        }

        with self._http.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                content = delta.get("content")
                if content:
                    yield content

    def stream_measured(
        self,
        messages: list[dict[str, Any]],
        **overrides: Any,
    ) -> tuple[Completion, list[str]]:

        start = time.perf_counter()
        ttft_ms: float | None = None
        parts: list[str] = []

        for delta in self.stream(messages, **overrides):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            parts.append(delta)

        return (
            Completion(
                text="".join(parts),
                ttft_ms=ttft_ms,
                total_ms=(time.perf_counter() - start) * 1000,
            ),
            parts,
        )


class GroqLLMClient(_OpenAICompatLLMClient):
    """Groq-hosted models. Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL."""

    def __init__(self, config: LLMConfig | None = None):
        config = config or LLMConfig()
        if not config.api_key:
            log.warning("LLM_API_KEY is empty -- Groq requests will be rejected; set it in .env")
        super().__init__(config)


class VLLMClient(_OpenAICompatLLMClient):
    """Company vLLM endpoint (OpenAI-compatible). Reads COMPANY_LLM_BASE_URL,
    COMPANY_LLM_MODEL, COMPANY_LLM_API_KEY (optional -- vLLM usually runs
    without auth)."""

    def __init__(self, config: LLMConfig | None = None):
        if config is None:
            base_url = os.getenv("COMPANY_LLM_BASE_URL", "http://localhost:8000/v1")
            model = os.getenv("COMPANY_LLM_MODEL", "")
            if not base_url:
                raise ValueError("LLM_PROVIDER=vllm but COMPANY_LLM_BASE_URL is empty -- set it in .env")
            if not model:
                raise ValueError(
                    "LLM_PROVIDER=vllm but COMPANY_LLM_MODEL is empty -- set it in .env "
                    "(e.g. COMPANY_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct)"
                )
            config = LLMConfig(
                base_url=base_url,
                api_key=os.getenv("COMPANY_LLM_API_KEY", ""),
                model=model,
            )
        super().__init__(config)


class LlamaCppClient(_OpenAICompatLLMClient):
    """llama.cpp server (llama-server, OpenAI-compatible). Reads
    LLAMACPP_BASE_URL, LLAMACPP_MODEL. llama-server ignores auth unless
    started with --api-key, so LLAMACPP_API_KEY is optional."""

    def __init__(self, config: LLMConfig | None = None):
        if config is None:
            base_url = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
            model = os.getenv("LLAMACPP_MODEL", "")
            if not base_url:
                raise ValueError("LLM_PROVIDER=llamacpp but LLAMACPP_BASE_URL is empty -- set it in .env")
            if not model:
                raise ValueError(
                    "LLM_PROVIDER=llamacpp but LLAMACPP_MODEL is empty -- set it in .env "
                    "(e.g. LLAMACPP_MODEL=qwen2.5-7b)"
                )
            config = LLMConfig(
                base_url=base_url,
                api_key=os.getenv("LLAMACPP_API_KEY", ""),
                model=model,
            )
        super().__init__(config)


LLM_PROVIDERS: dict[str, type[BaseLLMClient]] = {
    "groq": GroqLLMClient,
    "vllm": VLLMClient,
    "llamacpp": LlamaCppClient,
}


def get_llm_client() -> BaseLLMClient:
    """Build the LLM client named by LLM_PROVIDER in .env (default: groq).
    This is the only place LLM_PROVIDER is read."""
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    cls = LLM_PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r} -- set LLM_PROVIDER in .env to one of: "
            + ", ".join(LLM_PROVIDERS)
        )
    log.info("LLM_PROVIDER=%s -> %s", provider, cls.__name__)
    return cls()


# Backward compatibility -- `from app.llm.client import LLMClient` still works
# and still means the Groq client.
LLMClient = GroqLLMClient


if __name__ == "__main__":
    import app  # noqa: F401  -- triggers load_dotenv() + logging setup

    with get_llm_client() as client:
        result, _ = client.stream_measured(
            [{"role": "user", "content": "Say 'pipeline connected' and nothing else."}]
        )
        log.info('response: "%s"', result.text.strip())
        log.info("TTFT: %.0f ms   total: %.0f ms", result.ttft_ms, result.total_ms)
