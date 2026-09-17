from __future__ import annotations   # MUST be first statement

import argparse

from dotenv import load_dotenv

from app.stt.whisper_stt import WhisperSTT
from app.telemetry import Telemetry

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
 
import httpx
 
 
@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))
 
    temperature: float = 0.2
    max_tokens: int = 512
 
 
@dataclass
class Completion:
    text: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
 
 
class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
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
 
    def __enter__(self) -> "LLMClient":
        return self
 
    def __exit__(self, *exc: object) -> None:
        self.close()
 
 
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
 
        return Completion(
            text=choice["message"].get("content") or "",
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
 
 
if __name__ == "__main__":
    with LLMClient() as client:
        print(f"model={client.config.model} base_url={client.config.base_url}")
        result, _ = client.stream_measured(
            [{"role": "user", "content": "Say 'pipeline connected' and nothing else."}]
        )
        print(f'response: "{result.text.strip()}"')
        print(f"TTFT: {result.ttft_ms:.0f} ms   total: {result.total_ms:.0f} ms")
 
