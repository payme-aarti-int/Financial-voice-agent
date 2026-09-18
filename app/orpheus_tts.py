from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class OrpheusConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
    voice: str = os.getenv("TTS_VOICE", "tara")
    response_format: str = os.getenv("TTS_FORMAT", "wav")
    path: str = os.getenv("TTS_PATH", "/audio/speech")
    timeout_s: float = float(os.getenv("TTS_TIMEOUT_S", "30"))


@dataclass
class Speech:
    ok: bool
    wav_path: str | None = None
    audio: bytes | None = None
    played: bool = False
    error: str | None = None
    latency_ms: float | None = None


class OrpheusTTS:
    def __init__(self, config: OrpheusConfig | None = None) -> None:
        self.config = config or OrpheusConfig()
        if not self.config.api_key:
            raise RuntimeError("LLM_API_KEY is empty -- check .env and app/__init__.py")
        self._client = httpx.Client(timeout=self.config.timeout_s)

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        """Text in, audio bytes out. Never raises: a raised exception means the
        user hears silence and cannot tell whether the agent broke."""
        text = (text or "").strip()
        if not text:
            return Speech(ok=False, error="empty text")

        payload = {
            "model": self.config.model,
            "input": text,
            "voice": self.config.voice,
            "response_format": self.config.response_format,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.config.base_url}{self.config.path}",
                json=payload,
                headers=headers,
            )
        except Exception as exc:
            return Speech(ok=False, error=f"{type(exc).__name__}: {exc}")

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            detail = response.text[:400] if response.text else "(empty body)"
            return Speech(
                ok=False,
                error=f"HTTP {response.status_code}: {detail}",
                latency_ms=elapsed_ms,
            )

        audio = response.content
        if not audio:
            return Speech(ok=False, error="200 but empty body", latency_ms=elapsed_ms)

        written = None
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(audio)
            written = str(p)

        return Speech(ok=True, wav_path=written, audio=audio, latency_ms=elapsed_ms)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _self_test(text: str) -> None:
    import app  # noqa: F401  -- triggers load_dotenv()

    c = OrpheusConfig()
    print(f"model={c.model}")
    print(f"voice={c.voice}  format={c.response_format}")
    print(f"url={c.base_url}{c.path}")
    print(f"key length={len(c.api_key)}")
    print()

    with OrpheusTTS(c) as tts:
        s = tts.synthesize(text, out_path="audio/output/orpheus_test.wav")

    if s.ok:
        print(f"OK   {len(s.audio):,} bytes in {s.latency_ms:.0f} ms")
        print(f"     saved {s.wav_path}")
    else:
        print(f"FAILED after {s.latency_ms or 0:.0f} ms")
        print(f"  {s.error}")
        print()
        print("  401 -> credentials; check LLM_API_KEY")
        print("  404 -> wrong model id or path; try TTS_PATH variants")
        print("  400 -> parameter wrong; body names it (usually TTS_VOICE)")
        print("  429 -> rate limited")


if __name__ == "__main__":
    import sys

    _self_test(" ".join(sys.argv[1:]) or "Revenue in March was 128 thousand dollars.")