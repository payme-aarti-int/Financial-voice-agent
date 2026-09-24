from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# Groq's 429 body reads like "...Please try again in 38m0s." -- parse that
# instead of retrying a doomed request every turn until the quota resets.
_RETRY_AFTER_RE = re.compile(
    r"try again in\s*(?:(?P<hours>\d+)h)?\s*(?:(?P<minutes>\d+)m)?\s*(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
    re.IGNORECASE,
)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Seconds until it's worth trying again, if the server told us."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    match = _RETRY_AFTER_RE.search(response.text or "")
    if match and any(match.groups()):
        hours = float(match.group("hours") or 0)
        minutes = float(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        total = hours * 3600 + minutes * 60 + seconds
        if total > 0:
            return total
    return None


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
        # Set after a 429 so subsequent calls fail fast (no network round
        # trip) instead of retrying a request that's guaranteed to fail
        # again until the quota resets.
        self._rate_limited_until: float | None = None

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        """Text in, audio bytes out. Never raises: a raised exception means the
        user hears silence and cannot tell whether the agent broke."""
        text = (text or "").strip()
        if not text:
            return Speech(ok=False, error="empty text")

        if self._rate_limited_until is not None:
            remaining = self._rate_limited_until - time.monotonic()
            if remaining > 0:
                return Speech(ok=False, error=f"rate limited -- retry in {remaining:.0f}s")
            log.info("Orpheus rate-limit cooldown over -- trying it again")
            self._rate_limited_until = None

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
            if response.status_code == 429:
                retry_after = _parse_retry_after(response)
                if retry_after:
                    self._rate_limited_until = time.monotonic() + retry_after
                    log.warning(
                        "Orpheus rate-limited -- falling back to espeak for the "
                        "next %.0fs, until the quota resets",
                        retry_after,
                    )
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
    import app  # noqa: F401  -- triggers load_dotenv() + logging setup

    c = OrpheusConfig()
    log.info("model=%s", c.model)
    log.info("voice=%s  format=%s", c.voice, c.response_format)
    log.info("url=%s%s", c.base_url, c.path)
    log.info("key length=%d", len(c.api_key))

    with OrpheusTTS(c) as tts:
        s = tts.synthesize(text, out_path="audio/output/orpheus_test.wav")

    if s.ok:
        log.info("OK   %s bytes in %.0f ms", f"{len(s.audio):,}", s.latency_ms)
        log.info("     saved %s", s.wav_path)
    else:
        log.error("FAILED after %.0f ms", s.latency_ms or 0)
        log.error("  %s", s.error)
        log.error("  401 -> credentials; check LLM_API_KEY")
        log.error("  404 -> wrong model id or path; try TTS_PATH variants")
        log.error("  400 -> parameter wrong; body names it (usually TTS_VOICE)")
        log.error("  429 -> rate limited")


if __name__ == "__main__":
    import sys

    _self_test(" ".join(sys.argv[1:]) or "Revenue in March was 128 thousand dollars.")