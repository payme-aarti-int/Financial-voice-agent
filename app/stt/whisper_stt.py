"""Speech-to-text via Groq's hosted Whisper API."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

import httpx
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    audio_duration_s: float
    no_speech_prob: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def is_suspect(self) -> bool:
        """Treat this transcript as untrustworthy.

        Matters more for a financial agent: a confidently wrong number spoken
        aloud has no scrollback and no citation. When this trips, ask the user
        to repeat rather than answering.
        """
        if self.language_probability < 0.5:
            return True
        if self.no_speech_prob is not None and self.no_speech_prob > 0.6:
            return True
        return False


class GroqSTT:
    """Groq-hosted Whisper STT — replaces local faster-whisper."""

    def __init__(self):
        self._api_key = os.getenv("LLM_API_KEY", "")
        self._model = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        self._client = httpx.Client(timeout=30)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)

        try:
            response = self._client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={
                    "model": self._model,
                    "language": "en",
                    "response_format": "json",
                },
            )
            response.raise_for_status()
            text = response.json().get("text", "").strip()

            return Transcript(
                text=text,
                language="en",
                language_probability=1.0,
                audio_duration_s=len(audio) / sample_rate,
                no_speech_prob=0.0 if text else 1.0,
            )

        except Exception as exc:  # noqa: BLE001
            log.error("Groq transcription failed: %s", exc)
            return Transcript(
                text="",
                language="en",
                language_probability=0.0,
                audio_duration_s=len(audio) / sample_rate,
                no_speech_prob=1.0,
            )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GroqSTT":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
