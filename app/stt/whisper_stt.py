"""Speech-to-text — local faster-whisper and Groq-hosted Whisper."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

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
        if self.language_probability < 0.5:
            return True
        if self.no_speech_prob is not None and self.no_speech_prob > 0.6:
            return True
        return False


class WhisperSTT:
    """Local faster-whisper — kept as fallback."""

    def __init__(self, model_size=None, device=None, compute_type=None):
        from faster_whisper import WhisperModel
        from app.config import STT
        self._stt_config = STT
        self.model = WhisperModel(
            model_size or STT.model_size,
            device=device or STT.device,
            compute_type=compute_type or STT.compute_type,
        )

    def warm_up(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio) -> Transcript:
        from faster_whisper.vad import VadOptions
        from app.config import STT

        source = str(audio) if isinstance(audio, (str, Path)) else audio
        segments, info = self.model.transcribe(
            source,
            language=STT.language,
            beam_size=STT.beam_size,
            vad_filter=STT.vad_filter,
            vad_parameters=VadOptions(min_silence_duration_ms=STT.vad_min_silence_ms),
        )
        collected = list(segments)
        text = " ".join(s.text.strip() for s in collected).strip()
        no_speech = float(np.mean([s.no_speech_prob for s in collected])) if collected else None

        return Transcript(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            audio_duration_s=info.duration,
            no_speech_prob=no_speech,
        )


def _vad_speech_bounds(audio: np.ndarray, sample_rate: int) -> tuple[int, int] | None:
    """Locate where speech starts and ends using the same Silero VAD
    faster-whisper bundles (already used for WhisperSTT's vad_filter and
    the --listen endpointer in app/audio/utterance.py). Returns None if no
    speech is found at all, so a caller can skip a wasted API round trip
    on pure silence/noise instead of sending it off to be transcribed.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from app.config import STT

    options = VadOptions(min_silence_duration_ms=STT.vad_min_silence_ms)
    timestamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    if not timestamps:
        return None

    # A little padding either side so a fast VAD onset/offset doesn't clip
    # the first or last word.
    pad = int(0.2 * sample_rate)
    start = max(0, timestamps[0]["start"] - pad)
    end = min(len(audio), timestamps[-1]["end"] + pad)
    return start, end


class GroqSTT:
    """Groq-hosted whisper-large-v3-turbo — faster than local CPU Whisper."""

    def __init__(self):
        import os
        import httpx
        self._api_key = os.getenv("LLM_API_KEY", "")
        self._model = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        self._client = httpx.Client(timeout=30)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        from app.config import STT

        duration_s = len(audio) / sample_rate

        if STT.vad_filter:
            try:
                bounds = _vad_speech_bounds(audio, sample_rate)
            except Exception:
                log.warning("VAD check failed -- sending audio to Groq unfiltered", exc_info=True)
                bounds = (0, len(audio))

            if bounds is None:
                # No speech at all: don't spend a network round trip on it.
                return Transcript(
                    text="",
                    language="en",
                    language_probability=0.0,
                    audio_duration_s=duration_s,
                    no_speech_prob=1.0,
                )
            start, end = bounds
            audio = audio[start:end]

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
                audio_duration_s=duration_s,
                no_speech_prob=0.0 if text else 1.0,
            )

        except Exception as exc:
            return Transcript(
                text="",
                language="en",
                language_probability=0.0,
                audio_duration_s=duration_s,
                no_speech_prob=1.0,
            )

    def warm_up(self) -> None:
        pass

    def close(self) -> None:
        self._client.close()
