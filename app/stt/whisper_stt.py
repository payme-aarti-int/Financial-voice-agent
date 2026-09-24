"""Speech-to-text via faster-whisper (CTranslate2 backend)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions

from app.config import STT


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


class WhisperSTT:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.model = WhisperModel(
            model_size or STT.model_size,
            device=device or STT.device,
            compute_type=compute_type or STT.compute_type,
        )

    def warm_up(self) -> None:
        """Force lazy init using one second of silence.

        The first transcribe() is far slower: lazy weight loading, allocation,
        kernel selection. In a service you pay that at startup, or your first
        real user gets the worst latency of the day.
        """
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray | str | Path) -> Transcript:
        source = str(audio) if isinstance(audio, (str, Path)) else audio

        segments, info = self.model.transcribe(
            source,
            language=STT.language,
            beam_size=STT.beam_size,
            vad_filter=STT.vad_filter,
            vad_parameters=VadOptions(min_silence_duration_ms=STT.vad_min_silence_ms),
        )

        # transcribe() returns a GENERATOR: nothing computes until consumed.
        # Time the list() call, not the transcribe() call.
        collected = list(segments)
        text = " ".join(s.text.strip() for s in collected).strip()
        no_speech = max((s.no_speech_prob for s in collected), default=None)

        return Transcript(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            audio_duration_s=info.duration,
            no_speech_prob=no_speech,
        )
