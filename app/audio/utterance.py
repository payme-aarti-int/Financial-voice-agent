"""Endpointing: turn a stream of audio chunks into discrete utterances.

Pure logic, no microphone I/O -- feed it audio chunks one at a time, get
back a complete utterance array once the speaker pauses. This is what
makes --listen hands-free: no button, no fixed --duration guess, just
"the model decides when you started and stopped talking".

Testable without real audio: `is_speech` is injected, so tests can fake
"speech" as "chunk is louder than X" instead of running the real VAD model.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator

import numpy as np

from app.config import LISTEN, ListenConfig

SpeechProber = Callable[[np.ndarray], bool]


class UtteranceDetector:
    def __init__(
        self,
        is_speech: SpeechProber,
        config: ListenConfig | None = None,
        sample_rate: int | None = None,
    ):
        self._is_speech = is_speech
        self.config = config or LISTEN
        self.sample_rate = sample_rate or 16_000
        self._reset()

    def _reset(self) -> None:
        self._pre_roll: list[np.ndarray] = []
        self._pre_roll_samples = 0
        self._utterance: list[np.ndarray] = []
        self._started_at: float | None = None
        self._last_speech_at: float | None = None

    @property
    def recording(self) -> bool:
        return self._started_at is not None

    def feed(self, chunk: np.ndarray, now: float | None = None) -> np.ndarray | None:
        """Feed one chunk of mono audio. Returns a complete utterance once
        the speaker pauses (or the max-duration safety cap trips), else None.
        """
        now = time.monotonic() if now is None else now
        cfg = self.config

        if not self.recording:
            self._pre_roll.append(chunk)
            self._pre_roll_samples += len(chunk)
            max_pre_roll = int(cfg.pre_roll_s * self.sample_rate)
            while self._pre_roll_samples > max_pre_roll and len(self._pre_roll) > 1:
                dropped = self._pre_roll.pop(0)
                self._pre_roll_samples -= len(dropped)

            window = _concat(self._pre_roll)
            if len(window) >= int(cfg.chunk_s * self.sample_rate) and self._is_speech(window):
                self._utterance = list(self._pre_roll)
                self._started_at = now
                self._last_speech_at = now
            return None

        self._utterance.append(chunk)
        elapsed = now - self._started_at
        if elapsed >= cfg.max_utterance_s:
            return self._finish()

        window_samples = int(cfg.check_window_s * self.sample_rate)
        tail = _concat(self._utterance)[-window_samples:]
        if self._is_speech(tail):
            self._last_speech_at = now
        elif now - self._last_speech_at >= cfg.min_silence_s:
            return self._finish()

        return None

    def _finish(self) -> np.ndarray | None:
        audio = _concat(self._utterance)
        self._reset()
        if len(audio) / self.sample_rate < self.config.min_utterance_s:
            return None
        return audio

    def process(self, chunks: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Convenience for feeding a whole iterable at once (tests, or a
        pre-recorded file played through the detector).
        """
        for chunk in chunks:
            utterance = self.feed(chunk)
            if utterance is not None:
                yield utterance


def _concat(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def make_silero_prober(min_speech_duration_ms: int = 100) -> SpeechProber:
    """Wrap faster-whisper's bundled Silero VAD as a per-window speech check.

    Same model already used to filter STT input -- no extra dependency, and
    it stays consistent with app/config.py's STTConfig.vad_* settings.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_speech_duration_ms=min_speech_duration_ms)

    def _is_speech(window: np.ndarray) -> bool:
        if window.size == 0:
            return False
        return bool(get_speech_timestamps(window, options))

    return _is_speech
