"""Hands-free microphone listening, with barge-in support.

Opens the mic once for the whole session. A background thread continuously
feeds incoming audio through VAD-based endpointing, so:
  - the main loop can block for the next complete utterance via
    get_utterance(), and
  - is_speech_active flips true the instant the user starts talking --
    fast enough to interrupt TTS playback, not just queue their words for
    after it finishes.

Caveat: is_speech_active reads the same microphone that's playing back the
agent's own voice. Without acoustic echo cancellation, speaker bleed into
the mic can occasionally look like the user interrupting. Headphones avoid
this; LISTEN_BARGE_IN_ENABLED=false disables barge-in entirely if it
misbehaves in an echo-prone room.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Iterator

import numpy as np
import sounddevice as sd

from app.audio.utterance import UtteranceDetector, make_silero_prober
from app.config import AUDIO, LISTEN, ListenConfig

log = logging.getLogger(__name__)

BARGE_IN_ENABLED = os.getenv("LISTEN_BARGE_IN_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "",
)


class AutoListener:
    def __init__(self, config: ListenConfig | None = None):
        self.config = config or LISTEN
        self.detector = UtteranceDetector(
            make_silero_prober(), config=self.config, sample_rate=AUDIO.sample_rate
        )
        self._chunk_samples = max(1, int(AUDIO.sample_rate * self.config.chunk_s))
        self._chunks: queue.Queue[np.ndarray] = queue.Queue()
        self._utterances: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def is_speech_active(self) -> bool:
        """True the instant speech onset is confirmed.

        Cheap (just reads a flag the background thread maintains) --
        meant to be polled frequently while TTS is playing.
        """
        if not BARGE_IN_ENABLED:
            return False
        return self.detector.recording

    def __enter__(self) -> "AutoListener":
        def _callback(indata, frames, time_info, status) -> None:
            if status:
                log.warning("mic stream status: %s", status)
            self._chunks.put(np.squeeze(indata).astype(np.float32, copy=True))

        self._stream = sd.InputStream(
            samplerate=AUDIO.sample_rate,
            channels=AUDIO.channels,
            dtype=AUDIO.dtype,
            device=AUDIO.device,
            blocksize=self._chunk_samples,
            callback=_callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            utterance = self.detector.feed(chunk)
            if utterance is not None:
                self._utterances.put(utterance)

    def get_utterance(self) -> np.ndarray:
        """Block until the next complete utterance is ready."""
        return self._utterances.get()

    def utterances(self) -> Iterator[np.ndarray]:
        """Convenience iterator form of get_utterance(), for simple loops."""
        while True:
            yield self.get_utterance()

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
