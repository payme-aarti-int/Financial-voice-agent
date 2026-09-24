"""Hands-free microphone listening.

Opens the mic once and yields one array per utterance, using VAD to decide
when speech starts and stops -- no button press, no --duration guess.
"""

from __future__ import annotations

import logging
import queue
from typing import Iterator

import numpy as np
import sounddevice as sd

from app.audio.utterance import UtteranceDetector, make_silero_prober
from app.config import AUDIO, LISTEN, ListenConfig

log = logging.getLogger(__name__)


class AutoListener:
    def __init__(self, config: ListenConfig | None = None):
        self.config = config or LISTEN
        self.detector = UtteranceDetector(
            make_silero_prober(), config=self.config, sample_rate=AUDIO.sample_rate
        )
        self._chunk_samples = max(1, int(AUDIO.sample_rate * self.config.chunk_s))

    def utterances(self) -> Iterator[np.ndarray]:
        """Open the mic and yield complete utterances until the caller stops
        iterating (e.g. on Ctrl+C / break).
        """
        chunks: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time_info, status) -> None:
            if status:
                log.warning("mic stream status: %s", status)
            # sounddevice hands back (frames, channels); STT/VAD both want 1D.
            chunks.put(np.squeeze(indata).astype(np.float32, copy=True))

        with sd.InputStream(
            samplerate=AUDIO.sample_rate,
            channels=AUDIO.channels,
            dtype=AUDIO.dtype,
            device=AUDIO.device,
            blocksize=self._chunk_samples,
            callback=_callback,
        ):
            while True:
                chunk = chunks.get()
                utterance = self.detector.feed(chunk)
                if utterance is not None:
                    yield utterance
