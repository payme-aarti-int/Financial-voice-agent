"""Microphone capture.

STILL BATCH -- deliberate. This is the baseline we measure, not what we ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.config import AUDIO


class Recorder:
    def __init__(self, sample_rate: int | None = None, channels: int | None = None):
        self.sample_rate = sample_rate or AUDIO.sample_rate
        self.channels = channels or AUDIO.channels

    def record(self, duration: float) -> np.ndarray:
        """Block for `duration` seconds and return mono float32 audio.

        Note the flaw this API forces on you: the caller must guess how long the
        user will speak. Eliminating that guess is what endpointing is for.
        """
        frames = int(duration * self.sample_rate)
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=AUDIO.dtype,
            device=AUDIO.device,
        )
        sd.wait()
        # sounddevice returns (n, 1) for mono; Whisper wants (n,). Passing the
        # 2-D array does not raise -- it silently misbehaves. Normalise here.
        return np.squeeze(audio)

    @staticmethod
    def save(audio: np.ndarray, path: str | Path, sample_rate: int | None = None) -> None:
        """Debug helper. Not in the hot path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), audio, sample_rate or AUDIO.sample_rate)

    @staticmethod
    def list_devices() -> str:
        return str(sd.query_devices())


def peak_level(audio: np.ndarray) -> float:
    """Loudest sample, 0.0-1.0.

    Near-zero means you recorded silence: wrong device, muted mic, or
    permissions. Check this before blaming the ASR for an empty transcript.
    """
    return float(np.max(np.abs(audio))) if audio.size else 0.0
