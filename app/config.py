"""Single source of truth for pipeline settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000   # Whisper's native rate. Do not change.
    channels: int = 1           # Mono.
    dtype: str = "float32"      # What faster-whisper wants; avoids a conversion.

    # None lets PortAudio pick its own default. On some Linux setups that
    # default routes through a dmix/PipeWire ALSA plugin that never signals
    # completion, so sd.wait() blocks forever with no error. Set AUDIO_DEVICE
    # (e.g. "sysdefault") to point at a device that actually completes.
    device: str | None = os.getenv("AUDIO_DEVICE") or None


@dataclass(frozen=True)
class STTConfig:
    model_size: str = os.getenv("STT_MODEL_SIZE", "base.en")
    device: str = os.getenv("STT_DEVICE", "cpu")
    compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8")

    # Pinned language skips Whisper's detection pass: saves time and prevents
    # misdetection on short/noisy clips, which corrupts the whole transcript.
    language: str = "en"

    # beam_size=1 is greedy decoding. Beam search buys ~1 point of WER for a
    # large latency cost -- wrong trade for real-time.
    beam_size: int = 1


AUDIO = AudioConfig()
STT = STTConfig()
