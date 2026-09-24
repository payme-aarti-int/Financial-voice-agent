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
    model_size: str = os.getenv("STT_MODEL_SIZE", "large-v3-turbo")
    device: str = os.getenv("STT_DEVICE", "cpu")
    compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8")

    # Pinned language skips Whisper's detection pass: saves time and prevents
    # misdetection on short/noisy clips, which corrupts the whole transcript.
    language: str = "en"

    # beam_size=1 is greedy decoding. Beam search buys ~1 point of WER for a
    # large latency cost -- wrong trade for real-time.
    beam_size: int = 1

    # VAD (Silero, bundled with faster-whisper) drops non-speech before
    # decoding: less silence/noise for Whisper to hallucinate a transcript
    # from, and less audio to decode in the first place.
    vad_filter: bool = os.getenv("STT_VAD_FILTER", "true").lower() not in ("false", "0", "")

    # The library default (2000ms) waits a long time before deciding speech
    # has ended -- fine for transcribing a file, too slow for a turn-taking
    # voice agent. Shorter cuts latency at the cost of trimming pauses mid-
    # sentence more aggressively.
    vad_min_silence_ms: int = int(os.getenv("STT_VAD_MIN_SILENCE_MS", "500"))


@dataclass(frozen=True)
class ListenConfig:
    """Hands-free mic listening: VAD decides when an utterance starts and
    ends, so there is no button to press and no --duration to guess.
    """

    # Audio kept from before speech is detected, so the first word isn't
    # clipped while the VAD is still confirming onset.
    pre_roll_s: float = float(os.getenv("LISTEN_PRE_ROLL_S", "0.5"))

    # How much trailing audio the VAD looks at each time it re-checks
    # whether the speaker is still talking.
    check_window_s: float = float(os.getenv("LISTEN_CHECK_WINDOW_S", "1.0"))

    # Trailing silence required before an utterance is considered finished.
    min_silence_s: float = float(os.getenv("LISTEN_MIN_SILENCE_S", "0.6"))

    # Utterances shorter than this are discarded as noise, not sent to STT.
    min_utterance_s: float = float(os.getenv("LISTEN_MIN_UTTERANCE_S", "0.3"))

    # Hard cap so a VAD miss can't record forever.
    max_utterance_s: float = float(os.getenv("LISTEN_MAX_UTTERANCE_S", "30"))

    # Mic is read in chunks this long; also the callback's latency floor.
    chunk_s: float = float(os.getenv("LISTEN_CHUNK_S", "0.1"))


AUDIO = AudioConfig()
STT = STTConfig()
LISTEN = ListenConfig()
