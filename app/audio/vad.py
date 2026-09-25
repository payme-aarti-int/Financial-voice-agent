"""Shared Silero VAD helper (bundled with faster-whisper) for finding where
speech starts and ends in a clip of audio.

Used on both sides of a turn: trimming STT input before it's sent to Groq
(app/stt/whisper_stt.py), and trimming TTS output before playback
(app/tts/engine.py) -- so neither wastes time on dead air. Same model
already used by the --listen endpointer in app/audio/utterance.py.
"""

from __future__ import annotations

import numpy as np


def speech_bounds(
    audio: np.ndarray,
    sample_rate: int,
    min_silence_ms: int = 500,
    pad_s: float = 0.2,
) -> tuple[int, int] | None:
    """Return (start, end) sample indices bounding the detected speech,
    padded a little either side so a fast VAD onset/offset doesn't clip
    the first or last word. None if no speech is found at all.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=min_silence_ms)
    timestamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    if not timestamps:
        return None

    pad = int(pad_s * sample_rate)
    start = max(0, timestamps[0]["start"] - pad)
    end = min(len(audio), timestamps[-1]["end"] + pad)
    return start, end
