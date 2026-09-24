"""UtteranceDetector is pure logic (no mic) -- fake "is speech" as "loud",
so these run instantly and don't depend on real VAD weights or hardware.
"""

import numpy as np

from app.audio.utterance import UtteranceDetector
from app.config import ListenConfig

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 160  # 10ms


def loud_is_speech(window: np.ndarray) -> bool:
    return window.size > 0 and float(np.max(np.abs(window))) > 0.1


def make_chunk(speech: bool) -> np.ndarray:
    value = 0.5 if speech else 0.0
    return np.full(CHUNK_SAMPLES, value, dtype=np.float32)


def make_detector(**overrides) -> UtteranceDetector:
    defaults = dict(
        pre_roll_s=0.05,
        check_window_s=0.1,
        min_silence_s=0.05,
        min_utterance_s=0.02,
        max_utterance_s=1.0,
        chunk_s=0.01,
    )
    defaults.update(overrides)
    config = ListenConfig(**defaults)
    return UtteranceDetector(loud_is_speech, config=config, sample_rate=SAMPLE_RATE)


def feed_chunks(detector: UtteranceDetector, speech: bool, count: int, now: float, step: float = 0.01):
    """Feed `count` chunks, returning (last_result, next_now)."""
    result = None
    for _ in range(count):
        result = detector.feed(make_chunk(speech), now=now)
        now += step
        if result is not None:
            break
    return result, now


def test_silence_only_never_yields():
    detector = make_detector()
    result, _ = feed_chunks(detector, speech=False, count=50, now=0.0)
    assert result is None
    assert not detector.recording


def test_speech_then_silence_yields_one_utterance():
    detector = make_detector()

    # pre-roll silence, then speech starts
    _, now = feed_chunks(detector, speech=False, count=3, now=0.0)
    result, now = feed_chunks(detector, speech=True, count=1, now=now)
    assert result is None
    assert detector.recording

    # keep talking
    result, now = feed_chunks(detector, speech=True, count=5, now=now)
    assert result is None

    # pause long enough to end the utterance
    utterance, now = feed_chunks(detector, speech=False, count=20, now=now)

    assert utterance is not None
    assert utterance.size > 0
    assert not detector.recording
    # the utterance should contain the pre-roll silence and the speech
    assert np.any(utterance > 0.1)


def test_short_blip_is_discarded_as_noise():
    detector = make_detector(min_utterance_s=1.0)  # nothing this short survives

    _, now = feed_chunks(detector, speech=True, count=1, now=0.0)
    assert detector.recording

    utterance, _ = feed_chunks(detector, speech=False, count=20, now=now)

    assert utterance is None
    assert not detector.recording


def test_max_utterance_cap_ends_recording_without_silence():
    detector = make_detector(max_utterance_s=0.05)

    _, now = feed_chunks(detector, speech=True, count=1, now=0.0)
    assert detector.recording

    # keep talking past the safety cap -- never goes silent
    utterance, _ = feed_chunks(detector, speech=True, count=50, now=now)

    assert utterance is not None
    assert not detector.recording


def test_detector_resets_for_a_second_utterance():
    detector = make_detector()

    _, now = feed_chunks(detector, speech=True, count=1, now=0.0)
    first, now = feed_chunks(detector, speech=False, count=20, now=now)
    assert first is not None
    assert not detector.recording

    # a second utterance should work the same way
    _, now = feed_chunks(detector, speech=True, count=1, now=now)
    assert detector.recording
    second, _ = feed_chunks(detector, speech=False, count=20, now=now)
    assert second is not None
