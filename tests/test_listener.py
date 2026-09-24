"""AutoListener's barge-in flag, tested without opening a real mic --
constructing it doesn't touch sounddevice at all, only __enter__ does.
"""

import app.audio.listener as listener_module
from app.audio.listener import AutoListener


def test_is_speech_active_reflects_detector_recording():
    listener = AutoListener()
    assert listener.is_speech_active is False

    listener.detector._started_at = 0.0  # simulate confirmed speech onset
    assert listener.is_speech_active is True


def test_is_speech_active_false_when_barge_in_disabled(monkeypatch):
    monkeypatch.setattr(listener_module, "BARGE_IN_ENABLED", False)

    listener = AutoListener()
    listener.detector._started_at = 0.0

    assert listener.is_speech_active is False
