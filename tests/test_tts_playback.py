"""play()'s interrupt polling, tested without a real audio device by
faking sounddevice's play/get_stream/stop.
"""

import numpy as np
import soundfile as sf
import sounddevice as sd

from app.tts.engine import play


class FakeStream:
    def __init__(self):
        self.active = True


def test_play_stops_immediately_when_interrupted(tmp_path, monkeypatch):
    wav_path = tmp_path / "clip.wav"
    sf.write(str(wav_path), np.zeros(16_000, dtype="float32"), 16_000)  # 1s

    fake_stream = FakeStream()
    monkeypatch.setattr(sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(sd, "get_stream", lambda: fake_stream)
    stopped = {"called": False}

    def fake_stop():
        stopped["called"] = True
        fake_stream.active = False

    monkeypatch.setattr(sd, "stop", fake_stop)

    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        return calls["n"] >= 2  # interrupt on the second poll

    played, interrupted, error = play(str(wav_path), interrupt=interrupt, poll_s=0.001)

    assert interrupted is True
    assert played is False
    assert error is None
    assert stopped["called"] is True


def test_play_runs_to_completion_without_interrupt(tmp_path, monkeypatch):
    wav_path = tmp_path / "clip.wav"
    sf.write(str(wav_path), np.zeros(1_600, dtype="float32"), 16_000)  # 0.1s

    fake_stream = FakeStream()
    monkeypatch.setattr(sd, "play", lambda *a, **k: None)

    polls = {"n": 0}

    def get_stream():
        polls["n"] += 1
        if polls["n"] >= 2:
            fake_stream.active = False
        return fake_stream

    monkeypatch.setattr(sd, "get_stream", get_stream)

    played, interrupted, error = play(str(wav_path), interrupt=None, poll_s=0.001)

    assert played is True
    assert interrupted is False
    assert error is None


def test_play_never_interrupted_without_a_predicate(tmp_path, monkeypatch):
    wav_path = tmp_path / "clip.wav"
    sf.write(str(wav_path), np.zeros(800, dtype="float32"), 16_000)

    fake_stream = FakeStream()
    monkeypatch.setattr(sd, "play", lambda *a, **k: None)
    fake_stream.active = False  # finishes on the first check
    monkeypatch.setattr(sd, "get_stream", lambda: fake_stream)

    played, interrupted, error = play(str(wav_path), poll_s=0.001)

    assert played is True
    assert interrupted is False
    assert error is None
