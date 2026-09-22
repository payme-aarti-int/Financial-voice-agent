import numpy as np
import pytest
import soundfile as sf

from app.audio.recorder import Recorder, peak_level


def test_peak_level_silence_is_zero():
    audio = np.zeros(16_000, dtype=np.float32)
    assert peak_level(audio) == 0.0


def test_peak_level_reports_loudest_sample():
    audio = np.array([0.1, -0.8, 0.3], dtype=np.float32)
    assert peak_level(audio) == pytest.approx(0.8)


def test_peak_level_empty_array_is_zero():
    assert peak_level(np.array([], dtype=np.float32)) == 0.0


def test_save_and_reload_round_trips_audio(tmp_path):
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    out_path = tmp_path / "nested" / "clip.wav"

    Recorder.save(audio, out_path, sample_rate=16_000)

    assert out_path.exists()
    reloaded, rate = sf.read(str(out_path), dtype="float32")
    assert rate == 16_000
    np.testing.assert_allclose(reloaded, audio, atol=1e-4)


def test_list_devices_returns_a_string():
    assert isinstance(Recorder.list_devices(), str)
