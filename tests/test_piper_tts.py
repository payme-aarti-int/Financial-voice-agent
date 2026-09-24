"""PiperEngine: local, offline, no-rate-limit TTS fallback.

The "missing model" path needs no download and always runs. The real
synthesis path is skipped if the voice model hasn't been downloaded
(python -m piper.download_voices en_US-lessac-medium --download-dir
models/piper) -- CI environments without it still pass.
"""

from pathlib import Path

import pytest

from app.tts.engine import DEFAULT_PIPER_MODEL, PiperEngine

MODEL_PRESENT = Path(DEFAULT_PIPER_MODEL).exists()


def test_unavailable_with_a_missing_model_path(tmp_path):
    engine = PiperEngine(model_path=tmp_path / "no-such-model.onnx")
    assert engine.available is False

    speech = engine.synthesize("hello", tmp_path / "out.wav")
    assert not speech.ok
    assert "not found" in speech.error


def test_empty_text_is_rejected(tmp_path):
    engine = PiperEngine(model_path=tmp_path / "no-such-model.onnx")
    speech = engine.synthesize("   ", tmp_path / "out.wav")
    assert not speech.ok
    assert speech.error == "empty text"


@pytest.mark.skipif(not MODEL_PRESENT, reason="Piper voice model not downloaded")
def test_real_synthesis_produces_playable_audio(tmp_path):
    import soundfile as sf

    engine = PiperEngine()
    assert engine.available

    out_path = tmp_path / "clip.wav"
    speech = engine.synthesize("This is a test of the Piper voice.", out_path)

    assert speech.ok
    assert out_path.exists()
    info = sf.info(str(out_path))
    assert info.duration > 0
