"""Speech-to-text — Groq-hosted Whisper, local faster-whisper, and local
NVIDIA Parakeet, all behind one BaseSTT interface chosen by STT_PROVIDER."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000  # Whisper's and Parakeet's native rate
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    audio_duration_s: float
    no_speech_prob: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def is_suspect(self) -> bool:
        if self.language_probability < 0.5:
            return True
        if self.no_speech_prob is not None and self.no_speech_prob > 0.6:
            return True
        return False


def _empty_transcript(duration_s: float) -> Transcript:
    return Transcript(
        text="",
        language="en",
        language_probability=0.0,
        audio_duration_s=duration_s,
        no_speech_prob=1.0,
    )


def _load_audio(audio: np.ndarray | str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    """Accept a mono float32 array or a path; always return (array, rate)."""
    if isinstance(audio, (str, Path)):
        samples, rate = sf.read(str(audio), dtype="float32")
    else:
        samples, rate = np.asarray(audio, dtype=np.float32), sample_rate
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, rate


def _resample(audio: np.ndarray, rate: int, target: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Linear resample -- good enough for speech going into an ASR model
    that is robust to mild aliasing; avoids pulling in librosa/torchaudio."""
    if rate == target or len(audio) == 0:
        return audio
    n_out = int(round(len(audio) * target / rate))
    return np.interp(
        np.linspace(0.0, len(audio) - 1, n_out), np.arange(len(audio)), audio
    ).astype(np.float32)


class BaseSTT(ABC):
    """Every STT provider must implement this. `transcribe` must never
    raise: a failed transcription returns an empty Transcript (so the
    pipeline asks the user to repeat) instead of crashing the turn."""

    _init_error: str | None = None

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript: ...

    @property
    def available(self) -> bool:
        return self._init_error is None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def warm_up(self) -> None:
        pass

    def close(self) -> None:
        pass


class LocalWhisperSTT(BaseSTT):
    """Local faster-whisper. Reads STT_MODEL_SIZE, STT_DEVICE,
    STT_COMPUTE_TYPE (via app.config.STT). Offline, no quota -- but on CPU
    it is several times slower than Groq's hosted Whisper."""

    def __init__(self, model_size=None, device=None, compute_type=None):
        from app.config import STT
        self._stt_config = STT
        self.model = None
        self._init_error: str | None = None
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                model_size or STT.model_size,
                device=device or STT.device,
                compute_type=compute_type or STT.compute_type,
            )
        except ImportError:
            self._init_error = "faster-whisper not installed (pip install faster-whisper)"
        except Exception as exc:  # noqa: BLE001
            self._init_error = (
                f"faster-whisper failed to load STT_MODEL_SIZE={STT.model_size} "
                f"on STT_DEVICE={STT.device}: {exc}"
            )
        if self._init_error:
            log.warning("LocalWhisperSTT unavailable (%s)", self._init_error)

    def warm_up(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio, sample_rate: int = 16000) -> Transcript:
        from app.config import STT

        if isinstance(audio, (str, Path)):
            source = str(audio)
            duration_s = sf.info(source).duration
        else:
            source = _resample(np.asarray(audio, dtype=np.float32), sample_rate)
            duration_s = len(audio) / sample_rate

        if self.model is None:
            return _empty_transcript(duration_s)

        try:
            from faster_whisper.vad import VadOptions

            segments, info = self.model.transcribe(
                source,
                language=STT.language,
                beam_size=STT.beam_size,
                vad_filter=STT.vad_filter,
                vad_parameters=VadOptions(min_silence_duration_ms=STT.vad_min_silence_ms),
            )
            collected = list(segments)
        except Exception:
            log.warning("faster-whisper transcription failed", exc_info=True)
            return _empty_transcript(duration_s)

        text = " ".join(s.text.strip() for s in collected).strip()
        no_speech = float(np.mean([s.no_speech_prob for s in collected])) if collected else None

        return Transcript(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            audio_duration_s=info.duration,
            no_speech_prob=no_speech,
        )


class GroqSTT(BaseSTT):
    """Groq-hosted whisper-large-v3-turbo — faster than local CPU Whisper.
    Reads LLM_API_KEY, STT_MODEL."""

    def __init__(self):
        import httpx
        self._api_key = os.getenv("LLM_API_KEY", "")
        self._model = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        self._client = httpx.Client(timeout=30)
        self._init_error: str | None = None
        if not self._api_key:
            self._init_error = "LLM_API_KEY is empty -- check .env"
            log.warning("GroqSTT: %s -- transcriptions will come back empty", self._init_error)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        from app.config import STT

        audio, sample_rate = _load_audio(audio, sample_rate)
        duration_s = len(audio) / sample_rate

        if STT.vad_filter:
            try:
                from app.audio.vad import speech_bounds
                bounds = speech_bounds(audio, sample_rate, min_silence_ms=STT.vad_min_silence_ms)
            except Exception:
                log.warning("VAD check failed -- sending audio to Groq unfiltered", exc_info=True)
                bounds = (0, len(audio))

            if bounds is None:
                # No speech at all: don't spend a network round trip on it.
                return Transcript(
                    text="",
                    language="en",
                    language_probability=0.0,
                    audio_duration_s=duration_s,
                    no_speech_prob=1.0,
                )
            start, end = bounds
            audio = audio[start:end]

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)

        try:
            response = self._client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={
                    "model": self._model,
                    "language": "en",
                    "response_format": "json",
                },
            )
            response.raise_for_status()
            text = response.json().get("text", "").strip()

            return Transcript(
                text=text,
                language="en",
                language_probability=1.0,
                audio_duration_s=duration_s,
                no_speech_prob=0.0 if text else 1.0,
            )

        except Exception as exc:
            log.warning("GroqSTT transcription failed: %s", exc)
            return Transcript(
                text="",
                language="en",
                language_probability=0.0,
                audio_duration_s=duration_s,
                no_speech_prob=1.0,
            )

    def warm_up(self) -> None:
        pass

    def close(self) -> None:
        self._client.close()


class ParakeetSTT(BaseSTT):
    """Local NVIDIA Parakeet TDT 0.6B v2 via NeMo -- fast, accurate English
    ASR, fully offline. Reads PARAKEET_MODEL_PATH: either a local .nemo
    file or a HuggingFace id (default nvidia/parakeet-tdt-0.6b-v2, fetched
    on first use).

    NeMo is a heavy optional dependency (pip install "nemo_toolkit[asr]").
    If it is missing, this engine logs a warning once and returns empty
    transcripts rather than crashing the pipeline.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("PARAKEET_MODEL_PATH", DEFAULT_PARAKEET_MODEL)
        self.model = None
        self._init_error: str | None = None
        try:
            from nemo.collections.asr.models import ASRModel

            if self.model_path.endswith(".nemo") or Path(self.model_path).exists():
                self.model = ASRModel.restore_from(self.model_path)
            else:
                self.model = ASRModel.from_pretrained(self.model_path)
            self.model.eval()
        except ImportError:
            self._init_error = (
                "nemo_toolkit not installed (pip install \"nemo_toolkit[asr]\") -- "
                "set STT_PROVIDER to groq or whisper, or install it"
            )
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Parakeet failed to load PARAKEET_MODEL_PATH={self.model_path}: {exc}"
        if self._init_error:
            log.warning("ParakeetSTT unavailable (%s) -- transcriptions will come back empty", self._init_error)

    def warm_up(self) -> None:
        if self.model is not None:
            self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        from app.config import STT

        audio, sample_rate = _load_audio(audio, sample_rate)
        duration_s = len(audio) / sample_rate
        if self.model is None:
            return _empty_transcript(duration_s)

        if STT.vad_filter:
            try:
                from app.audio.vad import speech_bounds
                bounds = speech_bounds(audio, sample_rate, min_silence_ms=STT.vad_min_silence_ms)
            except Exception:
                log.warning("VAD check failed -- sending audio to Parakeet unfiltered", exc_info=True)
                bounds = (0, len(audio))
            if bounds is None:
                return _empty_transcript(duration_s)
            start, end = bounds
            audio = audio[start:end]

        audio = _resample(audio, sample_rate)

        # NeMo's transcribe() API is most stable with file paths across
        # versions, so hand it a temp WAV rather than a raw array.
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio, TARGET_SAMPLE_RATE)
                outputs = self.model.transcribe([tmp.name], batch_size=1, verbose=False)
        except Exception:
            log.warning("Parakeet transcription failed", exc_info=True)
            return _empty_transcript(duration_s)

        # Newer NeMo returns Hypothesis objects; older returns strings
        # (and some versions wrap results in a (best, all) tuple).
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        first = outputs[0] if outputs else ""
        text = (getattr(first, "text", None) or (first if isinstance(first, str) else "")).strip()

        return Transcript(
            text=text,
            language="en",
            language_probability=1.0,
            audio_duration_s=duration_s,
            no_speech_prob=0.0 if text else 1.0,
        )

    def close(self) -> None:
        self.model = None


STT_PROVIDERS: dict[str, type[BaseSTT]] = {
    "groq": GroqSTT,
    "parakeet": ParakeetSTT,
    "whisper": LocalWhisperSTT,
}


def get_stt_client() -> BaseSTT:
    """Build the STT engine named by STT_PROVIDER in .env (default: groq).
    This is the only place STT_PROVIDER is read."""
    provider = os.getenv("STT_PROVIDER", "groq").strip().lower()
    cls = STT_PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown STT_PROVIDER: {provider}. Valid: " + ", ".join(STT_PROVIDERS)
            + " -- set STT_PROVIDER in .env"
        )
    log.info("STT_PROVIDER=%s -> %s", provider, cls.__name__)
    return cls()


# Backward compatibility -- the old local-whisper class name still works.
WhisperSTT = LocalWhisperSTT
