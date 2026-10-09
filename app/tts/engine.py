from __future__ import annotations
 
import io
import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
 
import numpy as np
import soundfile as sf

from app.config import AUDIO

log = logging.getLogger(__name__)

DEFAULT_PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx"
DEFAULT_OUTPUT_PATH = Path("audio/output/reply.wav")
 
 
@dataclass
class Speech:
 
    text: str
    wav_path: Path | None = None
    duration_s: float | None = None
    played: bool = False
    interrupted: bool = False
    error: str | None = None
 
    @property
    def ok(self) -> bool:
        return self.wav_path is not None and self.error is None


class BaseTTSClient(ABC):
    """Every TTS provider must implement this. `synthesize` must never
    raise: a failed Speech (error set) lets TTSService fall through to the
    next engine, whereas an exception means the user hears nothing and
    cannot tell whether the agent broke.
    """

    _init_error: str | None = None

    @abstractmethod
    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    def available(self) -> bool:
        return self._init_error is None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def __enter__(self) -> "BaseTTSClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
 
 
class EspeakTTS(BaseTTSClient):
 
    def __init__(self, words_per_minute: int = 165, voice: str = "en-us"):
        self.words_per_minute = words_per_minute
        self.voice = voice
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if self.binary is None:
            self._init_error = "espeak-ng not installed (sudo apt install espeak-ng)"
 
    @property
    def available(self) -> bool:
        return self.binary is not None
 
    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        if not text.strip():
            return Speech(text=text, error="empty text")
        if not self.available:
            return Speech(
                text=text,
                error="espeak-ng not installed (sudo apt install espeak-ng)",
            )
 
        path = Path(out_path or DEFAULT_OUTPUT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        try:
            subprocess.run(
                [
                    self.binary,
                    "-v", self.voice,
                    "-s", str(self.words_per_minute),
                    "-w", str(path),
                    text,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            return Speech(text=text, error=f"espeak failed: {exc.stderr.decode()}")
        except subprocess.TimeoutExpired:
            return Speech(text=text, error="espeak timed out")
 
        info = sf.info(str(path))
        return Speech(text=text, wav_path=path, duration_s=info.duration)

    def close(self) -> None:
        pass
 
 
class GroqOrpheusTTS(BaseTTSClient):
    """Groq-hosted Orpheus (natural voice). Reads TTS_MODEL, TTS_VOICE,
    TTS_FORMAT and LLM_API_KEY from .env via OrpheusConfig; the HTTP
    transport and 429 cooldown live in app/orpheus_tts.py, which
    app/api.py's /api/speak streams from directly.
    """

    def __init__(self, config=None):
        from app.orpheus_tts import OrpheusConfig, OrpheusTTS

        self._config = config or OrpheusConfig()
        self._client: OrpheusTTS | None = None
        self._init_error: str | None = None
        if not self._config.api_key:
            self._init_error = "LLM_API_KEY is empty -- check .env"
            return
        try:
            self._client = OrpheusTTS(self._config)
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        if not self.available:
            return Speech(text=text, error=self._init_error or "Orpheus unavailable")

        result = self._client.synthesize(text, out_path=out_path or DEFAULT_OUTPUT_PATH)
        if not result.ok:
            return Speech(text=text, error=result.error)

        info = sf.info(result.wav_path) if result.wav_path else None
        return Speech(
            text=text,
            wav_path=Path(result.wav_path) if result.wav_path else None,
            duration_s=info.duration if info else None,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class PiperTTS(BaseTTSClient):
    """Local, offline neural TTS (Piper: https://github.com/OHF-Voice/piper1-gpl,
    MIT-licensed voices). Runs entirely on-device via onnxruntime -- no
    network call, no API key, no rate limit, ever. Not as expressive as
    Orpheus, but a large step up from espeak's formant synthesis, and it
    never runs out of quota.

    Reads PIPER_MODEL_PATH (default models/piper/en_US-lessac-medium.onnx).
    With `auto_download=True` (what the factory uses when TTS_PROVIDER=piper)
    a missing voice is fetched once from HuggingFace, the same as:
        python -m piper.download_voices en_US-lessac-medium \\
            --download-dir models/piper
    """

    def __init__(self, model_path: str | Path | None = None, auto_download: bool = False):
        self._model_path = Path(model_path or os.getenv("PIPER_MODEL_PATH", DEFAULT_PIPER_MODEL))
        self._voice = None
        self._init_error: str | None = None

        if not self._model_path.exists() and auto_download:
            self._download_model()
        if not self._model_path.exists():
            self._init_error = (
                f"Piper model not found at {self._model_path} -- run: "
                "python -m piper.download_voices en_US-lessac-medium "
                "--download-dir models/piper"
            )
            return
        try:
            from piper import PiperVoice

            self._voice = PiperVoice.load(str(self._model_path))
        except ImportError:
            self._init_error = "piper-tts not installed (pip install piper-tts)"
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)

    def _download_model(self) -> None:
        voice = self._model_path.name.removesuffix(".onnx")
        log.info("Piper model missing -- downloading voice %s to %s", voice, self._model_path.parent)
        try:
            from piper.download_voices import download_voice

            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            download_voice(voice, self._model_path.parent)
        except ImportError:
            log.warning("Cannot download Piper voice: piper-tts not installed (pip install piper-tts)")
        except Exception:  # noqa: BLE001
            log.warning("Piper voice download failed for %s (PIPER_MODEL_PATH)", voice, exc_info=True)

    @property
    def available(self) -> bool:
        return self._voice is not None

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        if not text.strip():
            return Speech(text=text, error="empty text")
        if not self.available:
            return Speech(text=text, error=self._init_error or "Piper unavailable")

        path = Path(out_path or DEFAULT_OUTPUT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import wave

            with wave.open(str(path), "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)
        except Exception as exc:  # noqa: BLE001
            return Speech(text=text, error=f"Piper synthesis failed: {exc}")

        info = sf.info(str(path))
        return Speech(text=text, wav_path=path, duration_s=info.duration)

    def close(self) -> None:
        pass


class KokoroTTS(BaseTTSClient):
    """Local Kokoro-82M neural TTS (https://github.com/hexgrad/kokoro),
    24 kHz output, Python 3.11 compatible. Reads KOKORO_VOICE (default
    af_heart) and KOKORO_LANG (default "a" = American English).

    The `kokoro` package is optional (pip install kokoro soundfile) and is
    only imported here, so an environment without it still starts -- this
    engine just reports itself unavailable and TTSService falls through.
    """

    SAMPLE_RATE = 24_000

    def __init__(self, voice: str | None = None, lang_code: str | None = None):
        self.voice = voice or os.getenv("KOKORO_VOICE", "af_heart")
        self.lang_code = lang_code or os.getenv("KOKORO_LANG", "a")
        self._pipeline = None
        self._init_error: str | None = None
        try:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M")
        except ImportError:
            self._init_error = "kokoro not installed (pip install kokoro) -- set TTS_PROVIDER to another engine or install it"
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Kokoro failed to load (KOKORO_VOICE={self.voice}, KOKORO_LANG={self.lang_code}): {exc}"

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        if not text.strip():
            return Speech(text=text, error="empty text")
        if not self.available:
            return Speech(text=text, error=self._init_error or "Kokoro unavailable")

        path = Path(out_path or DEFAULT_OUTPUT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            chunks = [
                np.asarray(audio, dtype="float32")
                for _, _, audio in self._pipeline(text, voice=self.voice)
                if audio is not None
            ]
            if not chunks:
                return Speech(text=text, error="Kokoro produced no audio")
            sf.write(str(path), np.concatenate(chunks), self.SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            return Speech(text=text, error=f"Kokoro synthesis failed: {exc}")

        info = sf.info(str(path))
        return Speech(text=text, wav_path=path, duration_s=info.duration)

    def close(self) -> None:
        self._pipeline = None


# Backward compatibility -- older imports and tests still use these names.
OrpheusEngine = GroqOrpheusTTS
PiperEngine = PiperTTS


def _trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim the dead air Piper/Orpheus/espeak often pad a clip with, using
    the same Silero VAD used to trim STT input (app/audio/vad.py, shared
    with app/stt/whisper_stt.py) -- less silent lead-in before the agent's
    reply is audible, and playback ends right after the last word instead
    of running on into silence.

    Falls back to the untrimmed clip if VAD finds nothing or errors, so a
    VAD miss can never mean playing back silence instead of the reply.
    """
    try:
        from app.audio.vad import speech_bounds

        bounds = speech_bounds(np.squeeze(audio), sample_rate)
    except Exception:
        log.warning("TTS VAD trim failed -- playing untrimmed audio", exc_info=True)
        return audio
    if bounds is None:
        return audio
    start, end = bounds
    return audio[start:end]


def trim_wav_bytes(wav_bytes: bytes) -> bytes:
    """Same silence trim as `play()`, but for a WAV already in memory --
    e.g. app/api.py's /api/speak, which streams Orpheus's response
    straight to the browser and never touches disk or `play()`. Falls
    back to the original bytes if decoding/re-encoding fails.
    """
    try:
        audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        trimmed = _trim_silence(audio, sample_rate)
        out = io.BytesIO()
        sf.write(out, trimmed, sample_rate, format="WAV")
        return out.getvalue()
    except Exception:
        log.warning("Could not trim WAV bytes -- returning untrimmed audio", exc_info=True)
        return wav_bytes


def play(
    wav_path: str | Path,
    interrupt: Callable[[], bool] | None = None,
    poll_s: float = 0.05,
) -> tuple[bool, bool, str | None]:
    """Play a WAV file, watching `interrupt` (if given) so a real person
    talking over the agent can cut it off instead of waiting it out.

    Returns (played_to_completion, was_interrupted, error).
    """
    try:
        import sounddevice as sd

        from app.audio.recorder import TIMEOUT_SLACK_S
    except Exception as exc:  # noqa: BLE001 - OSError when PortAudio is absent
        return False, False, f"audio output unavailable: {exc}"

    try:
        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        audio = _trim_silence(audio, sample_rate)
        duration_s = len(audio) / sample_rate
        sd.play(np.squeeze(audio), sample_rate, device=AUDIO.device)

        deadline = time.monotonic() + duration_s + TIMEOUT_SLACK_S
        while True:
            if interrupt is not None and interrupt():
                sd.stop()
                return False, True, None

            stream = sd.get_stream()
            if stream is None or not stream.active:
                return True, False, None

            if time.monotonic() >= deadline:
                sd.stop()
                return False, False, (
                    f"playback did not finish within {duration_s + TIMEOUT_SLACK_S:.0f}s "
                    "-- the audio output device may be unresponsive or disconnected"
                )
            time.sleep(poll_s)
    except Exception as exc:  # noqa: BLE001
        return False, False, f"playback failed: {exc}"


class TTSService(BaseTTSClient):
    """Synthesise then play, reporting what actually happened.

    Tries a chain of engines in order and speaks with whichever one works
    first. By default (TTS_PROVIDER=groq):
      1. Orpheus (best quality, natural voice, Groq-hosted -- limited by
         an external daily token quota; TTS_MODEL/TTS_VOICE in .env).
      2. Piper (local, offline, free, no rate limit ever -- a solid
         neural voice, just less expressive than Orpheus).
      3. espeak-ng (robotic, offline, always available) as the last
         resort, in case Piper has no voice model downloaded either.
    A wrong-sounding voice beats no voice at all. `get_tts_client()` puts
    whichever provider .env names at the head of this chain.

    TTSService is itself a BaseTTSClient (composite), so callers can hold
    either a single engine or the whole fallback chain behind one type.
    """

    def __init__(
        self,
        engines: list[BaseTTSClient] | None = None,
        output_dir: str = "audio/output",
    ):
        self.engines = engines or _default_engines()
        self.output_dir = Path(output_dir)

    @property
    def available(self) -> bool:
        return any(engine.available for engine in self.engines)

    def synthesize(self, text: str, out_path: str | Path | None = None) -> Speech:
        """Synthesise with the first engine in the chain that succeeds."""
        speech = None
        path = Path(out_path or self.output_dir / "reply.wav")
        for i, engine in enumerate(self.engines):
            speech = engine.synthesize(text, path)
            if speech.ok:
                break

            is_last = i == len(self.engines) - 1
            if not is_last:
                # "rate limited" means the engine itself already logged
                # this once and is deliberately short-circuiting -- not a
                # new failure, so don't re-warn on every single turn.
                error = speech.error or ""
                log_fn = log.info if error.startswith("rate limited") else log.warning
                next_engine = type(self.engines[i + 1]).__name__
                log_fn(
                    "%s failed (%s), trying %s",
                    type(engine).__name__, speech.error, next_engine,
                )
        return speech if speech is not None else Speech(text=text, error="no TTS engines configured")

    def close(self) -> None:
        for engine in self.engines:
            engine.close()

    def speak(
        self,
        text: str,
        filename: str = "reply.wav",
        autoplay: bool = True,
        interrupt: Callable[[], bool] | None = None,
    ) -> Speech:
        """`interrupt`, if given, is polled while playing: return True from
        it (e.g. "the user started talking") and playback stops immediately
        instead of running to completion.
        """
        speech = self.synthesize(text, self.output_dir / filename)

        if not speech.ok or not autoplay:
            return speech

        played, interrupted, error = play(speech.wav_path, interrupt=interrupt)
        speech.played = played
        speech.interrupted = interrupted
        if error:
            # Not fatal: the WAV exists and the user can still play it manually.
            speech.error = error
        return speech


TTS_PROVIDERS: dict[str, type[BaseTTSClient]] = {
    "groq": GroqOrpheusTTS,
    "piper": PiperTTS,
    "kokoro": KokoroTTS,
}


def _with_local_fallbacks(primary: BaseTTSClient | None) -> list[BaseTTSClient]:
    """`primary` first, then the local engines that never hit a quota --
    Piper if a voice is downloaded, espeak-ng always -- skipping any that
    duplicate the primary. Groq is never added as a fallback: it costs
    quota, so it only speaks when .env explicitly asks for it."""
    engines: list[BaseTTSClient] = []
    if primary is not None:
        engines.append(primary)

    if not isinstance(primary, PiperTTS):
        piper = PiperTTS()
        if piper.available:
            engines.append(piper)
        else:
            log.warning("Piper TTS unavailable (%s)", piper.init_error)

    engines.append(EspeakTTS())
    return engines


def _default_engines() -> list[BaseTTSClient]:
    orpheus = GroqOrpheusTTS()
    if not orpheus.available:
        log.warning("Orpheus TTS unavailable (%s)", orpheus.init_error)
    return _with_local_fallbacks(orpheus if orpheus.available else None)


def _build_tts_provider(provider: str) -> BaseTTSClient:
    cls = TTS_PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown TTS_PROVIDER={provider!r} -- set TTS_PROVIDER in .env to one of: "
            + ", ".join(TTS_PROVIDERS)
        )
    if cls is PiperTTS:
        return PiperTTS(auto_download=True)
    return cls()


def get_tts_client() -> BaseTTSClient:
    """Build the TTS engine named by TTS_PROVIDER in .env (default: groq),
    wrapped in a TTSService so playback and the local fallback chain keep
    working whichever provider is chosen. This is the only place
    TTS_PROVIDER is read.

    If the chosen provider cannot start (missing package, no API key, no
    model) a clear warning names the problem and the service degrades to
    the local fallbacks instead of crashing.
    """
    provider = os.getenv("TTS_PROVIDER", "groq").strip().lower()
    primary = _build_tts_provider(provider)
    log.info("TTS_PROVIDER=%s -> %s", provider, type(primary).__name__)
    if not primary.available:
        log.warning(
            "TTS_PROVIDER=%s selected but %s is unavailable (%s) -- falling back to local engines",
            provider, type(primary).__name__, primary.init_error,
        )
        primary = None
    return TTSService(engines=_with_local_fallbacks(primary))
