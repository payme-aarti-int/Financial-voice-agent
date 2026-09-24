from __future__ import annotations
 
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
 
import numpy as np
import soundfile as sf

from app.config import AUDIO

log = logging.getLogger(__name__)

DEFAULT_PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx"
 
 
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
 
 
class EspeakTTS:
 
    def __init__(self, words_per_minute: int = 165, voice: str = "en-us"):
        self.words_per_minute = words_per_minute
        self.voice = voice
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
 
    @property
    def available(self) -> bool:
        return self.binary is not None
 
    def synthesize(self, text: str, out_path: str | Path) -> Speech:
        if not text.strip():
            return Speech(text=text, error="empty text")
        if not self.available:
            return Speech(
                text=text,
                error="espeak-ng not installed (sudo apt install espeak-ng)",
            )
 
        path = Path(out_path)
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
 
 
class OrpheusEngine:
    """Wraps OrpheusTTS (Groq-hosted, natural voice) behind the same
    synthesize() interface as EspeakTTS, so TTSService doesn't need to
    know which engine is actually speaking.
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

    def synthesize(self, text: str, out_path: str | Path) -> Speech:
        if not self.available:
            return Speech(text=text, error=self._init_error or "Orpheus unavailable")

        result = self._client.synthesize(text, out_path=out_path)
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


class PiperEngine:
    """Local, offline neural TTS (Piper: https://github.com/OHF-Voice/piper1-gpl,
    MIT-licensed voices). Runs entirely on-device via onnxruntime -- no
    network call, no API key, no rate limit, ever. Not as expressive as
    Orpheus, but a large step up from espeak's formant synthesis, and it
    never runs out of quota.

    Get a voice with:
        python -m piper.download_voices en_US-lessac-medium \\
            --download-dir models/piper
    """

    def __init__(self, model_path: str | Path | None = None):
        self._model_path = Path(model_path or os.getenv("PIPER_MODEL_PATH", DEFAULT_PIPER_MODEL))
        self._voice = None
        self._init_error: str | None = None

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
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._voice is not None

    def synthesize(self, text: str, out_path: str | Path) -> Speech:
        if not text.strip():
            return Speech(text=text, error="empty text")
        if not self.available:
            return Speech(text=text, error=self._init_error or "Piper unavailable")

        path = Path(out_path)
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


class TTSService:
    """Synthesise then play, reporting what actually happened.

    Tries a chain of engines in order and speaks with whichever one works
    first:
      1. Orpheus (best quality, natural voice, Groq-hosted -- limited by
         an external daily token quota; TTS_MODEL/TTS_VOICE in .env).
      2. Piper (local, offline, free, no rate limit ever -- a solid
         neural voice, just less expressive than Orpheus).
      3. espeak-ng (robotic, offline, always available) as the last
         resort, in case Piper has no voice model downloaded either.
    A wrong-sounding voice beats no voice at all.
    """

    def __init__(
        self,
        engines: list[EspeakTTS | OrpheusEngine | PiperEngine] | None = None,
        output_dir: str = "audio/output",
    ):
        self.engines = engines or _default_engines()
        self.output_dir = Path(output_dir)

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
        speech = None
        for i, engine in enumerate(self.engines):
            speech = engine.synthesize(text, self.output_dir / filename)
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

        if speech is None or not speech.ok or not autoplay:
            return speech

        played, interrupted, error = play(speech.wav_path, interrupt=interrupt)
        speech.played = played
        speech.interrupted = interrupted
        if error:
            # Not fatal: the WAV exists and the user can still play it manually.
            speech.error = error
        return speech


def _default_engines() -> list[EspeakTTS | OrpheusEngine | PiperEngine]:
    engines: list[EspeakTTS | OrpheusEngine | PiperEngine] = []

    orpheus = OrpheusEngine()
    if orpheus.available:
        engines.append(orpheus)
    else:
        log.warning("Orpheus TTS unavailable (%s)", orpheus._init_error)

    piper = PiperEngine()
    if piper.available:
        engines.append(piper)
    else:
        log.warning("Piper TTS unavailable (%s)", piper._init_error)

    engines.append(EspeakTTS())
    return engines
