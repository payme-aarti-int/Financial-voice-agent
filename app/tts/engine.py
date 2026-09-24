from __future__ import annotations
 
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import soundfile as sf

from app.config import AUDIO

log = logging.getLogger(__name__)
 
 
@dataclass
class Speech:
 
    text: str
    wav_path: Path | None = None
    duration_s: float | None = None
    played: bool = False
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


def play(wav_path: str | Path) -> tuple[bool, str | None]:

    try:
        import sounddevice as sd

        from app.audio.recorder import TIMEOUT_SLACK_S, wait_with_timeout
    except Exception as exc:  # noqa: BLE001 - OSError when PortAudio is absent
        return False, f"audio output unavailable: {exc}"

    try:
        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        duration_s = len(audio) / sample_rate
        sd.play(np.squeeze(audio), sample_rate, device=AUDIO.device)
        if not wait_with_timeout(duration_s + TIMEOUT_SLACK_S):
            return False, (
                f"playback did not finish within {duration_s + TIMEOUT_SLACK_S:.0f}s "
                "-- the audio output device may be unresponsive or disconnected"
            )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"playback failed: {exc}"
 
 
class TTSService:
    """Synthesise then play, reporting what actually happened.

    Prefers Orpheus (natural voice, via Groq -- TTS_MODEL/TTS_VOICE in
    .env) and falls back to espeak-ng (robotic, offline, always available)
    if Orpheus isn't configured or a synthesis call fails. A wrong-sounding
    voice beats no voice at all.
    """

    def __init__(
        self,
        engine: EspeakTTS | OrpheusEngine | None = None,
        fallback_engine: EspeakTTS | None = None,
        output_dir: str = "audio/output",
    ):
        self.engine = engine or _default_engine()
        self.fallback_engine = fallback_engine or EspeakTTS()
        self.output_dir = Path(output_dir)

    def speak(self, text: str, filename: str = "reply.wav", autoplay: bool = True) -> Speech:
        speech = self.engine.synthesize(text, self.output_dir / filename)
        if not speech.ok and self.engine is not self.fallback_engine:
            log.warning("primary TTS failed (%s), falling back to espeak", speech.error)
            speech = self.fallback_engine.synthesize(text, self.output_dir / filename)
        if not speech.ok or not autoplay:
            return speech

        played, error = play(speech.wav_path)
        speech.played = played
        if error:
            # Not fatal: the WAV exists and the user can still play it manually.
            speech.error = error
        return speech


def _default_engine() -> EspeakTTS | OrpheusEngine:
    orpheus = OrpheusEngine()
    if orpheus.available:
        return orpheus
    log.warning("Orpheus TTS unavailable (%s); falling back to espeak-ng", orpheus._init_error)
    return EspeakTTS()
