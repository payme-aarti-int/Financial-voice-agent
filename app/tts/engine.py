from __future__ import annotations
 
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import soundfile as sf

from app.config import AUDIO
 
 
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
 
 
def play(wav_path: str | Path) -> tuple[bool, str | None]:
    
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001 - OSError when PortAudio is absent
        return False, f"audio output unavailable: {exc}"
 
    try:
        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        sd.play(np.squeeze(audio), sample_rate, device=AUDIO.device)
        sd.wait()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"playback failed: {exc}"
 
 
class TTSService:
    """Synthesise then play, reporting what actually happened."""
 
    def __init__(self, engine: EspeakTTS | None = None, output_dir: str = "audio/output"):
        self.engine = engine or EspeakTTS()
        self.output_dir = Path(output_dir)
 
    def speak(self, text: str, filename: str = "reply.wav", autoplay: bool = True) -> Speech:
        speech = self.engine.synthesize(text, self.output_dir / filename)
        if not speech.ok or not autoplay:
            return speech
 
        played, error = play(speech.wav_path)
        speech.played = played
        if error:
            # Not fatal: the WAV exists and the user can still play it manually.
            speech.error = error
        return speech
 
