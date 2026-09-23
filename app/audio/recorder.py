"""Microphone capture.

STILL BATCH -- deliberate. This is the baseline we measure, not what we ship.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.config import AUDIO

# Slack added on top of the requested duration before we give up on a stream
# and call it dead. A flaky device (e.g. a USB headset dropping out) can leave
# sd.wait() blocked forever with no exception -- worse than a crash, since
# nothing tells the caller anything is wrong.
TIMEOUT_SLACK_S = 5.0


def wait_with_timeout(timeout_s: float) -> bool:
    """sd.wait(), but give up after `timeout_s` instead of blocking forever.

    Returns True if the stream finished on its own, False if we timed out and
    called sd.stop() to abort it.
    """
    done = threading.Event()

    def _waiter() -> None:
        sd.wait()
        done.set()

    thread = threading.Thread(target=_waiter, daemon=True)
    thread.start()
    if done.wait(timeout_s):
        return True
    sd.stop()
    return False


_UNSET = object()


def call_with_timeout(func, timeout_s: float):
    """Run a PortAudio call that can block forever (e.g. query_devices() on a
    flaky device) off-thread, so a dead device gives us a clear error instead
    of hanging the caller. The stuck call itself keeps running in a daemon
    thread -- there's no way to cancel it -- but the caller gets control back.
    """
    result = [_UNSET]
    error = [None]

    def _runner() -> None:
        try:
            result[0] = func()
        except Exception as exc:  # noqa: BLE001
            error[0] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"did not respond within {timeout_s:.0f}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


class Recorder:
    def __init__(self, sample_rate: int | None = None, channels: int | None = None):
        self.sample_rate = sample_rate or AUDIO.sample_rate
        self.channels = channels or AUDIO.channels

    def record(self, duration: float) -> np.ndarray:
        """Block for `duration` seconds and return mono float32 audio.

        Note the flaw this API forces on you: the caller must guess how long the
        user will speak. Eliminating that guess is what endpointing is for.
        """
        frames = int(duration * self.sample_rate)
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=AUDIO.dtype,
            device=AUDIO.device,
        )
        if not wait_with_timeout(duration + TIMEOUT_SLACK_S):
            raise RuntimeError(
                f"recording did not finish within {duration + TIMEOUT_SLACK_S:.0f}s "
                "-- the audio input device may be unresponsive or disconnected"
            )
        # sounddevice returns (n, 1) for mono; Whisper wants (n,). Passing the
        # 2-D array does not raise -- it silently misbehaves. Normalise here.
        return np.squeeze(audio)

    @staticmethod
    def save(audio: np.ndarray, path: str | Path, sample_rate: int | None = None) -> None:
        """Debug helper. Not in the hot path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), audio, sample_rate or AUDIO.sample_rate)

    @staticmethod
    def list_devices() -> str:
        try:
            devices = call_with_timeout(sd.query_devices, timeout_s=8.0)
        except TimeoutError:
            return "device enumeration timed out -- an audio device may be stuck"
        except Exception as exc:  # noqa: BLE001
            return f"device enumeration failed: {exc}"
        return str(devices)


def peak_level(audio: np.ndarray) -> float:
    """Loudest sample, 0.0-1.0.

    Near-zero means you recorded silence: wrong device, muted mic, or
    permissions. Check this before blaming the ASR for an empty transcript.
    """
    return float(np.max(np.abs(audio))) if audio.size else 0.0
