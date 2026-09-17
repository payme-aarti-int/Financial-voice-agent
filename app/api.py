"""HTTP layer over the existing pipeline.

Adds no logic. Takes a question, hands it to Pipeline, returns what comes back.
Every guarantee -- the model never produces a figure, tool errors return as
data, missing periods refuse rather than guess -- lives below this layer.

Audio is the browser's job. The server has no microphone and never will:
getUserMedia() captures, the page downsamples to 16 kHz mono WAV, and posts the
bytes here. That sidesteps ALSA, PulseAudio, PipeWire and PortAudio entirely.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Financial Voice Agent")

pipeline = Pipeline()


class Question(BaseModel):
    text: str


def _serialise(result) -> dict:
    return {
        "heard": result.transcript.text if result.transcript is not None else None,
        "rejected": result.rejected_reason,
        "answer": result.agent_turn.answer if result.agent_turn is not None else None,
        "spoken": result.spoken_text,
        "tools": [
            {
                "name": call["name"],
                "arguments": call["arguments"],
                "available": call["result"].get("available"),
            }
            for call in result.tool_calls
        ],
        "timing": (
            {name: round(ms) for name, ms in result.timing.stages.items()}
            if result.timing is not None
            else {}
        ),
        "total_ms": round(result.timing.total_ms) if result.timing is not None else None,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/ask")
def ask(question: Question):
    text = question.text.strip()
    if not text:
        return JSONResponse({"error": "empty question"}, status_code=400)

    log.info("ask: %s", text)
    try:
        result = pipeline.run_text(text, autoplay=False)
    except Exception as exc:
        log.exception("ask failed")
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)

    return _serialise(result)


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...)):
    raw = await audio.read()
    log.info("voice: %d bytes", len(raw))

    try:
        samples, rate = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        return JSONResponse({"error": f"could not decode audio: {exc}"}, status_code=400)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    log.info("voice: %d samples at %d Hz, peak %.4f", samples.size, rate, peak)

    if peak < 0.01:
        return {
            "heard": None,
            "rejected": "near-silent capture -- microphone picked up nothing",
            "answer": None,
            "spoken": "I did not hear anything. Please check your microphone.",
            "tools": [],
            "timing": {},
            "total_ms": None,
            "peak": peak,
        }

    try:
        result = pipeline.run_audio(samples, autoplay=False)
    except Exception as exc:
        log.exception("voice failed")
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)

    payload = _serialise(result)
    payload["peak"] = peak
    return payload
