from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app import observability
from app.agent.loop import AgentTurn, FinancialAgent
from app.stt.whisper_stt import Transcript, WhisperSTT, GroqSTT
from app.telemetry import Telemetry, Turn
from app.tts.engine import Speech, TTSService
from app.tts.speakable import to_speakable

REPEAT_PROMPT = "Sorry, I did not catch that. Could you say it again?"


@dataclass
class PipelineResult:
    transcript: Transcript | None = None
    agent_turn: AgentTurn | None = None
    spoken_text: str = ""
    speech: Speech | None = None
    timing: Turn | None = None
    rejected_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        stt: GroqSTT | None = None,
        agent: FinancialAgent | None = None,
        tts: TTSService | None = None,
        telemetry: Telemetry | None = None,
    ):
        self._stt = stt
        self.agent = agent or FinancialAgent()
        self.tts = tts or TTSService()
        self.telemetry = telemetry or Telemetry()
        observability.start_pipeline_run()

    @property
    def stt(self) -> GroqSTT:
        """Loaded lazily: text-only runs should not pay for model load."""
        if self._stt is None:
            self._stt = GroqSTT()
        return self._stt

    def _finish(self, result: PipelineResult) -> PipelineResult:
        result.timing = self.telemetry.end_turn()
        observability.log_turn(
            result.timing.stages, result.timing.total_ms, step=len(self.telemetry.turns)
        )
        return result

    def run_text(self, question: str, autoplay: bool = True) -> PipelineResult:
        """Answer a typed question and speak the reply."""
        self.telemetry.begin_turn()
        result = PipelineResult()

        with self.telemetry.stage("agent"):
            result.agent_turn = self.agent.ask(question)
        result.tool_calls = result.agent_turn.tool_calls

        with self.telemetry.stage("speakable"):
            result.spoken_text = to_speakable(result.agent_turn.answer)

        with self.telemetry.stage("tts"):
            result.speech = self.tts.speak(result.spoken_text, autoplay=autoplay)

        return self._finish(result)

    def run_audio(
        self,
        audio: np.ndarray | str | Path,
        autoplay: bool = True,
    ) -> PipelineResult:
        self.telemetry.begin_turn()
        result = PipelineResult()

        with self.telemetry.stage("transcribe"):
            transcript = self.stt.transcribe(audio)
        result.transcript = transcript
        self.telemetry.set_audio_duration(transcript.audio_duration_s)

        if transcript.is_empty:
            result.rejected_reason = "empty transcript"
        elif transcript.is_suspect:
            result.rejected_reason = "low ASR confidence"

        if result.rejected_reason:
            result.spoken_text = REPEAT_PROMPT
            with self.telemetry.stage("tts"):
                result.speech = self.tts.speak(REPEAT_PROMPT, autoplay=autoplay)
            return self._finish(result)

        with self.telemetry.stage("agent"):
            result.agent_turn = self.agent.ask(transcript.text)
        result.tool_calls = result.agent_turn.tool_calls

        with self.telemetry.stage("speakable"):
            result.spoken_text = to_speakable(result.agent_turn.answer)

        with self.telemetry.stage("tts"):
            result.speech = self.tts.speak(result.spoken_text, autoplay=autoplay)

        return self._finish(result)

    def close(self) -> None:
        observability.end_pipeline_run(self.telemetry)
        self.agent.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()