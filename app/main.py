from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.pipeline import Pipeline, PipelineResult

log = logging.getLogger(__name__)


def load_audio_file(path: str | Path, target_rate: int = 16_000):

    import soundfile as sf

    if not Path(path).exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    audio, rate = sf.read(str(path), dtype="float32")

    if audio.ndim > 1:
        log.info("note: %d channels, averaging to mono", audio.shape[1])
        audio = audio.mean(axis=1)

    if rate != target_rate:
        log.warning(
            "file is %d Hz, Whisper expects %d Hz -- transcription accuracy "
            "will be lower",
            rate,
            target_rate,
        )

    return audio


def report(result: PipelineResult) -> None:
    if result.transcript is not None:
        log.info('heard: "%s"', result.transcript.text)
        log.info(
            "lang=%s p=%.2f",
            result.transcript.language,
            result.transcript.language_probability,
        )

    if result.rejected_reason:
        log.warning("REJECTED (%s) -- asking user to repeat", result.rejected_reason)

    for call in result.tool_calls:
        available = call["result"].get("available")
        log.info(
            "tool: %s(%s) -> available=%s", call["name"], call["arguments"], available
        )

    if result.agent_turn is not None:
        log.info('answer: "%s"', result.agent_turn.answer)
        if result.agent_turn.hit_iteration_limit:
            log.warning("hit tool-call iteration limit")

    log.info('spoken: "%s"', result.spoken_text)

    if result.speech is not None:
        if result.speech.ok:
            state = (
                "played" if result.speech.played else f"saved {result.speech.wav_path}"
            )
            suffix = f" ({result.speech.error})" if result.speech.error else ""
            log.info("tts: %s%s", state, suffix)
        else:
            log.error("tts FAILED: %s", result.speech.error)

    if result.timing is not None:
        stages = "  ".join(f"{n}={ms:.0f}ms" for n, ms in result.timing.stages.items())
        log.info("timing: %s  TOTAL=%.0fms", stages, result.timing.total_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Voice Agent")
    parser.add_argument("--ask", type=str, help="Ask one question as text")
    parser.add_argument("--chat", action="store_true", help="Interactive text loop")
    parser.add_argument("--audio-file", type=str, help="Answer from an audio file")
    parser.add_argument("--turns", type=int, default=0, help="Live mic turns")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--no-play", action="store_true", help="Synthesize, do not play")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        from app.audio.recorder import Recorder

        devices = Recorder.list_devices()
        log.info("%s", devices if devices.strip() else "NO AUDIO DEVICES FOUND")
        return

    autoplay = not args.no_play

    with Pipeline() as pipeline:
        if args.ask:
            log.info("Q: %s", args.ask)
            report(pipeline.run_text(args.ask, autoplay=autoplay))
            return

        if args.chat:
            print("Type a question, or 'quit' to exit.\n")
            while True:
                try:
                    question = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if question.lower() in {"quit", "exit", "q"}:
                    break
                if not question:
                    continue
                report(pipeline.run_text(question, autoplay=autoplay))
                print()
            log.info("%s", pipeline.telemetry.report_summary())
            return

        if args.audio_file:
            log.info("File: %s", args.audio_file)
            try:
                audio = load_audio_file(args.audio_file)
            except FileNotFoundError as exc:
                log.error("%s", exc)
                return
            report(pipeline.run_audio(audio, autoplay=autoplay))
            return

        if args.turns:
            from app.audio.recorder import Recorder, peak_level

            recorder = Recorder()
            for i in range(1, args.turns + 1):
                log.info("--- turn %d/%d --- speak now (%.0fs)", i, args.turns, args.duration)
                try:
                    audio = recorder.record(args.duration)
                except Exception as exc:
                    log.error("recording failed: %s -- skipping this turn", exc)
                    continue
                if peak_level(audio) < 0.01:
                    log.warning("near-silent capture -- mic recorded nothing?")
                report(pipeline.run_audio(audio, autoplay=autoplay))
            log.info("%s", pipeline.telemetry.report_summary())
            return

        parser.print_help()


if __name__ == "__main__":
    main()
