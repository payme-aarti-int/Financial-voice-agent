from __future__ import annotations
 
import argparse
from pathlib import Path
 
from app.pipeline import Pipeline, PipelineResult
 
 
def load_audio_file(path: str | Path, target_rate: int = 16_000):
    
    import soundfile as sf
 
    audio, rate = sf.read(str(path), dtype="float32")
 
    if audio.ndim > 1:
        print(f"  note: {audio.shape[1]} channels, averaging to mono")
        audio = audio.mean(axis=1)
 
    if rate != target_rate:
        print(
            f"  WARNING: file is {rate} Hz, Whisper expects {target_rate} Hz "
            "-- transcription accuracy will be lower"
        )
 
    return audio
 
 
def report(result: PipelineResult) -> None:
    if result.transcript is not None:
        print(f'  heard: "{result.transcript.text}"')
        print(
            f"  lang={result.transcript.language} "
            f"p={result.transcript.language_probability:.2f}"
        )
 
    if result.rejected_reason:
        print(f"  REJECTED ({result.rejected_reason}) -- asking user to repeat")
 
    for call in result.tool_calls:
        available = call["result"].get("available")
        print(f"  tool: {call['name']}({call['arguments']}) -> available={available}")
 
    if result.agent_turn is not None:
        print(f'  answer: "{result.agent_turn.answer}"')
        if result.agent_turn.hit_iteration_limit:
            print("  WARNING: hit tool-call iteration limit")
 
    print(f'  spoken: "{result.spoken_text}"')
 
    if result.speech is not None:
        if result.speech.ok:
            state = (
                "played" if result.speech.played else f"saved {result.speech.wav_path}"
            )
            suffix = f" ({result.speech.error})" if result.speech.error else ""
            print(f"  tts: {state}{suffix}")
        else:
            print(f"  tts FAILED: {result.speech.error}")
 
    if result.timing is not None:
        stages = "  ".join(f"{n}={ms:.0f}ms" for n, ms in result.timing.stages.items())
        print(f"  timing: {stages}  TOTAL={result.timing.total_ms:.0f}ms")
 
 
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
        print(devices if devices.strip() else "NO AUDIO DEVICES FOUND")
        return
 
    autoplay = not args.no_play
 
    with Pipeline() as pipeline:
        if args.ask:
            print(f"\nQ: {args.ask}")
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
            print(pipeline.telemetry.report_summary())
            return
 
        if args.audio_file:
            print(f"\nFile: {args.audio_file}")
            audio = load_audio_file(args.audio_file)
            report(pipeline.run_audio(audio, autoplay=autoplay))
            return
 
        if args.turns:
            from app.audio.recorder import Recorder, peak_level
 
            recorder = Recorder()
            for i in range(1, args.turns + 1):
                print(f"\n--- turn {i}/{args.turns} --- speak now ({args.duration:.0f}s)")
                audio = recorder.record(args.duration)
                if peak_level(audio) < 0.01:
                    print("  WARNING: near-silent capture -- mic recorded nothing?")
                report(pipeline.run_audio(audio, autoplay=autoplay))
            print(pipeline.telemetry.report_summary())
            return
 
        parser.print_help()
 
 
if __name__ == "__main__":
    main()
 
