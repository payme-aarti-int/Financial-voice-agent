from __future__ import annotations
 
import argparse
 
from app.audio.recorder import Recorder, peak_level
from app.stt.whisper_stt import WhisperSTT
from app.telemetry import Telemetry
 
 
def run(turns: int, duration: float, save_audio: bool) -> None:
    telemetry = Telemetry()
    recorder = Recorder()
 
    telemetry.begin_turn()
    with telemetry.stage("model_load"):
        stt = WhisperSTT()
    with telemetry.stage("warm_up"):
        stt.warm_up()
    startup = telemetry.end_turn()
    print(f"Startup: model_load + warm_up = {startup.total_ms:.0f} ms")
    print(telemetry.report_turn(startup))
    telemetry.turns.clear()  # exclude startup from the steady-state summary
 
    for i in range(1, turns + 1):
        print(f"\n--- turn {i}/{turns} --- speak now ({duration:.0f}s window)")
        telemetry.begin_turn()
 
        with telemetry.stage("record"):
            audio = recorder.record(duration)
 
        telemetry.set_audio_duration(len(audio) / recorder.sample_rate)
 
        level = peak_level(audio)
        if level < 0.01:
            print(f"  WARNING: peak level {level:.4f} -- did the mic capture anything?")
 
        if save_audio:
            with telemetry.stage("disk_write"):
                recorder.save(audio, f"audio/input/turn_{i}.wav")
 
        with telemetry.stage("transcribe"):
            result = stt.transcribe(audio)
 
        print(f'  transcript: "{result.text}"')
        print(
            f"  lang={result.language} p={result.language_probability:.2f} "
            f"no_speech={result.no_speech_prob}"
        )
        if result.is_empty:
            print("  -> empty transcript")
        elif result.is_suspect:
            print("  -> SUSPECT: would ask the user to repeat rather than answer")
 
        turn = telemetry.end_turn()
        print(telemetry.report_turn(turn))
 
    print(telemetry.report_summary())
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Voice Agent -- baseline")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Write each turn to audio/input/ for debugging",
    )
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
 
    if args.list_devices:
        print(Recorder.list_devices())
        return
 
    run(turns=args.turns, duration=args.duration, save_audio=args.save_audio)
 
 
if __name__ == "__main__":
    main()
 
