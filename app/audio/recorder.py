import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16_000
CHANNELS = 1


def record_audio(duration: int, output_path: str) -> None:
    print(f"Recording for {duration} seconds...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )

    sd.wait()

    sf.write(
        output_path,
        audio,
        SAMPLE_RATE,
    )

    print(f"Audio saved to: {output_path}")