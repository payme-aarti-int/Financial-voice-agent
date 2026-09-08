import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16_000
CHANNELS = 1


def record_audio(duration: int, output_path: str) -> None:
    """
    Record audio from the default microphone.

    Parameters:
        duration: Recording duration in seconds.
        output_path: Where the WAV file will be saved.
    """

    print(f"🎤 Recording for {duration} seconds...")
    print("Speak now...")

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

    print(f"✅ Audio saved to: {output_path}")