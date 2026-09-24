import soundfile as sf

from app.stt.whisper_stt import GroqSTT


def main():
    print("Loading STT model...")

    stt = GroqSTT()

    print("Transcribing audio...")

    audio, rate = sf.read("audio/input/test.mp3", dtype="float32")
    transcript = stt.transcribe(audio, sample_rate=rate)

    print("\n========== TRANSCRIPTION ==========")
    print(transcript.text)
    print("===================================")


if __name__ == "__main__":
    main()
