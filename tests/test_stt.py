from app.stt.whisper_stt import WhisperSTT


def main():
    print("Loading STT model...")

    stt = WhisperSTT()

    print("Transcribing audio...")

    text = stt.transcribe("audio/input/test.mp3")

    print("\n========== TRANSCRIPTION ==========")
    print(text)
    print("===================================")


if __name__ == "__main__":
    main()