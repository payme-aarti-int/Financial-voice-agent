class GroqSTT:
    """Groq-hosted Whisper STT — replaces local faster-whisper."""

    def __init__(self):
        import os
        import httpx
        self._api_key = os.getenv("LLM_API_KEY", "")
        self._model = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        self._client = httpx.Client(timeout=30)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        import io
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)

        try:
            response = self._client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={
                    "model": self._model,
                    "language": "en",
                    "response_format": "json",
                },
            )
            response.raise_for_status()
            text = response.json().get("text", "").strip()

            return Transcript(
                text=text,
                language="en",
                language_probability=1.0,
                audio_duration_s=len(audio) / sample_rate,
                no_speech_prob=0.0 if text else 1.0,
            )

        except Exception as exc:
            return Transcript(
                text="",
                language="en",
                language_probability=0.0,
                audio_duration_s=len(audio) / sample_rate,
                no_speech_prob=1.0,
            )

    def close(self):
        self._client.close()