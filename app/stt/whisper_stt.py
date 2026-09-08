from faster_whisper import WhisperModel


class WhisperSTT:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str) -> str:
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return text