"""faster-whisper wrapper. Everything runs locally; models are cached on disk
after the first download, and ``local_files_only`` keeps it fully offline."""

import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Result:
    text: str
    audio_seconds: float
    infer_seconds: float
    language: str

    @property
    def ratio(self) -> float:
        """Inference time / audio time. Below 1.0 keeps up with live speech."""
        return self.infer_seconds / self.audio_seconds if self.audio_seconds else 0.0


class Transcriber:
    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        language: str | None = "en",
        beam_size: int = 5,
        offline: bool = False,
    ) -> None:
        # Imported here so the rest of the package (search, store) works
        # without the model runtime installed.
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            local_files_only=offline,
        )
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Result:
        start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # Skip non-speech inside the chunk; reduces made-up text in silence.
            vad_filter=True,
            # The tail of the previous chunk keeps names and spelling consistent.
            initial_prompt=prompt,
        )
        # segments is lazy: inference happens while iterating.
        text = "".join(s.text for s in segments).strip()
        return Result(
            text=text,
            audio_seconds=len(audio) / SAMPLE_RATE,
            infer_seconds=time.perf_counter() - start,
            language=info.language,
        )
