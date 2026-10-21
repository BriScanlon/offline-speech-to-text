"""Glue between capture, chunking, transcription and the text store.

Capture and transcription run on separate threads so a slow chunk never makes
the microphone drop audio. Each chunk's audio is released as soon as its text
has been written.
"""

import logging
import queue
import threading
from datetime import datetime, timedelta

import numpy as np

from offline_stt.chunker import Chunk, Chunker
from offline_stt.store import TranscriptStore

log = logging.getLogger(__name__)

# Characters of previous text passed to Whisper as context for the next chunk.
PROMPT_CHARS = 200


class LiveSession:
    def __init__(self, chunker: Chunker, transcriber, store: TranscriptStore,
                 started_at: datetime | None = None, on_text=None) -> None:
        self.chunker = chunker
        self.transcriber = transcriber
        self.store = store
        self.started_at = started_at or datetime.now()
        self.on_text = on_text
        self.audio_seconds = 0.0
        self.infer_seconds = 0.0
        self.chunks_written = 0
        self._q: queue.Queue[Chunk | None] = queue.Queue()
        self._prompt: str | None = None
        self._worker = threading.Thread(target=self._run, name="transcriber", daemon=True)
        self._worker.start()

    @property
    def backlog(self) -> int:
        """Chunks waiting for transcription. If this keeps growing, the model is
        too slow for this machine: pick a smaller one."""
        return self._q.qsize()

    def feed(self, block: np.ndarray) -> None:
        for chunk in self.chunker.feed(block):
            self._q.put(chunk)

    def close(self) -> None:
        """Transcribe whatever is buffered and wait for the worker to finish."""
        last = self.chunker.flush()
        if last is not None:
            self._q.put(last)
        self._q.put(None)
        self._worker.join()

    def _run(self) -> None:
        while True:
            chunk = self._q.get()
            if chunk is None:
                return
            try:
                self._handle(chunk)
            except Exception:
                log.exception("Transcription of a chunk failed; continuing")

    def _handle(self, chunk: Chunk) -> None:
        sr = self.chunker.sample_rate
        when = self.started_at + timedelta(seconds=chunk.start_seconds(sr))
        result = self.transcriber.transcribe(chunk.audio, prompt=self._prompt)
        del chunk  # drop the audio as soon as it has been transcribed
        self.audio_seconds += result.audio_seconds
        self.infer_seconds += result.infer_seconds
        path = self.store.append(when, result.text)
        if path is None:
            return
        self.chunks_written += 1
        self._prompt = result.text[-PROMPT_CHARS:]
        if self.on_text:
            self.on_text(when, result, path)
