"""Cut a continuous audio stream into chunks at natural pauses.

Whisper works on whole utterances, so cutting mid-word costs accuracy. The
chunker collects incoming blocks and cuts once the buffer is at least
``min_seconds`` long and the speaker has paused for ``pause_seconds``. If
nobody pauses, it cuts anyway at ``max_seconds`` so text keeps flowing and
memory stays bounded. Chunks that never rise above the silence threshold are
dropped rather than transcribed, which saves CPU and avoids Whisper inventing
words in silence.

"Silence" adapts to the room: it is anything quieter than ``silence_db`` or
within ``margin_db`` of the recent background noise, whichever is louder. That
way a noisy room or a laptop mic with automatic gain still gets cut at pauses.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from offline_stt.levels import rms_db


@dataclass
class Chunk:
    audio: np.ndarray
    # Offset of the first sample from the start of the stream, in samples.
    start_sample: int

    def start_seconds(self, sample_rate: int) -> float:
        return self.start_sample / sample_rate

    def duration(self, sample_rate: int) -> float:
        return len(self.audio) / sample_rate


class Chunker:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_seconds: float = 5.0,
        max_seconds: float = 30.0,
        pause_seconds: float = 0.6,
        silence_db: float = -45.0,
        margin_db: float = 10.0,
        history_blocks: int = 300,
        adaptive: bool = True,
    ) -> None:
        if not 0 < min_seconds <= max_seconds:
            raise ValueError("need 0 < min_seconds <= max_seconds")
        self.sample_rate = sample_rate
        self.min_samples = int(min_seconds * sample_rate)
        self.max_samples = int(max_seconds * sample_rate)
        self.pause_samples = int(pause_seconds * sample_rate)
        self.silence_db = silence_db
        self.margin_db = margin_db
        self.adaptive = adaptive
        self._levels: deque[float] = deque(maxlen=history_blocks)

        self._blocks: list[np.ndarray] = []
        self._buffered = 0
        self._start_sample = 0
        self._trailing_silence = 0
        self._heard_speech = False

    def feed(self, block: np.ndarray) -> list[Chunk]:
        """Add a block of mono float32 samples; return any finished chunks."""
        out: list[Chunk] = []
        # Split oversized blocks so a cut never overshoots max_seconds.
        pos = 0
        while pos < len(block):
            room = self.max_samples - self._buffered
            piece = block[pos:pos + room]
            pos += len(piece)
            self._append(piece)
            chunk = self._maybe_cut()
            if chunk is not None:
                out.append(chunk)
        return [c for c in out if c is not _DROPPED]

    def flush(self) -> Chunk | None:
        """Return whatever is buffered (e.g. on shutdown)."""
        if self._buffered == 0:
            return None
        chunk = self._cut()
        return None if chunk is _DROPPED else chunk

    def _append(self, piece: np.ndarray) -> None:
        if len(piece) == 0:
            return
        self._blocks.append(piece)
        self._buffered += len(piece)
        level = rms_db(piece)
        self._levels.append(level)
        if level < self.threshold_db:
            self._trailing_silence += len(piece)
        else:
            self._trailing_silence = 0
            self._heard_speech = True

    @property
    def noise_floor_db(self) -> float | None:
        """Quiet end (10th percentile) of recent block levels."""
        if not self._levels:
            return None
        return float(np.percentile(self._levels, 10))

    @property
    def threshold_db(self) -> float:
        floor = self.noise_floor_db if self.adaptive else None
        if floor is None:
            return self.silence_db
        return max(self.silence_db, floor + self.margin_db)

    def _maybe_cut(self):
        if self._buffered >= self.max_samples:
            return self._cut()
        if self._buffered >= self.min_samples and self._trailing_silence >= self.pause_samples:
            return self._cut()
        if not self._heard_speech and self._trailing_silence >= self.min_samples:
            # Long stretch of silence: discard it without transcribing.
            return self._cut()
        return None

    def _cut(self):
        audio = np.concatenate(self._blocks) if len(self._blocks) > 1 else self._blocks[0]
        chunk = Chunk(audio=audio, start_sample=self._start_sample) if self._heard_speech else _DROPPED
        self._start_sample += self._buffered
        self._blocks = []
        self._buffered = 0
        self._trailing_silence = 0
        self._heard_speech = False
        return chunk


# Sentinel for silent chunks that are discarded.
_DROPPED = Chunk(audio=np.zeros(0, dtype=np.float32), start_sample=-1)
