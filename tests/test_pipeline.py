from datetime import datetime

import numpy as np

from offline_stt.chunker import Chunker
from offline_stt.pipeline import LiveSession
from offline_stt.store import TranscriptStore
from offline_stt.transcriber import Result

SR = 16000


class FakeTranscriber:
    def __init__(self):
        self.prompts = []
        self.n = 0

    def transcribe(self, audio, prompt=None):
        self.prompts.append(prompt)
        self.n += 1
        return Result(text=f"chunk {self.n}", audio_seconds=len(audio) / SR,
                      infer_seconds=0.01, language="en")


def test_session_writes_timestamped_lines_across_windows(tmp_path):
    rng = np.random.default_rng(2)
    speech = (0.1 * rng.standard_normal(SR * 4)).astype(np.float32)
    # Start 3 s before 09:10 so the second chunk lands in the next window.
    started = datetime(2026, 10, 21, 9, 9, 57)
    fake = FakeTranscriber()
    session = LiveSession(Chunker(min_seconds=2, max_seconds=4, adaptive=False), fake,
                          TranscriptStore(tmp_path, "mic"), started_at=started)
    for i in range(0, len(speech) * 2, SR // 10):
        session.feed(np.concatenate([speech, speech])[i:i + SR // 10])
    session.close()

    first = tmp_path / "mic" / "2026-10-21" / "09" / "00.txt"
    second = tmp_path / "mic" / "2026-10-21" / "09" / "10.txt"
    assert first.read_text().splitlines()[1:] == ["[09:09:57] chunk 1"]
    assert second.read_text().splitlines()[1:] == ["[09:10:01] chunk 2"]
    assert fake.prompts == [None, "chunk 1"]
    assert session.chunks_written == 2
