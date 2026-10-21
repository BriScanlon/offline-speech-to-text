import numpy as np

from offline_stt.chunker import Chunker

SR = 16000
BLOCK = SR // 10  # 100 ms, as captured live


def speech(seconds):
    rng = np.random.default_rng(1)
    return (0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def feed_in_blocks(chunker, audio):
    out = []
    for i in range(0, len(audio), BLOCK):
        out += chunker.feed(audio[i:i + BLOCK])
    return out


def test_cuts_at_pause_after_minimum_length():
    c = Chunker(adaptive=False, min_seconds=2, max_seconds=30, pause_seconds=0.5)
    audio = np.concatenate([speech(3), silence(1), speech(3)])
    chunks = feed_in_blocks(c, audio)
    assert len(chunks) == 1
    assert chunks[0].start_sample == 0
    # Cut after 3 s of speech plus the 0.5 s pause.
    assert abs(chunks[0].duration(SR) - 3.5) < 0.11
    last = c.flush()
    assert last is not None
    assert last.start_sample == len(chunks[0].audio)


def test_short_pause_before_minimum_does_not_cut():
    c = Chunker(adaptive=False, min_seconds=5, max_seconds=30, pause_seconds=0.5)
    chunks = feed_in_blocks(c, np.concatenate([speech(1), silence(1), speech(1)]))
    assert chunks == []


def test_forces_cut_at_max_length():
    c = Chunker(adaptive=False, min_seconds=2, max_seconds=4, pause_seconds=0.5)
    chunks = feed_in_blocks(c, speech(10))
    assert [ch.duration(SR) for ch in chunks] == [4.0, 4.0]
    assert [ch.start_sample for ch in chunks] == [0, 4 * SR]


def test_large_block_is_split_at_max_length():
    c = Chunker(adaptive=False, min_seconds=1, max_seconds=2, pause_seconds=0.5)
    chunks = c.feed(speech(5))
    assert [ch.duration(SR) for ch in chunks] == [2.0, 2.0]


def test_silence_is_dropped_but_keeps_time_offsets():
    c = Chunker(adaptive=False, min_seconds=2, max_seconds=30, pause_seconds=0.5)
    chunks = feed_in_blocks(c, np.concatenate([silence(5), speech(3), silence(1)]))
    assert len(chunks) == 1
    # The speech chunk begins where the dropped silence ended.
    assert chunks[0].start_sample >= 2 * SR
    assert chunks[0].audio.std() > 0
    assert c.flush() is None


def test_adapts_to_background_noise():
    rng = np.random.default_rng(3)

    def room(seconds):  # steady hum around -40 dBFS, louder than the -45 default
        return (0.01 * rng.standard_normal(int(seconds * SR))).astype(np.float32)

    c = Chunker(min_seconds=2, max_seconds=30, pause_seconds=0.5)
    audio = np.concatenate([room(3), speech(3) + room(3), room(1), speech(3) + room(3), room(1)])
    chunks = feed_in_blocks(c, audio)
    # Without adapting, the hum never counts as a pause and nothing is cut.
    assert len(chunks) == 2
    assert c.threshold_db > -45
    assert chunks[0].start_sample >= 2 * SR  # leading hum was dropped
