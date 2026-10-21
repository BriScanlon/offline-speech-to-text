"""Signal level measurements used by the level meter and the mic test."""

from dataclasses import dataclass

import numpy as np

# Floor for silent blocks so log10 never sees zero.
_MIN_DB = -120.0
# A sample this close to full scale counts as clipped.
CLIP_THRESHOLD = 0.999


def to_db(value: float) -> float:
    """Convert a linear amplitude (0..1) to dBFS."""
    if value <= 0:
        return _MIN_DB
    return max(_MIN_DB, 20.0 * float(np.log10(value)))


def rms_db(block: np.ndarray) -> float:
    if block.size == 0:
        return _MIN_DB
    return to_db(float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))))


def peak_db(block: np.ndarray) -> float:
    if block.size == 0:
        return _MIN_DB
    return to_db(float(np.max(np.abs(block))))


def clip_fraction(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.mean(np.abs(block) >= CLIP_THRESHOLD))


@dataclass(frozen=True)
class LevelStats:
    peak_db: float
    rms_db: float
    noise_floor_db: float
    speech_db: float
    snr_db: float
    clip_pct: float

    def verdict(self) -> str:
        """One-line advice on whether the input gain looks right."""
        if self.clip_pct > 0.1:
            return "CLIPPING: lower the input gain or move the mic further away"
        if self.peak_db > -3:
            return "HOT: peaks are close to clipping, consider a little less gain"
        if self.speech_db < -40:
            return "QUIET: raise the input gain or move the mic closer"
        if self.snr_db < 15:
            return "NOISY: speech is not far above the background noise"
        return "OK"


def summarise(audio: np.ndarray, sample_rate: int, window_s: float = 0.1) -> LevelStats:
    """Level statistics over a recording.

    The recording is split into short windows. The noise floor is the quiet end
    of the window levels (10th percentile) and the speech level is the loud end
    (90th percentile), so their difference is a rough signal-to-noise ratio
    without needing a separate silence recording.
    """
    win = max(1, int(sample_rate * window_s))
    n = len(audio) // win
    if n == 0:
        window_dbs = np.array([rms_db(audio)])
    else:
        window_dbs = np.array([rms_db(audio[i * win:(i + 1) * win]) for i in range(n)])
    noise = float(np.percentile(window_dbs, 10))
    speech = float(np.percentile(window_dbs, 90))
    return LevelStats(
        peak_db=round(peak_db(audio), 1),
        rms_db=round(rms_db(audio), 1),
        noise_floor_db=round(noise, 1),
        speech_db=round(speech, 1),
        snr_db=round(speech - noise, 1),
        clip_pct=round(100.0 * clip_fraction(audio), 3),
    )


def meter_bar(level_db: float, width: int = 40, floor_db: float = -60.0) -> str:
    """Text bar for a live meter: empty at floor_db, full at 0 dBFS."""
    frac = min(1.0, max(0.0, (level_db - floor_db) / -floor_db))
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)
