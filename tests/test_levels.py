import numpy as np

from offline_stt.levels import meter_bar, peak_db, rms_db, summarise


def test_full_scale_sine_levels():
    t = np.arange(16000) / 16000
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert abs(peak_db(sine)) < 0.01
    assert abs(rms_db(sine) - (-3.01)) < 0.05


def test_silence_is_floor():
    assert rms_db(np.zeros(100, dtype=np.float32)) == -120.0


def test_summarise_estimates_snr_and_clipping():
    sr = 16000
    rng = np.random.default_rng(0)
    noise = 0.001 * rng.standard_normal(sr)          # about -60 dB
    speech = 0.1 * rng.standard_normal(sr)           # about -20 dB
    audio = np.concatenate([noise, speech]).astype(np.float32)
    s = summarise(audio, sr)
    assert 35 < s.snr_db < 45
    assert s.clip_pct == 0
    assert s.verdict() == "OK"

    clipped = np.clip(audio * 50, -1, 1)
    assert summarise(clipped, sr).verdict().startswith("CLIPPING")


def test_meter_bar_bounds():
    assert meter_bar(-100, width=10) == "-" * 10
    assert meter_bar(0, width=10) == "#" * 10
