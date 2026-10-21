"""Microphone capture. Audio lives only in memory: blocks go from the sound
card into a queue, are resampled to 16 kHz mono, and are never written out."""

import queue
from dataclasses import dataclass

import numpy as np

TARGET_RATE = 16000


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    channels: int
    default_rate: int
    is_default: bool


def list_input_devices() -> list[InputDevice]:
    import sounddevice as sd

    apis = sd.query_hostapis()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        out.append(InputDevice(
            index=i,
            name=d["name"],
            host_api=apis[d["hostapi"]]["name"],
            channels=d["max_input_channels"],
            default_rate=int(d["default_samplerate"]),
            is_default=(i == default_in),
        ))
    return out


def resolve_device(spec: str | None) -> int | None:
    """Accept a device index or part of its name; None means the system default."""
    if spec is None or spec == "":
        return None
    if spec.isdigit():
        return int(spec)
    matches = [d for d in list_input_devices() if spec.lower() in d.name.lower()]
    if not matches:
        raise SystemExit(f"No input device matches {spec!r}. Run `ostt devices` to list them.")
    return matches[0].index


def device_name(index: int | None) -> str:
    import sounddevice as sd

    return sd.query_devices(index if index is not None else sd.default.device[0])["name"]


class MicStream:
    """Context manager that yields 16 kHz mono float32 blocks from a mic.

    ``channel`` picks one input channel (0-based), which matters for sound
    cards where an XLR mic arrives on, say, the right channel only.
    """

    def __init__(self, device: int | None = None, channel: int | None = None,
                 block_seconds: float = 0.1) -> None:
        import sounddevice as sd
        import soxr

        info = sd.query_devices(device if device is not None else sd.default.device[0])
        self.device = device
        self.channel = channel
        self.channels = (channel + 1) if channel is not None else 1
        if self.channels > info["max_input_channels"]:
            raise SystemExit(
                f"Device has {info['max_input_channels']} input channel(s); "
                f"channel {channel} does not exist."
            )
        self.rate = self._pick_rate(sd, device, self.channels)
        self.block = int(self.rate * block_seconds)
        self._resampler = None if self.rate == TARGET_RATE else soxr.ResampleStream(
            self.rate, TARGET_RATE, 1, dtype="float32")
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self.overflows = 0
        self._stream = sd.InputStream(
            device=device, channels=self.channels, samplerate=self.rate,
            blocksize=self.block, dtype="float32", callback=self._callback,
        )

    @staticmethod
    def _pick_rate(sd, device, channels: int) -> int:
        """Capture at 16 kHz if the driver allows, else the device's own rate."""
        try:
            sd.check_input_settings(device=device, samplerate=TARGET_RATE, channels=channels)
            return TARGET_RATE
        except Exception:
            info = sd.query_devices(device if device is not None else sd.default.device[0])
            return int(info["default_samplerate"])

    def _callback(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        col = self.channel if self.channel is not None else 0
        self._q.put(indata[:, col].copy())

    def __enter__(self) -> "MicStream":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next block at 16 kHz, or None if nothing arrived within ``timeout``."""
        try:
            raw = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        if self._resampler is None:
            return raw
        return self._resampler.resample_chunk(raw)
