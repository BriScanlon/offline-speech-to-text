"""Write transcript text into a ZoneMinder-style date/time folder tree.

ZoneMinder files event frames under ``<monitor>/<YY-MM-DD>/<HH>/<MM>/...``.
This keeps the same idea for text, with one file per 10-minute window::

    transcripts/
      lecture-room/            <- source (like a ZoneMinder monitor)
        2026-10-21/
          09/
            00.txt             <- 09:00:00 to 09:09:59
            10.txt             <- 09:10:00 to 09:19:59

Each line inside a file starts with the wall-clock time the chunk started, so
text can be located to within a few seconds even though the file only
promises the 10-minute window. Times are local time, like ZoneMinder.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """Turn a free-text label (e.g. a device name) into a folder name."""
    cleaned = _SAFE.sub("-", name.strip()).strip("-.")
    return cleaned or "default"


def window_start(when: datetime, window_minutes: int = 10) -> datetime:
    minute = when.minute - (when.minute % window_minutes)
    return when.replace(minute=minute, second=0, microsecond=0)


def window_path(root: Path, source: str, when: datetime, window_minutes: int = 10) -> Path:
    start = window_start(when, window_minutes)
    return root / source / start.strftime("%Y-%m-%d") / start.strftime("%H") / f"{start:%M}.txt"


class TranscriptStore:
    def __init__(self, root: Path | str, source: str, window_minutes: int = 10,
                 header_info: str = "") -> None:
        if 60 % window_minutes != 0:
            raise ValueError("window_minutes must divide 60 (e.g. 5, 10, 15, 30)")
        self.root = Path(root)
        self.source = safe_name(source)
        self.window_minutes = window_minutes
        self.header_info = header_info

    def path_for(self, when: datetime) -> Path:
        return window_path(self.root, self.source, when, self.window_minutes)

    def append(self, when: datetime, text: str) -> Path | None:
        """Append one timestamped line to the window file covering ``when``."""
        text = " ".join(text.split())
        if not text:
            return None
        path = self.path_for(when)
        new_file = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            if new_file:
                f.write(self._header(when))
            f.write(f"[{when:%H:%M:%S}] {text}\n")
        return path

    def _header(self, when: datetime) -> str:
        start = window_start(when, self.window_minutes)
        end = start + timedelta(minutes=self.window_minutes)
        line = f"# source={self.source} window={start:%Y-%m-%d %H:%M}-{end:%H:%M}"
        if self.header_info:
            line += f" {self.header_info}"
        return line + "\n"
