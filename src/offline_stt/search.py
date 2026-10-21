"""Search the transcript tree by text, source and date range."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    path: Path
    source: str
    when: datetime
    text: str


def _parse_window(path: Path, root: Path) -> tuple[str, datetime] | None:
    """Read source and window start from ``<source>/<YYYY-MM-DD>/<HH>/<MM>.txt``."""
    try:
        source, day, hour, minute_file = path.relative_to(root).parts
        start = datetime.strptime(f"{day} {hour}:{minute_file[:-4]}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return source, start


def _line_time(line: str, window: datetime) -> datetime:
    m = re.match(r"\[(\d\d):(\d\d):(\d\d)\]", line)
    if not m:
        return window
    h, mi, s = (int(x) for x in m.groups())
    return window.replace(hour=h, minute=mi, second=s)


def search(
    root: Path | str,
    pattern: str,
    *,
    source: str | None = None,
    since: date | None = None,
    until: date | None = None,
    regex: bool = False,
    ignore_case: bool = True,
) -> list[Hit]:
    """Find transcript lines matching ``pattern``, oldest first.

    ``pattern`` is a plain phrase unless ``regex`` is set. ``since`` and
    ``until`` are inclusive dates.
    """
    root = Path(root)
    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(pattern if regex else re.escape(pattern), flags)

    hits: list[Hit] = []
    for path in sorted(root.glob("*/*/*/*.txt")):
        parsed = _parse_window(path, root)
        if parsed is None:
            continue
        src, window = parsed
        if source and src != source:
            continue
        if since and window.date() < since:
            continue
        if until and window.date() > until:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not rx.search(line):
                continue
            hits.append(Hit(path=path, source=src, when=_line_time(line, window), text=line))
    return hits
