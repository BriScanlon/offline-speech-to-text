from datetime import datetime

import pytest

from offline_stt.store import TranscriptStore, safe_name, window_path


def test_window_path_follows_date_tree(tmp_path):
    when = datetime(2026, 10, 21, 9, 17, 42)
    assert window_path(tmp_path, "room", when) == tmp_path / "room" / "2026-10-21" / "09" / "10.txt"


def test_append_writes_header_once_and_timestamps_lines(tmp_path):
    store = TranscriptStore(tmp_path, "Lecture Hall (USB)", header_info="model=small")
    p1 = store.append(datetime(2026, 10, 21, 9, 0, 5), "hello   there")
    p2 = store.append(datetime(2026, 10, 21, 9, 9, 59), "second line")
    p3 = store.append(datetime(2026, 10, 21, 9, 10, 0), "next window")
    assert p1 == p2 != p3
    assert p1.parent.parent.parent.name == "Lecture-Hall-USB"
    assert p1.read_text().splitlines() == [
        "# source=Lecture-Hall-USB window=2026-10-21 09:00-09:10 model=small",
        "[09:00:05] hello there",
        "[09:09:59] second line",
    ]
    assert p3.name == "10.txt"


def test_blank_text_writes_nothing(tmp_path):
    store = TranscriptStore(tmp_path, "x")
    assert store.append(datetime(2026, 1, 1), "   ") is None
    assert not any(tmp_path.iterdir())


def test_window_must_divide_hour(tmp_path):
    with pytest.raises(ValueError):
        TranscriptStore(tmp_path, "x", window_minutes=7)


def test_safe_name():
    assert safe_name("  ") == "default"
    assert safe_name("Microphone Array (Realtek(R) Audio)") == "Microphone-Array-Realtek-R-Audio"
