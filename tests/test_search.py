from datetime import date, datetime

from offline_stt.search import search
from offline_stt.store import TranscriptStore


def _populate(root):
    a = TranscriptStore(root, "lecture")
    a.append(datetime(2026, 10, 20, 14, 3, 0), "Today we cover the Fourier transform.")
    a.append(datetime(2026, 10, 21, 9, 12, 30), "Recap of the fourier series.")
    b = TranscriptStore(root, "site-diary")
    b.append(datetime(2026, 10, 21, 16, 45, 1), "Concrete pour delayed by rain.")


def test_phrase_search_is_case_insensitive_across_sources(tmp_path):
    _populate(tmp_path)
    hits = search(tmp_path, "fourier")
    assert [h.when for h in hits] == [datetime(2026, 10, 20, 14, 3, 0), datetime(2026, 10, 21, 9, 12, 30)]
    assert all(h.source == "lecture" for h in hits)


def test_filters_by_source_and_date(tmp_path):
    _populate(tmp_path)
    assert search(tmp_path, "fourier", since=date(2026, 10, 21))[0].text.startswith("[09:12:30]")
    assert search(tmp_path, "concrete", source="lecture") == []
    assert len(search(tmp_path, "concrete", until=date(2026, 10, 21))) == 1


def test_regex_and_headers_ignored(tmp_path):
    _populate(tmp_path)
    assert len(search(tmp_path, r"pour|cover", regex=True)) == 2
    assert search(tmp_path, "source=") == []
