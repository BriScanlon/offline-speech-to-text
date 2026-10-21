import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "next_version", Path(__file__).parent.parent / "packaging" / "next_version.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
next_version = _mod.next_version


def test_first_release():
    assert next_version("Merge pull request #2", []) == "0.1.0"


def test_patch_bump_uses_highest_tag_not_latest_listed():
    assert next_version("Fix chunker", ["v0.1.9", "v0.1.10", "v0.1.2", "nightly"]) == "0.1.11"


def test_minor_and_major_markers():
    assert next_version("Add search UI [minor]", ["v0.1.4"]) == "0.2.0"
    assert next_version("New file format [MAJOR]", ["v0.3.2"]) == "1.0.0"


def test_skip_release():
    assert next_version("Docs only [skip release]", ["v0.1.0"]) is None
