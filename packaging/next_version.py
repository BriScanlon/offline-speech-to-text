"""Work out the next release version from git tags and the merge message.

Usage: python packaging/next_version.py "<commit message>" [tag ...]

Prints the next version (no leading v), or nothing if the message asks to
skip the release. The bump is a patch by default; put [minor] or [major] in
the PR title (or merge message) for a bigger one, or [skip release] to merge
without releasing. With no existing tags the first release is 0.1.0.
"""

import re
import sys

_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
FIRST = (0, 1, 0)


def next_version(message: str, tags: list[str]) -> str | None:
    msg = message.lower()
    if "[skip release]" in msg:
        return None
    versions = [tuple(int(x) for x in m.groups()) for t in tags if (m := _TAG.match(t.strip()))]
    if not versions:
        return "%d.%d.%d" % FIRST
    major, minor, patch = max(versions)
    if "[major]" in msg:
        major, minor, patch = major + 1, 0, 0
    elif "[minor]" in msg:
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


if __name__ == "__main__":
    print(next_version(sys.argv[1], sys.argv[2:]) or "")
