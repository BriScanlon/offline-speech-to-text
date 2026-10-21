"""Write a WiX fragment listing every file in the PyInstaller output folder.

Usage: python packaging/make_files_wxs.py dist/ostt build/files.wxs

Each file gets its own component, placed in the matching subfolder of
INSTALLFOLDER. Ids are derived from the relative path so they are stable
between builds, which keeps upgrades clean.
"""

import hashlib
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr


def _id(prefix: str, rel: str) -> str:
    return prefix + hashlib.sha1(rel.lower().encode("utf-8")).hexdigest()[:24]


def main(src: str, out: str) -> None:
    root = Path(src).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found under {root}")
    lines = [
        '<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">',
        "  <Fragment>",
        '    <ComponentGroup Id="AppFiles" Directory="INSTALLFOLDER">',
    ]
    for path in files:
        rel = path.relative_to(root).as_posix()
        subdir = path.parent.relative_to(root).as_posix()
        sub_attr = "" if subdir == "." else f" Subdirectory={quoteattr(subdir.replace('/', chr(92)))}"
        lines.append(
            f'      <Component Id="{_id("c", rel)}"{sub_attr}>'
            f'<File Id="{_id("f", rel)}" Source={quoteattr(str(path))} KeyPath="yes" />'
            "</Component>"
        )
    lines += ["    </ComponentGroup>", "  </Fragment>", "</Wix>", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(files)} files to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
