# PyInstaller spec for the Windows build: one folder with ostt.exe.
# Run from the repo root: pyinstaller packaging/ostt.spec
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# Native libraries and data files that import analysis doesn't find on its own:
# the CTranslate2 DLLs, faster-whisper's VAD model, and PortAudio for sounddevice.
for pkg in ("ctranslate2", "faster_whisper", "_sounddevice_data", "soxr"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["ostt_entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ostt",
    console=True,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ostt", upx=False)
