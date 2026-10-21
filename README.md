# Offline speech-to-text

A digital dictaphone that writes text instead of audio. It listens to a
locally connected microphone, transcribes speech in short chunks as it goes,
and saves timestamped text files in 10-minute blocks. It runs fully offline,
and **audio is never written to disk**: each chunk is held in memory only
until it has been transcribed.

Typical uses: capturing a lecture, or self-reporting while on a project, then
searching the text later.

```
microphone ──► 100 ms blocks ──► chunker (cuts at pauses, 5–30 s) ──► Whisper ──► text files
   (any input:      in memory        drops silence, adapts to          (faster-whisper,
   webcam, USB,                      room noise                         CPU, int8)
   XLR via sound card)
```

## What it does

| Requirement | How it's met |
| --- | --- |
| Local processing, works offline | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU. After the model is downloaded once, `--offline` guarantees no network use. |
| Don't keep the audio | Audio only exists in memory. Each chunk is dropped once its text is written. Nothing in the code writes audio to disk. |
| Live mic capture in small chunks | `ostt run` streams the mic and cuts it at natural pauses (5–30 s chunks) so words aren't split. |
| Timestamped text in 10-minute windows | One file per 10-minute window. Each line also starts with the time its chunk began. |
| ZoneMinder-style folders | `transcripts/<source>/<YYYY-MM-DD>/<HH>/<MM>.txt`, where the source works like a ZoneMinder monitor (one per mic or room). |
| Testing mics, volume and placement | `ostt levels` is a live meter. `ostt test` records a known passage and logs level, noise, clipping, word error rate and speed to a CSV for comparison. |
| Searchable | `ostt search` finds phrases across all sources and dates. The files are plain text, so `grep`, Windows search, etc. also work. |

### Output layout

```
transcripts/
  lecture-room/
    2026-10-21/
      09/
        00.txt      09:00–09:09
        10.txt      09:10–09:19
```

```
# source=lecture-room window=2026-10-21 09:10-09:20 device="USB Audio Device" model=small
[09:10:04] Right, let's pick up where we left off with Fourier series.
[09:10:31] The key idea is that any periodic signal can be written as a sum of sines.
```

An hour of speech is roughly 50 KB of text.

## Install

**Windows installer (easiest):** download `offline-speech-to-text-X.Y.Z-x64.msi`
from the [Releases](../../releases) page and run it. It installs to
`C:\Program Files\Offline Speech to Text` and adds `ostt` to the PATH, so open
a new terminal and run `ostt selfcheck` to confirm it works. No Python needed.

**From source** needs Python 3.10 or newer.

**Windows (ThinkPad)**

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -e .
```

**Linux / Raspberry Pi OS**

```bash
sudo apt install libportaudio2      # microphone access for sounddevice
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

The first `run` or `test` downloads the chosen Whisper model (about 75 MB for
`tiny`, 480 MB for `small`) into the Hugging Face cache. Do that once while
online. From then on, add `--offline` (or set `HF_HUB_OFFLINE=1`) and the
machine can be disconnected.

## Use

```bash
ostt devices                         # list inputs; note the number or name
ostt levels -d 2                     # live meter to set gain and placement
ostt run -d 2 -s lecture-room        # transcribe until Ctrl+C
ostt search "fourier" --since 2026-10-01
```

`--device` takes the number from `ostt devices` or any part of the name
(`-d "USB"`). Without it, the system default input is used. For a sound card
where the XLR mic only arrives on one side, pick it with `--channel 0` (left)
or `--channel 1` (right).

`ostt run` options worth knowing:

| Option | Default | Notes |
| --- | --- | --- |
| `-m/--model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`. Bigger is more accurate and slower. |
| `-s/--source` | from device name | Top-level folder, like a ZoneMinder monitor. |
| `-o/--out` | `transcripts` | Root folder. |
| `--window` | `10` | Minutes per file (must divide 60). |
| `--language` | `en` | `auto` detects per chunk but is slower and less reliable. |
| `--threads` | auto | Limit CPU use, e.g. on a Pi that is doing other work. |
| `--min-chunk` / `--max-chunk` / `--pause` | 5 / 30 / 0.6 s | How audio is cut into chunks. |
| `--silence-db` | -45 | Quietest level that can count as speech. The threshold also rises with room noise. |
| `--offline` | off | Refuse any network access. |

If a chunk takes longer to transcribe than it took to say, the tool warns you.
That means the model is too big for the machine, so drop a size.

## Finding the best mic and placement

1. `ostt levels -d <mic>` and adjust the OS input volume (or the sound card
   gain) until normal speech peaks around **-12 to -6 dBFS** and never shows
   `CLIP`. If speech sits below about -40 dB, it's too quiet.
2. Run a scored test with the included Harvard sentences, a standard set of
   phonetically balanced test sentences:

   ```bash
   ostt test -d <mic> -r reference/harvard_sentences.txt \
       -l "USB headset, 3 cm, input 70%"
   ```

   It prints the transcript, peak and speech level, noise floor, rough SNR,
   clipping, **word error rate** (WER, lower is better) and speed, then adds a
   row to `mic_tests.csv`. The recording is discarded after scoring.
3. Repeat for each mic, volume setting and position, changing the `-l` label
   each time. Open `mic_tests.csv` in a spreadsheet and sort by `wer`.

Use the same model for every test so the comparison is fair. WER under about
10% is good for lecture notes.

## Hardware notes

**ThinkPad P16s Gen 2 (Ryzen 7 PRO 7840U, 64 GB).** Everything runs on the 8
CPU cores. `small` should run well under real time and is the default.
`medium` is worth trying for accuracy. `large-v3` may struggle to keep up live.
Check the speed column in `mic_tests.csv` to see for certain. faster-whisper
can't use the Radeon 780M or the NPU. Using them would mean swapping the
engine: whisper.cpp with its Vulkan backend for the iGPU, or AMD's Ryzen AI
software (ONNX, Windows) for the NPU. That's a possible next step, not needed
for this PoC.

**Raspberry Pi 5 (8 GB).** Works with the same code (faster-whisper ships ARM64
wheels). Start with `-m base` or `-m tiny`. `small` will probably fall behind
during continuous speech. Use a USB mic or a USB sound card, since the Pi has
no analogue mic input. An AI HAT could take the load off the CPU later, but
would need a Whisper build for that accelerator.

These speed expectations are estimates. `ostt test` reports the real figure
on each machine.

## Privacy

- No audio files are ever written. Audio is kept only in memory, for up to one
  chunk (30 s by default), and dropped once transcribed. `ostt test` holds its
  single recording only until it has been scored.
- No network use once the model is cached (enforce with `--offline`).
- Transcripts and `mic_tests.csv` are plain text on the local disk, and
  `.gitignore` keeps them out of git.

## Development and releases

`main` is protected. Changes go through a pull request, and the **CI** workflow
must pass before merging. It runs the tests on Linux (Python 3.10 and 3.12) and
Windows, then builds the MSI, installs it on a clean Windows runner and runs
`ostt selfcheck` from the installed copy. Each PR run keeps its MSI as a
downloadable artifact.

To publish a release, tag a commit on `main`:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The **Release** workflow runs the tests again, builds
`offline-speech-to-text-0.2.0-x64.msi`, and creates a GitHub release with
the MSI attached and notes generated from the merged PRs.

Packaging lives in `packaging/`. PyInstaller freezes `ostt.exe` with its
native libraries (`ostt.spec`), and WiX 5 wraps that folder in the MSI
(`ostt.wxs` plus a file list generated by `make_files_wxs.py`). The MSI does
not include a Whisper model. The first `ostt run` or `ostt test` downloads
one, unless a model folder is passed with `-m`.

To run the tests locally:

```bash
pip install -e '.[test]'
python -m pytest
```

The tests cover chunking, file layout, search, level maths, WER and the
capture-to-file pipeline (with a stub model). They need neither a microphone
nor a model download.
