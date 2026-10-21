"""Command line: ``ostt devices | levels | run | test | search``."""

import argparse
import csv
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

DEFAULT_OUT = "transcripts"


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-d", "--device", help="input device index or part of its name (default: system default)")
    p.add_argument("--channel", type=int, help="use only this input channel, 0-based (e.g. 1 for a mic on the right input)")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", default="small",
                   help="tiny, base, small, medium, large-v3, or a local model folder (default: small)")
    p.add_argument("--language", default="en", help="spoken language code, or 'auto' to detect (default: en)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for inference (0 = automatic)")
    p.add_argument("--compute-type", default="int8", help="int8 (default), int8_float32, float32")
    p.add_argument("--beam-size", type=int, default=5, help="lower is faster, higher is more accurate (default: 5)")
    p.add_argument("--offline", action="store_true",
                   help="never touch the network; the model must already be downloaded")


def _load_transcriber(args):
    from offline_stt.transcriber import Transcriber

    print(f"Loading model '{args.model}'...", file=sys.stderr)
    return Transcriber(
        model=args.model,
        cpu_threads=args.threads,
        compute_type=args.compute_type,
        language=None if args.language == "auto" else args.language,
        beam_size=args.beam_size,
        offline=args.offline,
    )


def cmd_devices(args) -> int:
    from offline_stt.audio import list_input_devices

    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return 1
    for d in devices:
        mark = "*" if d.is_default else " "
        print(f"{mark} [{d.index:>2}] {d.name}  ({d.host_api}, {d.channels} ch, {d.default_rate} Hz)")
    print("\n* = system default. Pass the number or part of the name to --device.")
    return 0


def cmd_levels(args) -> int:
    from offline_stt.audio import MicStream, device_name, resolve_device
    from offline_stt.levels import CLIP_THRESHOLD, meter_bar, peak_db, rms_db

    dev = resolve_device(args.device)
    print(f"Level meter for: {device_name(dev)}  (Ctrl+C to stop)")
    print("Aim for speech peaks around -12 to -6 dBFS with no CLIP. Below -40 dB RMS is too quiet.\n")
    peak_hold, clips = -120.0, 0
    with MicStream(dev, args.channel) as mic:
        try:
            while True:
                block = mic.read()
                if block is None or len(block) == 0:
                    continue
                r, p = rms_db(block), peak_db(block)
                peak_hold = max(peak_hold, p)
                if np.max(np.abs(block)) >= CLIP_THRESHOLD:
                    clips += 1
                flag = " CLIP" if p > -0.1 else "     "
                sys.stdout.write(f"\r{meter_bar(r)}  rms {r:6.1f} dB  peak {p:6.1f} dB{flag}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
    print(f"\n\nHighest peak: {peak_hold:.1f} dBFS, blocks clipped: {clips}")
    return 0


def cmd_run(args) -> int:
    from offline_stt.audio import MicStream, device_name, resolve_device
    from offline_stt.chunker import Chunker
    from offline_stt.pipeline import LiveSession
    from offline_stt.store import TranscriptStore, safe_name

    dev = resolve_device(args.device)
    name = device_name(dev)
    source = safe_name(args.source or name)
    transcriber = _load_transcriber(args)
    store = TranscriptStore(args.out, source, args.window,
                            header_info=f"device=\"{name}\" model={args.model}")
    chunker = Chunker(min_seconds=args.min_chunk, max_seconds=args.max_chunk,
                      pause_seconds=args.pause, silence_db=args.silence_db)

    def show(when, result, path):
        if not args.quiet:
            print(f"[{when:%H:%M:%S}] {result.text}")
        if result.ratio > 1.0:
            print(f"  (warning: this chunk took {result.ratio:.1f}x real time; "
                  "a smaller --model will keep up better)", file=sys.stderr)

    print(f"Listening on: {name}", file=sys.stderr)
    print(f"Writing to:   {Path(args.out) / source}/YYYY-MM-DD/HH/MM.txt "
          f"({args.window}-minute files). Audio is not saved. Ctrl+C to stop.\n", file=sys.stderr)

    with MicStream(dev, args.channel) as mic:
        session = LiveSession(chunker, transcriber, store, started_at=datetime.now(), on_text=show)
        try:
            while True:
                block = mic.read()
                if block is not None and len(block):
                    session.feed(block)
        except KeyboardInterrupt:
            print("\nStopping, finishing the last chunk...", file=sys.stderr)
        overflows = mic.overflows
    session.close()

    ratio = session.infer_seconds / session.audio_seconds if session.audio_seconds else 0
    print(f"Done. {session.chunks_written} chunks written, "
          f"{session.audio_seconds:.0f}s of speech transcribed at {ratio:.2f}x real time.",
          file=sys.stderr)
    if overflows:
        print(f"Note: the sound card dropped audio {overflows} time(s).", file=sys.stderr)
    return 0


def cmd_test(args) -> int:
    from offline_stt.audio import MicStream, device_name, resolve_device
    from offline_stt.levels import summarise
    from offline_stt.wer import word_error_rate

    reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else ""
    dev = resolve_device(args.device)
    name = device_name(dev)
    transcriber = _load_transcriber(args)

    print(f"\nMic test on: {name}")
    if args.label:
        print(f"Setup: {args.label}")
    if reference:
        print("\nRead this aloud at a normal pace:\n")
        print(reference.strip(), "\n")
    for n in (3, 2, 1):
        print(f"Starting in {n}...", end="\r", flush=True)
        time.sleep(1)

    blocks = []
    with MicStream(dev, args.channel) as mic:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            left = end - time.monotonic()
            print(f"Recording... {left:4.0f}s left (Ctrl+C to finish early)", end="\r", flush=True)
            try:
                block = mic.read()
            except KeyboardInterrupt:
                break
            if block is not None:
                blocks.append(block)
    print(" " * 60, end="\r")
    audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
    blocks.clear()
    if len(audio) == 0:
        print("No audio was captured.")
        return 1

    stats = summarise(audio, 16000)
    result = transcriber.transcribe(audio)
    del audio  # the recording is only kept long enough to score it
    wer = word_error_rate(reference, result.text) if reference else None

    print(f"Transcript: {result.text}\n")
    print(f"Peak {stats.peak_db} dBFS | speech {stats.speech_db} dB | noise floor "
          f"{stats.noise_floor_db} dB | SNR ~{stats.snr_db} dB | clipped {stats.clip_pct}%")
    if wer is not None:
        print(f"Word error rate: {wer:.1%} (lower is better)")
    print(f"Speed: {result.ratio:.2f}x real time")
    print(f"Level check: {stats.verdict()}")

    csv_path = Path(args.csv)
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "device", "channel", "label", "model", "seconds",
                        "peak_db", "speech_db", "noise_floor_db", "snr_db", "clip_pct",
                        "wer", "speed_ratio", "verdict", "transcript"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name,
                    "" if args.channel is None else args.channel, args.label or "", args.model,
                    round(result.audio_seconds, 1), stats.peak_db, stats.speech_db,
                    stats.noise_floor_db, stats.snr_db, stats.clip_pct,
                    "" if wer is None else round(wer, 4), round(result.ratio, 3),
                    stats.verdict(), result.text])
    print(f"\nLogged to {csv_path}")
    return 0


def cmd_search(args) -> int:
    from offline_stt.search import search

    hits = search(
        args.out, args.pattern,
        source=args.source, since=args.since, until=args.until,
        regex=args.regex, ignore_case=not args.case_sensitive,
    )
    for h in hits:
        print(f"{h.when:%Y-%m-%d %H:%M:%S}  {h.source}  {h.text}")
    print(f"\n{len(hits)} match(es).", file=sys.stderr)
    return 0 if hits else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ostt", description="Offline live speech-to-text. Audio is never saved; only text is.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list microphones and other inputs").set_defaults(func=cmd_devices)

    p = sub.add_parser("levels", help="live input level meter for setting mic gain and placement")
    _add_input_args(p)
    p.set_defaults(func=cmd_levels)

    p = sub.add_parser("run", help="transcribe the mic continuously into 10-minute text files")
    _add_input_args(p)
    _add_model_args(p)
    p.add_argument("-s", "--source", help="folder name for this recording source (default: from the device name)")
    p.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"root folder for transcripts (default: {DEFAULT_OUT})")
    p.add_argument("--window", type=int, default=10, help="minutes per text file (default: 10)")
    p.add_argument("--min-chunk", type=float, default=5.0, help="shortest chunk in seconds (default: 5)")
    p.add_argument("--max-chunk", type=float, default=30.0, help="longest chunk in seconds (default: 30)")
    p.add_argument("--pause", type=float, default=0.6, help="pause in seconds that ends a chunk (default: 0.6)")
    p.add_argument("--silence-db", type=float, default=-45.0,
                   help="level below which audio counts as silence (default: -45 dBFS)")
    p.add_argument("-q", "--quiet", action="store_true", help="don't echo text to the terminal")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("test", help="record a short test, score it, and log the result to a CSV")
    _add_input_args(p)
    _add_model_args(p)
    p.add_argument("-r", "--reference", help="text file with the passage you will read (enables word error rate)")
    p.add_argument("-l", "--label", help="describe the setup, e.g. 'USB headset, 30cm, gain 70%%'")
    p.add_argument("--seconds", type=float, default=30.0, help="recording length (default: 30)")
    p.add_argument("--csv", default="mic_tests.csv", help="results log (default: mic_tests.csv)")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("search", help="search saved transcripts")
    p.add_argument("pattern", help="phrase to find (or a regular expression with --regex)")
    p.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"root folder for transcripts (default: {DEFAULT_OUT})")
    p.add_argument("-s", "--source", help="only this source folder")
    p.add_argument("--since", type=date.fromisoformat, help="from this date, YYYY-MM-DD")
    p.add_argument("--until", type=date.fromisoformat, help="up to and including this date, YYYY-MM-DD")
    p.add_argument("--regex", action="store_true", help="treat the pattern as a regular expression")
    p.add_argument("--case-sensitive", action="store_true")
    p.set_defaults(func=cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
