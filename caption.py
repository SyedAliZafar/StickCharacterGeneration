# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich",
# ]
# ///

"""
caption.py

Convert a timestamped transcript into an SRT file for CapCut import.

CapCut: Captions → Import → select the .srt file

Usage:
  uv run caption.py                          # auto-detect transcript in video/
  uv run caption.py --source video/my.txt    # specific transcript
  uv run caption.py --out output/caps.srt    # custom output path
  uv run caption.py --merge 2.0              # merge segments shorter than 2s into neighbours
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Console

console = Console()

ROOT = Path(__file__).parent

TS_PATTERN = re.compile(
    r'\[(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\]\s*(.*)'
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_srt(s: float) -> str:
    ms = int(round(s * 1000))
    h  = ms // 3_600_000; ms %= 3_600_000
    m  = ms // 60_000;    ms %= 60_000
    sc = ms // 1_000;     ms %= 1_000
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"


def parse_txt(path: Path) -> list[dict]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = TS_PATTERN.match(line.strip())
        if m and m.group(3).strip():
            segments.append({
                "start": ts_to_seconds(m.group(1)),
                "end":   ts_to_seconds(m.group(2)),
                "text":  m.group(3).strip(),
            })
    return segments


def parse_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data if isinstance(data, list) else data.get("segments", [])
    return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
            for s in segs if s.get("text", "").strip()]


def load_transcript(path: Path) -> list[dict]:
    return parse_json(path) if path.suffix == ".json" else parse_txt(path)


def find_transcript() -> Path:
    video_dir = ROOT / "video"
    if video_dir.exists():
        for ext in (".txt", ".json"):
            found = sorted(video_dir.glob(f"*{ext}"))
            if found:
                return found[0]
    console.print("[red]ERROR:[/] No transcript found in video/. Use --source to specify one.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Merging short segments
# ---------------------------------------------------------------------------

def merge_short(segments: list[dict], min_dur: float) -> list[dict]:
    """Merge any segment shorter than min_dur seconds into the next one."""
    if not segments:
        return segments
    merged = []
    buf = segments[0].copy()
    for seg in segments[1:]:
        dur = buf["end"] - buf["start"]
        if dur < min_dur:
            buf["end"]  = seg["end"]
            buf["text"] = buf["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(buf)
            buf = seg.copy()
    merged.append(buf)
    return merged

# ---------------------------------------------------------------------------
# SRT writing
# ---------------------------------------------------------------------------

def write_srt(segments: list[dict], out_path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt(seg['start'])} --> {seconds_to_srt(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SRT captions from transcript")
    parser.add_argument("--source", default=None, help="Transcript .txt or .json (auto-detected if omitted)")
    parser.add_argument("--out",    default=None, help="Output .srt path (default: same folder as transcript)")
    parser.add_argument("--merge",  type=float,   default=0.0,
                        help="Merge segments shorter than N seconds into the next (e.g. --merge 1.5)")
    args = parser.parse_args()

    transcript = Path(args.source) if args.source else find_transcript()
    if not transcript.exists():
        console.print(f"[red]ERROR:[/] File not found: {transcript}")
        sys.exit(1)

    segments = load_transcript(transcript)
    console.print(f"  Loaded [cyan]{len(segments)}[/] segments from [cyan]{transcript.name}[/]")

    if args.merge > 0:
        before = len(segments)
        segments = merge_short(segments, args.merge)
        console.print(f"  Merged → [cyan]{len(segments)}[/] captions (was {before}, min_dur={args.merge}s)")

    out_path = Path(args.out) if args.out else transcript.with_suffix(".srt")
    write_srt(segments, out_path)

    console.print(f"\n  [green]SRT saved → {out_path}[/]")
    console.print(f"  [dim]CapCut: Captions → Import → select this file[/]")


if __name__ == "__main__":
    main()
