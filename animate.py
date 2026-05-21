# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "matplotlib",
#   "numpy",
#   "moviepy<2.0",
#   "rich",
# ]
# ///

"""
stick_character_automation / animate.py

Reads a NotebookLM MP4 + transcript, groups into scenes with DeepSeek,
renders animated stick figure frames with matplotlib, and assembles a
final MP4 synced to the original audio. Zero image-gen cost.

Usage:
  uv run animate.py                            # auto-detects first MP4 in video/
  uv run animate.py --source video/my.mp4      # custom source
  uv run animate.py --transcript video/my.txt  # use specific transcript
  uv run animate.py --dry-run                  # preview scene plan, no rendering
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
FRAMES_DIR = OUTPUT_DIR / "anim_frames2"
FINAL_MP4  = OUTPUT_DIR / "animation.mp4"

DEEPSEEK_BASE  = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

FPS               = 24
TRANSITION_FRAMES = 12      # 0.5 s smooth transition between poses
HEAD_R            = 55      # head circle radius in canvas pixels

# Line widths (matplotlib points; at dpi=100 these read as pixels)
LW_BODY   = 4.5
LW_CRACK  = 3.0
LW_HEAD   = 4.5
LW_SHADOW = 7.0

# Brand palette (from innerwar_character_designs.svg)
C_WARRIOR = "#3C3489"   # purple  — Wounded Warrior
C_CRACK   = "#D85A30"   # coral   — crack / pain
C_SHADOW  = "#2C2C2A"   # charcoal — Shadow Self
C_GLOW    = "#F4874B"   # light orange — glow behind breakthrough crack

SHADOW_OFFSET = (130, 0)   # Shadow Self positioned to the right of Warrior

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        fallback = ROOT.parent / "videoSequence" / ".env"
        if fallback.exists():
            env_path = fallback
            console.print(f"[dim]Using .env from {fallback}[/]")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

def require_key(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        console.print(f"[red]ERROR:[/] {name} not set. Add it to .env")
        sys.exit(1)
    return val

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

TS_PATTERN = re.compile(
    r'\[(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\]\s*(.*)'
)

def ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def seconds_to_mmss(s: float) -> str:
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"

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

def parse_json_transcript(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("segments", [])

def load_transcript(path: Path) -> list[dict]:
    return parse_json_transcript(path) if path.suffix == ".json" else parse_txt(path)

# ---------------------------------------------------------------------------
# Character rig — all coordinates on 1920×1080 canvas (y=0 at bottom)
#
# figsize=(19.2, 10.8) at dpi=100 → 1920×1080 pixels
# xlim(0,1920) ylim(0,1080) → 1 data unit = 1 pixel, no distortion
# ---------------------------------------------------------------------------

# Joint keys: head, neck, hips, shoulder_L, shoulder_R, hand_L, hand_R, foot_L, foot_R
POSES: dict[str, dict[str, tuple[float, float]]] = {
    "standing_neutral": {
        "head":       (960, 800),
        "neck":       (960, 745),
        "hips":       (960, 545),
        "shoulder_L": (855, 705),
        "shoulder_R": (1065, 710),
        "hand_L":     (820, 605),
        "hand_R":     (1095, 640),
        "foot_L":     (915, 365),
        "foot_R":     (1005, 365),
    },
    "hunched": {
        # anxiety/stress — dropped uneven shoulders, arms lower
        "head":       (955, 778),
        "neck":       (955, 723),
        "hips":       (955, 540),
        "shoulder_L": (835, 678),
        "shoulder_R": (1045, 695),
        "hand_L":     (792, 568),
        "hand_R":     (1062, 608),
        "foot_L":     (915, 365),
        "foot_R":     (1000, 365),
    },
    "collapsed": {
        # heavy trauma — head bowed forward, arms hanging
        "head":       (940, 755),
        "neck":       (940, 700),
        "hips":       (950, 535),
        "shoulder_L": (825, 660),
        "shoulder_R": (1020, 672),
        "hand_L":     (780, 548),
        "hand_R":     (1005, 552),
        "foot_L":     (905, 365),
        "foot_R":     (995, 365),
    },
    "breakthrough": {
        # right arm raised high, head up
        "head":       (963, 808),
        "neck":       (963, 753),
        "hips":       (963, 548),
        "shoulder_L": (858, 712),
        "shoulder_R": (1072, 717),
        "hand_L":     (822, 613),
        "hand_R":     (1118, 820),
        "foot_L":     (910, 362),
        "foot_R":     (1015, 368),
    },
    "looking_down": {
        # numbness — head tilted down, arms loose
        "head":       (948, 788),
        "neck":       (948, 733),
        "hips":       (952, 538),
        "shoulder_L": (842, 694),
        "shoulder_R": (1050, 698),
        "hand_L":     (824, 598),
        "hand_R":     (1068, 608),
        "foot_L":     (912, 365),
        "foot_R":     (1000, 365),
    },
    "defensive": {
        # arms crossed (anxiety alt)
        "head":       (960, 800),
        "neck":       (960, 745),
        "hips":       (960, 545),
        "shoulder_L": (855, 705),
        "shoulder_R": (1065, 710),
        "hand_L":     (1008, 678),
        "hand_R":     (912, 672),
        "foot_L":     (915, 365),
        "foot_R":     (1005, 365),
    },
}

# ---------------------------------------------------------------------------
# Crack paths — (dx, dy) offsets from head center (dy positive = upward)
# Start point near (0, HEAD_R) = top of head; end point extends below
# ---------------------------------------------------------------------------

CRACKS: dict[str, list[tuple[float, float]]] = {
    "standard":       [(0, 55), (-5, 28), (3, 8), (-4, -18), (-5, -50)],
    "deep_jagged":    [(0, 55), (-10, 22), (8, 5), (-7, -22), (-9, -60)],
    "branching":      [(0, 55), (-5, 28), (4, 10), (-3, -8)],  # branches added in renderer
    "faint_stitched": [(0, 55), (-3, 25), (2, 8), (-2, -18), (-3, -48)],
    "glowing":        [(0, 55), (-10, 22), (8, 5), (-7, -22), (-9, -60)],  # same path + glow
    "barely_visible": [(0, 55), (-2, 20), (1, 5), (-1, -15), (-2, -45)],
}

# ---------------------------------------------------------------------------
# Emotional tone → (pose_name, crack_state, show_shadow)
# ---------------------------------------------------------------------------

TONE_MAP: dict[str, tuple[str, str, bool]] = {
    "heavy trauma":       ("collapsed",        "deep_jagged",    True),
    "anxiety/stress":     ("hunched",          "branching",      False),
    "growth/healing":     ("standing_neutral", "faint_stitched", False),
    "breakthrough":       ("breakthrough",     "glowing",        False),
    "numbness":           ("looking_down",     "barely_visible", True),
    "neutral":            ("standing_neutral", "standard",       False),
    "neutral/reflective": ("standing_neutral", "standard",       False),
}

def resolve_tone(tone: str) -> tuple[str, str, bool]:
    t = tone.lower().strip()
    if t in TONE_MAP:
        return TONE_MAP[t]
    for key in TONE_MAP:
        if key in t or t in key:
            return TONE_MAP[key]
    return TONE_MAP["neutral"]

# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _lerp_poses(pose_a: dict, pose_b: dict, t: float) -> dict:
    result = {}
    for joint in pose_a:
        ax, ay = pose_a[joint]
        bx, by = pose_b[joint]
        result[joint] = (ax + (bx - ax) * t, ay + (by - ay) * t)
    return result

def _draw_stick(ax, pose: dict, color: str, lw: float, alpha: float, filled: bool):
    hx, hy   = pose["head"]
    nx, ny   = pose["neck"]
    px, py   = pose["hips"]
    slx, sly = pose["shoulder_L"]
    srx, sry = pose["shoulder_R"]
    hlx, hly = pose["hand_L"]
    hrx, hry = pose["hand_R"]
    flx, fly = pose["foot_L"]
    frx, fry = pose["foot_R"]

    head_patch = plt.Circle(
        (hx, hy), HEAD_R,
        fill=filled, facecolor=color if filled else "white",
        edgecolor=color, linewidth=lw, alpha=alpha, zorder=3,
    )
    ax.add_patch(head_patch)

    for x1, y1, x2, y2 in [
        (nx, ny, px, py),       # body
        (slx, sly, srx, sry),   # shoulders
        (slx, sly, hlx, hly),   # left arm
        (srx, sry, hrx, hry),   # right arm
        (px, py, flx, fly),     # left leg
        (px, py, frx, fry),     # right leg
    ]:
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                solid_capstyle="round", alpha=alpha, zorder=3)

def _draw_crack(ax, hx: float, hy: float, crack_state: str, alpha: float):
    offsets = CRACKS.get(crack_state, CRACKS["standard"])
    xs = [hx + dx for dx, dy in offsets]
    ys = [hy + dy for dx, dy in offsets]

    # Glow behind crack for breakthrough
    if crack_state == "glowing":
        ax.plot(xs, ys, color=C_GLOW, linewidth=LW_CRACK * 3.5,
                solid_capstyle="round", alpha=alpha * 0.4, zorder=4)

    lw = LW_CRACK * 0.45 if crack_state == "barely_visible" else LW_CRACK
    a  = alpha * 0.35 if crack_state == "barely_visible" else alpha

    ax.plot(xs, ys, color=C_CRACK, linewidth=lw,
            solid_capstyle="round", alpha=a, zorder=5)

    # Spiderweb branches for anxiety
    if crack_state == "branching":
        bx, by = xs[2], ys[2]
        ex, ey = xs[3], ys[3]
        ax.plot([bx, bx + 14], [by, by - 4],
                color=C_CRACK, linewidth=lw * 0.7, solid_capstyle="round", alpha=a, zorder=5)
        ax.plot([ex, ex - 14], [ey, ey - 20],
                color=C_CRACK, linewidth=lw * 0.7, solid_capstyle="round", alpha=a, zorder=5)

    # Healing stitches across the crack
    if crack_state == "faint_stitched":
        for i in range(1, len(offsets) - 1, 2):
            cx = hx + offsets[i][0]
            cy = hy + offsets[i][1]
            ax.plot([cx - 8, cx + 8], [cy + 5, cy - 5],
                    color=C_CRACK, linewidth=1.0, alpha=a * 0.7, zorder=5)

def _draw_character(ax, pose: dict, crack_state: str, show_shadow: bool, alpha: float = 1.0):
    if show_shadow:
        shadow_pose = {
            k: (v[0] + SHADOW_OFFSET[0], v[1] + SHADOW_OFFSET[1])
            for k, v in pose.items()
        }
        _draw_stick(ax, shadow_pose, C_SHADOW, LW_SHADOW, alpha=0.35, filled=True)

    _draw_stick(ax, pose, C_WARRIOR, LW_BODY, alpha=alpha, filled=False)
    _draw_crack(ax, pose["head"][0], pose["head"][1], crack_state, alpha=alpha)

def render_frame(
    pose: dict, crack_state: str, show_shadow: bool,
    scene_text: str, out_path: Path, alpha: float = 1.0,
):
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _draw_character(ax, pose, crack_state, show_shadow, alpha=alpha)

    if scene_text:
        ax.text(960, 55, scene_text,
                ha="center", va="center", fontsize=20,
                color="#999999", style="italic")

    fig.savefig(str(out_path), dpi=100, facecolor="white")
    plt.close(fig)

def render_transition(
    pose_a: dict, pose_b: dict,
    crack_state: str, show_shadow: bool, scene_text: str,
    out_dir: Path, start_idx: int,
) -> list[Path]:
    paths = []
    for i in range(TRANSITION_FRAMES):
        t = (i + 1) / TRANSITION_FRAMES
        interp = _lerp_poses(pose_a, pose_b, t)
        p = out_dir / f"frame_{start_idx + i:06d}.png"
        render_frame(interp, crack_state, show_shadow, scene_text, p)
        paths.append(p)
    return paths

# ---------------------------------------------------------------------------
# Scene grouping — DeepSeek
# ---------------------------------------------------------------------------

GROUP_SYSTEM = """You are an animation director for TheInnerWar, a psychology YouTube channel.

Given a timestamped transcript, group consecutive segments into 20–30 visual scenes.
Each scene represents ONE distinct psychological concept or emotional beat.
Group segments that continue the same idea; start a new scene when the topic or
emotional register clearly shifts.

Emotional tone — pick the closest match:
  heavy trauma      → grief, deep pain, childhood wounds
  anxiety/stress    → worry, overthinking, pressure, stress
  growth/healing    → recovery, self-awareness, hope, positive change
  breakthrough      → sudden insight, realization, turning point
  numbness          → dissociation, emptiness, shutdown, avoidance
  neutral           → background, transitions, factual explanation

Respond with a JSON array ONLY — no markdown, no explanation:
[
  {
    "scene_index": 1,
    "start": 0.0,
    "end": 36.2,
    "concept": "short label (3–5 words)",
    "emotional_tone": "one of the six options above",
    "combined_text": "full text of all segments in this scene"
  }
]"""

def group_segments(segments: list[dict], deepseek_key: str) -> list[dict]:
    console.print("[bold]Step 1:[/] Grouping segments into scenes with DeepSeek...")

    lines = [
        f"[{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}] {s['text']}"
        for s in segments
    ]

    resp = httpx.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {deepseek_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": GROUP_SYSTEM},
                {"role": "user",   "content": f"Transcript ({len(segments)} segments):\n\n" + "\n".join(lines)},
            ],
            "temperature": 0.2,
            "max_tokens": 8192,
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        scenes = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        console.print(f"[red]ERROR:[/] DeepSeek JSON parse failed: {e}")
        console.print(f"[dim]Last 200 chars: ...{raw[-200:]}[/]")
        sys.exit(1)

    console.print(f"  [green]OK[/] {len(scenes)} scenes identified")
    return scenes

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(source_mp4: Path) -> Path:
    out = OUTPUT_DIR / "_audio_temp.aac"
    if out.exists():
        out.unlink()

    # Try stream copy first (fast, no re-encode)
    r = subprocess.run(
        ["ffmpeg", "-i", str(source_mp4), "-vn", "-acodec", "copy", str(out), "-y"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        # Fallback: re-encode to AAC
        r2 = subprocess.run(
            ["ffmpeg", "-i", str(source_mp4), "-vn", "-acodec", "aac", "-b:a", "192k", str(out), "-y"],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            console.print(f"[red]ERROR:[/] FFmpeg audio extraction failed:\n{r2.stderr[-500:]}")
            sys.exit(1)

    console.print(f"  [green]OK[/] Audio extracted")
    return out

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(scenes: list[dict]):
    console.print("\n[bold yellow]DRY RUN - No frames will be rendered[/]\n")

    table = Table(title=f"Animation Plan ({len(scenes)} scenes)", show_lines=True)
    table.add_column("#",       style="dim",      width=4)
    table.add_column("Time",    style="cyan",     width=13)
    table.add_column("Concept", style="bold",     width=26)
    table.add_column("Tone",    style="magenta",  width=20)
    table.add_column("Pose",    style="green",    width=18)
    table.add_column("Shadow",  style="yellow",   width=8)

    total_s = 0.0
    for s in scenes:
        pose, crack, shadow = resolve_tone(s.get("emotional_tone", "neutral"))
        dur = float(s["end"]) - float(s["start"])
        total_s += dur
        table.add_row(
            str(s["scene_index"]),
            f"{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}",
            s["concept"],
            s.get("emotional_tone", "neutral"),
            pose,
            "yes" if shadow else "-",
        )

    console.print(table)
    console.print(f"\n  Total duration : {seconds_to_mmss(total_s)}")
    console.print(f"  Transition cost: {len(scenes) * TRANSITION_FRAMES} frames @ {FPS}fps = {len(scenes) * TRANSITION_FRAMES / FPS:.1f}s")
    console.print(f"  DeepSeek cost  : ~$0.01 (grouping only)")
    console.print(f"  Image-gen cost : [green]$0.00[/] (fully code-rendered)")
    console.print("\nRun without [yellow]--dry-run[/] to render the animation.")

# ---------------------------------------------------------------------------
# Source / transcript auto-detection
# ---------------------------------------------------------------------------

def find_source_mp4() -> Path:
    video_dir = ROOT / "video"
    if video_dir.exists():
        mp4s = sorted(video_dir.glob("*.mp4"))
        if mp4s:
            return mp4s[0]
    console.print("[red]ERROR:[/] No MP4 found in video/. Use --source to specify one.")
    sys.exit(1)

def find_transcript(source_mp4: Path) -> Path:
    stem = source_mp4.stem
    for ext in (".txt", ".json"):
        p = source_mp4.parent / (stem + ext)
        if p.exists():
            return p
    console.print(f"[red]ERROR:[/] No transcript found for {source_mp4.name}.")
    console.print("Expected same folder, same stem, .txt or .json extension.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Render animated stick figure video from transcript")
    parser.add_argument("--source",     default=None, help="Path to source MP4")
    parser.add_argument("--transcript", default=None, help="Path to transcript (.txt or .json)")
    parser.add_argument("--dry-run",    action="store_true", help="Preview scene plan, no rendering")
    args = parser.parse_args()

    source_mp4 = Path(args.source) if args.source else find_source_mp4()
    if not source_mp4.exists():
        console.print(f"[red]ERROR:[/] Source not found: {source_mp4}")
        sys.exit(1)

    transcript_path = Path(args.transcript) if args.transcript else find_transcript(source_mp4)

    deepseek_key = require_key("DEEPSEEK_API_KEY")

    segments = load_transcript(transcript_path)
    console.print(f"[bold]Loaded[/] {len(segments)} segments from [cyan]{transcript_path.name}[/]")

    scenes = group_segments(segments, deepseek_key)

    if args.dry_run:
        dry_run(scenes)
        return

    # Late import — moviepy is slow; skip it entirely for --dry-run
    try:
        from moviepy.editor import (
            ImageSequenceClip, ImageClip,
            concatenate_videoclips, AudioFileClip,
        )
    except ImportError:
        console.print("[red]ERROR:[/] moviepy not found. Let uv install it: uv run animate.py")
        sys.exit(1)

    # Clean up previous frame run
    if FRAMES_DIR.exists():
        for f in FRAMES_DIR.glob("*.png"):
            f.unlink()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Step 2:[/] Rendering {len(scenes)} scenes...\n")

    all_clips   = []
    frame_idx   = 0
    prev_pose   = POSES["standing_neutral"]

    for scene in track(scenes, description="Rendering scenes..."):
        pose_name, crack_state, show_shadow = resolve_tone(scene.get("emotional_tone", "neutral"))
        curr_pose = POSES[pose_name]
        concept   = scene.get("concept", "")
        duration  = float(scene["end"]) - float(scene["start"])

        # 0.5 s transition — interpolated frames
        trans_paths = render_transition(
            prev_pose, curr_pose, crack_state, show_shadow,
            concept, FRAMES_DIR, frame_idx,
        )
        frame_idx += len(trans_paths)

        # Hold frame — single PNG, held for remainder of scene duration
        hold_path = FRAMES_DIR / f"frame_{frame_idx:06d}_hold.png"
        render_frame(curr_pose, crack_state, show_shadow, concept, hold_path)
        frame_idx += 1

        hold_duration = max(0.1, duration - TRANSITION_FRAMES / FPS)

        trans_clip = ImageSequenceClip([str(p) for p in trans_paths], fps=FPS)
        hold_clip  = ImageClip(str(hold_path)).set_duration(hold_duration)
        all_clips.extend([trans_clip, hold_clip])

        console.print(
            f"  [green]OK[/] [{seconds_to_mmss(scene['start'])}] "
            f"[bold]{concept}[/] -> [cyan]{pose_name}[/]"
            + (" [dim](+ shadow)[/]" if show_shadow else "")
        )

        prev_pose = curr_pose

    console.print("\n[bold]Step 3:[/] Assembling video...")
    video = concatenate_videoclips(all_clips, method="compose")

    console.print("[bold]Step 4:[/] Extracting audio...")
    audio_path = extract_audio(source_mp4)
    audio      = AudioFileClip(str(audio_path))

    # Trim to shorter of the two (handles slight duration mismatches)
    final_dur = min(video.duration, audio.duration)
    video = video.subclip(0, final_dur)
    audio = audio.subclip(0, final_dur)
    final = video.set_audio(audio)

    console.print(f"[bold]Step 5:[/] Writing [cyan]{FINAL_MP4.name}[/] ...")
    final.write_videofile(
        str(FINAL_MP4),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_DIR / "_temp_mux.m4a"),
        remove_temp=True,
        logger=None,
    )

    # Close clips to release file handles before deleting temp audio (Windows requires this)
    final.close()
    audio.close()
    video.close()

    if audio_path.exists():
        try:
            audio_path.unlink()
        except PermissionError:
            pass  # not critical — temp file, OS will clean it up

    console.print(f"\n[bold green]Done.[/] -> [cyan]{FINAL_MP4}[/]")
    console.print(f"  Duration : {seconds_to_mmss(final_dur)}")
    console.print(f"  Scenes   : {len(scenes)}")
    console.print(f"  Frames   : {frame_idx} PNGs")


if __name__ == "__main__":
    main()
