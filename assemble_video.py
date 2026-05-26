# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "numpy",
#   "rich",
# ]
# ///

"""
assemble_video.py

Assemble a video from pre-generated scene PNGs without calling any image API.
Reads scenes.json (saved by test_animate.py --full-run) for tone data.

Usage:
  uv run assemble_video.py                              # still frames, no subtitles (default)
  uv run assemble_video.py --subtitles                  # burn keyword captions onto frames
  uv run assemble_video.py --motion smooth              # gentle ease-in/out zoom
  uv run assemble_video.py --motion cinematic           # dramatic zoom
  uv run assemble_video.py --motion low                 # minimal movement
  uv run assemble_video.py --motion normal              # original speed
  uv run assemble_video.py --motion still               # no movement (default)
  uv run assemble_video.py --frames output/full_run/frames/
  uv run assemble_video.py --source video/my.mp4
  uv run assemble_video.py --out output/my_video.mp4
  uv run assemble_video.py --thumbnail                  # thumbnail only, no video
"""

import argparse
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rich.console import Console

console = Console()

ROOT       = Path(__file__).parent
FPS        = 24
PAPER_TONE = (245, 241, 232)  # #F5F1E8

# ---------------------------------------------------------------------------
# Motion mode parameters
# ---------------------------------------------------------------------------

MOTION_PARAMS: dict[str, dict] = {
    "still":     {"zoom_scale": 0.0, "jitter_scale": 0, "fade_n": 18, "crack_n": 12},
    "low":       {"zoom_scale": 0.3, "jitter_scale": 0, "fade_n": 18, "crack_n": 12},
    "smooth":    {"zoom_scale": 0.6, "jitter_scale": 0, "fade_n": 30, "crack_n": 18},
    "normal":    {"zoom_scale": 1.0, "jitter_scale": 1, "fade_n": 18, "crack_n": 12},
    "cinematic": {"zoom_scale": 1.4, "jitter_scale": 2, "fade_n": 24, "crack_n": 14},
}

# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def _ease(t: float) -> float:
    """Smoothstep — ease-in-out so motion starts/ends gently."""
    return t * t * (3 - 2 * t)

# ---------------------------------------------------------------------------
# Transcript / scene helpers
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

def _parse_txt(path: Path) -> list[dict]:
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

def _parse_json_transcript(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("segments", [])

def load_transcript(path: Path) -> list[dict]:
    return _parse_json_transcript(path) if path.suffix == ".json" else _parse_txt(path)

def find_source_mp4() -> Path:
    video_dir = ROOT / "video"
    if video_dir.exists():
        mp4s = sorted(video_dir.glob("*.mp4"))
        if mp4s:
            return mp4s[0]
    console.print("[red]ERROR:[/] No MP4 found in video/. Use --source to specify one.")
    sys.exit(1)

def find_transcript(source_mp4: Path) -> Path | None:
    stem = source_mp4.stem
    for ext in (".txt", ".json"):
        p = source_mp4.parent / (stem + ext)
        if p.exists():
            return p
    return None

# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scenes(frames_dir: Path) -> list[dict]:
    """Load scenes.json from the frames dir's parent, or build neutral fallback."""
    scenes_path = frames_dir.parent / "scenes.json"
    if scenes_path.exists():
        try:
            scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
            console.print(f"  [green]OK[/] Loaded {len(scenes)} scenes from [cyan]{scenes_path.name}[/]")
            return scenes
        except Exception as e:
            console.print(f"  [yellow]WARN[/] Could not parse scenes.json: {e} — using neutral fallback")

    frame_files = sorted(frames_dir.glob("scene_*.png"))
    if not frame_files:
        console.print(f"[red]ERROR:[/] No scene_*.png files found in {frames_dir}")
        sys.exit(1)

    console.print(f"  [yellow]No scenes.json found[/] — generating neutral scene list for {len(frame_files)} frames")
    scenes = []
    t = 0.0
    for i, f in enumerate(frame_files, start=1):
        duration = 4.0
        scenes.append({
            "scene_index":    i,
            "start":          round(t, 3),
            "end":            round(t + duration, 3),
            "emotional_tone": "neutral",
            "concept":        f.stem,
            "combined_text":  "",
            "visual_description": "",
            "show_shadow":    False,
        })
        t += duration
    return scenes

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(source_mp4: Path, out_dir: Path) -> Path:
    out = out_dir / "_audio_temp.aac"
    if out.exists():
        out.unlink()
    r = subprocess.run(
        ["ffmpeg", "-i", str(source_mp4), "-vn", "-acodec", "copy", str(out), "-y"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
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
# Intensity scoring
# ---------------------------------------------------------------------------

def intensity_score(scene_index: int, total_scenes: int) -> float:
    if total_scenes <= 1:
        return 0.0
    return (scene_index - 1) / (total_scenes - 1)

# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------

def apply_overlays(img_pil: Image.Image, tone: str, intensity: float = 0.0,
                   motion: str = "smooth") -> Image.Image:
    arr = np.array(img_pil, dtype=np.float32)
    H, W = arr.shape[:2]

    grain_std = 8.0 + intensity * 6.0
    rng = np.random.default_rng()
    arr = np.clip(arr + rng.normal(0, grain_std, arr.shape), 0, 255)

    if motion == "cinematic":
        vignette_strength = 0.50
    else:
        vignette_strength = 0.35 + intensity * 0.20
    cx, cy = W / 2.0, H / 2.0
    Y_idx, X_idx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((X_idx - cx) / cx) ** 2 + ((Y_idx - cy) / cy) ** 2)
    arr = np.clip(arr * (1.0 - vignette_strength * np.minimum(dist, 1.0))[:, :, np.newaxis], 0, 255)

    result = Image.fromarray(arr.astype(np.uint8))

    tone_key = tone.lower().strip()
    if motion == "cinematic" or "breakthrough" in tone_key or "trauma" in tone_key:
        glow = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)
        gx, gy, radius = W // 2, H // 3, 40
        draw.ellipse(
            [gx - radius, gy - radius, gx + radius, gy + radius],
            fill=(216, 90, 48, int(255 * 0.15)),
        )
        result = Image.alpha_composite(result.convert("RGBA"), glow).convert("RGB")

    return result

# ---------------------------------------------------------------------------
# Shadow self
# ---------------------------------------------------------------------------

def apply_shadow_self(img_pil: Image.Image) -> Image.Image:
    W, H = img_pil.size
    right = img_pil.crop((W // 2, 0, W, H))
    overlay = Image.new("RGBA", right.size, (26, 26, 26, int(255 * 0.25)))
    composited = Image.alpha_composite(right.convert("RGBA"), overlay).convert("RGB")
    blurred = composited.filter(ImageFilter.GaussianBlur(radius=1))
    result = img_pil.copy()
    result.paste(blurred, (W // 2, 0))
    return result

# ---------------------------------------------------------------------------
# Ken Burns — with smoothstep easing
# ---------------------------------------------------------------------------

def apply_ken_burns(img_pil: Image.Image, tone: str, duration_s: float,
                    fps: int = 24, intensity: float = 0.0,
                    motion: str = "still") -> list:
    n_frames = max(1, int(round(duration_s * fps)))

    if motion == "still":
        return [img_pil] * n_frames

    W, H = img_pil.size
    drift_scale = 1.0 + intensity * 0.3
    tone_key = tone.lower().strip()

    params = MOTION_PARAMS.get(motion, MOTION_PARAMS["smooth"])
    zoom_scale   = params["zoom_scale"]
    jitter_scale = params["jitter_scale"]

    frames = []

    for i in range(n_frames):
        raw_t = i / max(n_frames - 1, 1)
        et = _ease(raw_t)  # smoothstep easing applied to all motion

        if "trauma" in tone_key:
            zoom = 1.0 + 0.12 * zoom_scale * et
            dx, dy = 0, int(8 * et * drift_scale)
        elif "anxiety" in tone_key:
            zoom = 1.0 + 0.06 * zoom_scale * et
            if jitter_scale > 0:
                jitter_rng = random.Random((i // 6) * 7919)
                dx = jitter_rng.randint(-2, 2) * jitter_scale
                dy = jitter_rng.randint(-2, 2) * jitter_scale
            else:
                dx, dy = 0, 0
        elif "growth" in tone_key:
            zoom = 1.0
            dx = int((-6 + 12 * et) * drift_scale)
            dy = 0
        elif "breakthrough" in tone_key:
            zoom = 1.0 + 0.08 * zoom_scale - 0.08 * zoom_scale * et
            dx = 0
            dy = int(-10 * et * drift_scale)
        elif "numbness" in tone_key:
            zoom = 1.0 - 0.06 * zoom_scale * et
            dx, dy = 0, int(4 * et * drift_scale)
        else:  # neutral
            zoom = 1.0 + 0.05 * zoom_scale * et
            dx = int(8 * et * drift_scale)
            dy = int(4 * et * drift_scale)

        if zoom >= 1.0:
            crop_w = max(1, int(W / zoom))
            crop_h = max(1, int(H / zoom))
            cx = W // 2 + dx
            cy = H // 2 + dy
            x1 = max(0, min(cx - crop_w // 2, W - crop_w))
            y1 = max(0, min(cy - crop_h // 2, H - crop_h))
            frame = img_pil.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((W, H), Image.LANCZOS)
        else:
            expand_w = max(W, int(W / zoom))
            expand_h = max(H, int(H / zoom))
            padded = Image.new("RGB", (expand_w, expand_h), PAPER_TONE)
            ox = max(0, min((expand_w - W) // 2 - dx, expand_w - W))
            oy = max(0, min((expand_h - H) // 2 - dy, expand_h - H))
            padded.paste(img_pil, (ox, oy))
            frame = padded.resize((W, H), Image.LANCZOS)

        frames.append(frame)

    return frames

# ---------------------------------------------------------------------------
# Transitions — configurable length
# ---------------------------------------------------------------------------

def shadow_fade_p3(out_pil: Image.Image, in_pil: Image.Image, n: int = 18) -> list:
    paper = Image.new("RGB", out_pil.size, PAPER_TONE)
    half = n // 2
    frames = []
    for i in range(half):
        t = _ease((i + 1) / half)
        frames.append(Image.blend(out_pil, paper, t))
    for i in range(n - half):
        t = _ease((i + 1) / (n - half))
        frames.append(Image.blend(paper, in_pil, t))
    return frames


def crack_spread_p3(out_pil: Image.Image, in_pil: Image.Image, n: int = 12) -> list:
    W, H = out_pil.size
    frames = []
    for i in range(n - 1):
        frame = out_pil.copy()
        draw = ImageDraw.Draw(frame)
        y_end = int(H * _ease((i + 1) / (n - 1)))
        draw.line([(W // 2, 0), (W // 2, y_end)], fill=(216, 90, 48), width=3)
        frames.append(frame)
    frames.append(in_pil.copy())
    return frames

# ---------------------------------------------------------------------------
# Subtitle helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "and", "that", "this", "with", "have", "from", "they", "will",
    "would", "could", "should", "there", "their", "about", "which", "what",
    "when", "then", "than", "were", "been", "being", "into", "through",
    "because", "before", "after", "during", "between", "yourself", "himself",
    "herself", "ourselves", "themselves",
}


def extract_subtitle_words(scene: dict, segments: list[dict]) -> list[str]:
    s_start, s_end = float(scene["start"]), float(scene["end"])
    seen: dict[str, None] = {}
    for seg in segments:
        if float(seg["end"]) > s_start and float(seg["start"]) < s_end:
            for w in seg["text"].split():
                clean = w.strip(".,!?;:\"'()[]—").lower()
                if len(clean) > 6 and clean not in _STOP_WORDS and clean not in seen:
                    seen[clean] = None
                if len(seen) == 3:
                    return list(seen)
    return list(seen)


def render_subtitles(frames: list, words: list[str], fade_frames: int = 12) -> list:
    if not words or not frames:
        return frames
    caption = "  ".join(w.upper() for w in words)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    result = []
    for i, frame in enumerate(frames):
        alpha_f = min(1.0, (i + 1) / max(fade_frames, 1))
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        img = img.copy()
        W, H = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)
        try:
            bbox = font.getbbox(caption)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = len(caption) * 7
        x = (W - text_w) // 2
        y = H - 52
        opacity = int(255 * 0.70 * alpha_f)
        draw.text((x, y), caption, font=font, fill=(44, 44, 42, opacity))
        result.append(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
    return result

# ---------------------------------------------------------------------------
# Thumbnail from existing frame
# ---------------------------------------------------------------------------

def make_thumbnail(frames_dir: Path, scenes: list[dict], output_path: Path) -> None:
    total = len(scenes)
    if scenes:
        best = max(
            scenes,
            key=lambda s: intensity_score(s["scene_index"], total)
                          + (1.0 if "breakthrough" in s.get("emotional_tone", "").lower() else 0.0),
        )
        frame_path = frames_dir / f"scene_{best['scene_index']:03d}.png"
        if not frame_path.exists():
            frame_path = sorted(frames_dir.glob("scene_*.png"))[-1]
        console.print(
            f"  Best scene: [cyan]{best['scene_index']}[/] "
            f"(tone={best.get('emotional_tone','neutral')})"
        )
    else:
        frame_path = sorted(frames_dir.glob("scene_*.png"))[-1]

    img = Image.open(frame_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape[:2]

    # Contrast boost
    arr = np.clip(arr * 1.1 - 10, 0, 255)

    # Heavy grain
    rng = np.random.default_rng()
    arr = np.clip(arr + rng.normal(0, 18, arr.shape), 0, 255)

    # Strong vignette (60%)
    cx, cy = W / 2.0, H / 2.0
    Y_idx, X_idx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((X_idx - cx) / cx) ** 2 + ((Y_idx - cy) / cy) ** 2)
    arr = np.clip(arr * (1.0 - 0.60 * np.minimum(dist, 1.0))[:, :, np.newaxis], 0, 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(str(output_path))
    console.print(f"  [green]Thumbnail saved → {output_path}[/] (source: {frame_path.name})")

# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def assemble(
    scenes: list[dict],
    frames_dir: Path,
    audio_path: Path,
    output_path: Path,
    motion: str = "smooth",
    subtitles: bool = False,
    segments: list[dict] | None = None,
) -> None:
    params = MOTION_PARAMS.get(motion, MOTION_PARAMS["smooth"])
    fade_n  = params["fade_n"]
    crack_n = params["crack_n"]

    # Resolve frame paths; drop scenes with missing images
    frame_map: dict[int, Path] = {}
    for f in sorted(frames_dir.glob("scene_*.png")):
        try:
            idx = int(f.stem.split("_")[1])
            frame_map[idx] = f
        except (IndexError, ValueError):
            pass

    valid_scenes = [s for s in scenes if s["scene_index"] in frame_map]
    if not valid_scenes:
        console.print("[red]ERROR:[/] No matching frames found.")
        sys.exit(1)
    if len(valid_scenes) < len(scenes):
        console.print(f"  [yellow]Warning:[/] {len(scenes) - len(valid_scenes)} scenes have no frame — skipped")

    with Image.open(str(frame_map[valid_scenes[0]["scene_index"]])) as _probe:
        W, H = _probe.size

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-acodec", "aac", "-shortest",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    prev_last_frame = None
    prev_tone       = ""
    total_scenes    = len(valid_scenes)
    total_frames    = 0

    console.print(f"  Motion mode : [bold]{motion}[/] (zoom_scale={params['zoom_scale']}, fade_n={fade_n})")
    console.print(f"  Scenes      : {total_scenes}")
    console.print(f"  Output      : [cyan]{output_path}[/]")
    console.print()

    try:
        for scene in valid_scenes:
            scene_idx  = scene["scene_index"]
            tone       = scene.get("emotional_tone", "neutral").lower().strip()
            duration   = float(scene["end"]) - float(scene["start"])
            show_shadow = scene.get("show_shadow", False)
            intensity  = intensity_score(scene_idx, total_scenes)
            img_path   = frame_map[scene_idx]

            img_pil = Image.open(str(img_path)).convert("RGB")
            if show_shadow:
                img_pil = apply_shadow_self(img_pil)
            img_pil = apply_overlays(img_pil, tone, intensity, motion=motion)

            # Transition
            if prev_last_frame is not None:
                use_crack = "anxiety" in prev_tone and motion != "low"
                trans_frames = (
                    crack_spread_p3(prev_last_frame, img_pil, n=crack_n)
                    if use_crack
                    else shadow_fade_p3(prev_last_frame, img_pil, n=fade_n)
                )
                for f in trans_frames:
                    proc.stdin.write(np.array(f, dtype=np.uint8).tobytes())
                    total_frames += 1

            kb_frames = apply_ken_burns(img_pil, tone, duration, fps=FPS,
                                        intensity=intensity, motion=motion)

            if subtitles and segments:
                words = extract_subtitle_words(scene, segments)
                if words:
                    kb_frames = render_subtitles(kb_frames, words)

            for f in kb_frames:
                proc.stdin.write(np.array(f, dtype=np.uint8).tobytes())
                total_frames += 1

            prev_last_frame = kb_frames[-1] if kb_frames else img_pil
            prev_tone = tone

            console.print(
                f"  [green]✓[/] scene_{scene_idx:03d}  "
                f"[dim]{seconds_to_mmss(scene['start'])}-{seconds_to_mmss(scene['end'])}[/]  "
                f"[magenta]{tone}[/]  intensity={intensity:.2f}"
            )

    finally:
        proc.stdin.close()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    console.print(f"\n[bold green]Done.[/] {total_frames} frames → [cyan]{output_path}[/]")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assemble video from existing scene frames")
    parser.add_argument("--frames",    default=None,
                        help="Path to frames directory (default: output/full_run/frames/)")
    parser.add_argument("--source",    default=None,
                        help="Source MP4 for audio (auto-detected from video/ if omitted)")
    parser.add_argument("--motion",    default="still",
                        choices=["still", "low", "smooth", "normal", "cinematic"],
                        help="Ken Burns motion style (default: still — no movement)")
    parser.add_argument("--out",       default=None,
                        help="Output video path (default: <frames-parent>/final_<motion>.mp4)")
    parser.add_argument("--subtitles", action=argparse.BooleanOptionalAction, default=False,
                        help="Burn keyword captions onto frames (--subtitles / --no-subtitles)")
    parser.add_argument("--thumbnail", action="store_true",
                        help="Generate thumbnail only (no video assembly)")
    args = parser.parse_args()

    frames_dir = Path(args.frames) if args.frames else ROOT / "output" / "full_run" / "frames"
    if not frames_dir.exists():
        console.print(f"[red]ERROR:[/] Frames directory not found: {frames_dir}")
        sys.exit(1)

    out_dir = frames_dir.parent

    # Thumbnail only
    if args.thumbnail:
        console.print("[bold]Thumbnail mode[/] — no video assembly")
        scenes = load_scenes(frames_dir)
        thumb_out = ROOT / "output" / "thumbnail.png"
        make_thumbnail(frames_dir, scenes, thumb_out)
        return

    # Resolve output path
    output_path = Path(args.out) if args.out else out_dir / f"final_{args.motion}.mp4"

    console.print(f"[bold]Assembling video[/] from [cyan]{frames_dir}[/]")

    scenes = load_scenes(frames_dir)

    source_mp4 = Path(args.source) if args.source else find_source_mp4()
    if not source_mp4.exists():
        console.print(f"[red]ERROR:[/] Source MP4 not found: {source_mp4}")
        sys.exit(1)

    # Load transcript for subtitles
    segments: list[dict] = []
    if args.subtitles:
        transcript = find_transcript(source_mp4)
        if transcript:
            segments = load_transcript(transcript)
            console.print(f"  [green]OK[/] Loaded {len(segments)} transcript segments")
        else:
            console.print("  [yellow]WARN[/] No transcript found — subtitles disabled")

    console.print(f"\n[bold]Step 1:[/] Extracting audio from [cyan]{source_mp4.name}[/]...")
    audio_path = extract_audio(source_mp4, out_dir)

    console.print(f"\n[bold]Step 2:[/] Rendering {len(scenes)} scenes...")
    try:
        assemble(
            scenes=scenes,
            frames_dir=frames_dir,
            audio_path=audio_path,
            output_path=output_path,
            motion=args.motion,
            subtitles=args.subtitles,
            segments=segments,
        )
    finally:
        try:
            if audio_path.exists():
                audio_path.unlink()
        except PermissionError:
            pass


if __name__ == "__main__":
    main()
