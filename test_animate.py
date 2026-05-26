# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "moviepy<2.0",
#   "Pillow",
#   "numpy",
#   "rich",
# ]
# ///

"""
test_animate.py

Generates images (via Gemini flash on OpenRouter) for the first ~20s of a transcript,
then produces two test videos to compare animation styles:

  output/phase1_test/kburns.mp4  -- Ken Burns slow zoom/pan (free)
  output/phase1_test/aivid.mp4   -- RunwayML AI video animation (~$0.25/scene)

Usage:
  uv run test_animate.py                         # both styles
  uv run test_animate.py --style kburns          # Ken Burns only
  uv run test_animate.py --style aivid           # AI video only
  uv run test_animate.py --seconds 30            # custom clip length
  uv run test_animate.py --source video/my.mp4   # custom source MP4
  uv run test_animate.py --dry-run               # preview scene plan, no cost
  uv run test_animate.py --interval 4 --seconds 50  # fixed 4s images, 50s clip
  uv run test_animate.py --regenerate            # delete cached images and regenerate all
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rich.console import Console
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent
OUTPUT_DIR  = ROOT / "output" / "phase1_test"
IMAGES_DIR  = OUTPUT_DIR / "images"
CLIPS_DIR   = OUTPUT_DIR / "aivid_clips"
PROMPTS_LOG = OUTPUT_DIR / "prompts_used.txt"
OUT_KBURNS  = OUTPUT_DIR / "kburns.mp4"
OUT_AIVID       = OUTPUT_DIR / "aivid.mp4"
OUT_KBURNS_P2   = OUTPUT_DIR / "kburns_phase2.mp4"
OUT_KBURNS_P3   = OUTPUT_DIR / "kburns_phase3.mp4"
OUT_KBURNS_P4   = OUTPUT_DIR / "kburns_phase4.mp4"
OUT_THUMBNAIL   = ROOT / "output" / "thumbnail.png"
CACHE_FILE      = ROOT / "output" / "prompt_cache.json"
REJECTIONS_LOG  = OUTPUT_DIR / "rejections.log"

PAPER_TONE = (245, 241, 232)  # #F5F1E8

OPENROUTER_BASE  = "https://openrouter.ai/api/v1"
GEMINI_IMG_MODEL = "google/gemini-2.5-flash-image"
DEEPSEEK_BASE    = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL   = "deepseek-chat"
RUNWAY_BASE         = "https://api.runwayml.com/v1"
RUNWAY_VERSION      = "2024-11-06"
GEMINI_DIRECT_BASE  = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_DIRECT_MODEL = "gemini-2.0-flash-preview-image-generation"

FPS = 24

TONE_CAPS: dict[str, float] = {
    "heavy trauma":   8.0,
    "numbness":       8.0,
    "growth/healing": 6.0,
    "breakthrough":   5.0,
    "neutral":        5.0,
    "anxiety/stress": 3.0,
}
DEFAULT_CAP = 5.0

# ---------------------------------------------------------------------------
# Prompt file loader
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    path = ROOT / "prompts" / filename
    if not path.exists():
        console.print(f"[red]ERROR:[/] Missing prompt file: {path}")
        console.print(f"  Check that the [cyan]prompts/[/] folder exists and contains [cyan]{filename}[/]")
        sys.exit(1)
    return path.read_text(encoding="utf-8").strip()

GROUP_SYSTEM  = _load_prompt("group_system.txt")
PROMPT_SYSTEM = _load_prompt("prompt_system.txt")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _env_file_path() -> Path | None:
    for candidate in [ROOT / ".env", ROOT.parent / "videoSequence" / ".env"]:
        if candidate.exists():
            return candidate
    return None

def load_env():
    path = _env_file_path()
    if not path:
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip()

def _read_key_direct(*names: str) -> str:
    """Read a key from os.environ first, then directly from .env as fallback."""
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    path = _env_file_path()
    if not path:
        return ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() in names and v.strip():
            return v.strip()
    return ""

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
    console.print(f"[red]ERROR:[/] No transcript found for {source_mp4.name}")
    console.print("Expected: same folder, same filename stem, .txt or .json extension.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Scene grouping
# ---------------------------------------------------------------------------

def group_segments(segments: list[dict], deepseek_key: str) -> list[dict]:
    console.print("[bold]Step 1:[/] Grouping transcript into scenes with DeepSeek...")
    lines = [
        f"[{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}] {s['text']}"
        for s in segments
    ]
    resp = httpx.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": GROUP_SYSTEM},
                {"role": "user",   "content": f"Transcript ({len(segments)} segments):\n\n" + "\n".join(lines)},
            ],
            "temperature": 0.2,
            "max_tokens": 8192,
        },
        timeout=180,
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
        sys.exit(1)
    console.print(f"  [green]OK[/] {len(scenes)} scenes identified")
    return scenes

def select_test_scenes(scenes: list[dict], max_seconds: float) -> list[dict]:
    selected, total = [], 0.0
    for s in scenes:
        dur = float(s["end"]) - float(s["start"])
        if total + dur > max_seconds:
            break
        selected.append(s)
        total += dur
    console.print(f"  [green]OK[/] {len(selected)} test scenes selected ({seconds_to_mmss(total)} total)")
    return selected

_DIRECTION_SEQS: dict[int, list[str]] = {
    2: ["→ rising",     "→ breaking"],
    3: ["→ intro",      "→ peak",        "→ release"],
    4: ["→ rising",     "→ peak",        "→ breaking",   "→ release"],
    5: ["→ rising",     "→ escalating",  "→ peak",       "→ breaking", "→ release"],
}

def _split_direction(i: int, n: int) -> str:
    seq = _DIRECTION_SEQS.get(n)
    if seq:
        return seq[i]
    if i == 0:       return "→ rising"
    if i == n - 1:   return "→ release"
    mid = n // 2
    return "→ peak" if i == mid else ("→ escalating" if i < mid else "→ fading")


def split_long_scenes(scenes: list[dict]) -> list[dict]:
    """Split any scene longer than its emotional-tone cap into equal sub-scenes."""
    result = []
    new_idx = 1
    for scene in scenes:
        tone = scene.get("emotional_tone", "neutral").lower().strip()
        cap  = TONE_CAPS.get(tone, DEFAULT_CAP)
        dur  = float(scene["end"]) - float(scene["start"])

        if dur <= cap:
            result.append({**scene, "scene_index": new_idx})
            new_idx += 1
        else:
            n_parts  = math.ceil(dur / cap)
            part_dur = dur / n_parts
            for i in range(n_parts):
                part_start = float(scene["start"]) + i * part_dur
                part_end   = min(float(scene["start"]) + (i + 1) * part_dur, float(scene["end"]))
                suffix     = f" {_split_direction(i, n_parts)}" if n_parts > 1 else ""
                result.append({
                    **scene,
                    "scene_index": new_idx,
                    "start": round(part_start, 3),
                    "end":   round(part_end, 3),
                    "concept": scene["concept"] + suffix,
                })
                new_idx += 1

    n_split = len(result) - len(scenes)
    if n_split:
        console.print(f"  [green]OK[/] {n_split} scene(s) split by tone cap → {len(result)} total")
    return result

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run_scenes(scenes: list[dict], provider: str = "openrouter") -> None:
    from rich.table import Table
    console.print("\n[bold yellow]DRY RUN — No images will be generated[/]\n")
    table = Table(title=f"Scene Plan ({len(scenes)} scenes)", show_lines=True)
    table.add_column("#",       style="dim",     width=4)
    table.add_column("Time",    style="cyan",    width=13)
    table.add_column("Dur",     style="dim",     width=5)
    table.add_column("Cap",     style="dim",     width=5)
    table.add_column("Concept", style="bold",    width=24)
    table.add_column("Tone",    style="magenta", width=18)
    table.add_column("Visual",  width=45)
    total_s = 0.0
    for s in scenes:
        dur  = float(s["end"]) - float(s["start"])
        tone = s.get("emotional_tone", "neutral").lower().strip()
        cap  = TONE_CAPS.get(tone, DEFAULT_CAP)
        total_s += dur
        table.add_row(
            str(s["scene_index"]),
            f"{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}",
            f"{dur:.1f}s",
            f"{cap:.0f}s",
            s["concept"],
            s.get("emotional_tone", "neutral"),
            s.get("visual_description", "")[:80],
        )
    console.print(table)
    console.print(f"\n  Duration : {seconds_to_mmss(total_s)}")
    if provider == "gemini_direct":
        cost_str = "[green]free[/] (Gemini direct API)"
    else:
        cost_str = f"~$0.02/image via OpenRouter ≈ [green]${len(scenes) * 0.02:.2f}[/]"
    console.print(f"  Images   : {len(scenes)} × {cost_str}")
    console.print("\nRun without [yellow]--dry-run[/] to generate images.")

# ---------------------------------------------------------------------------
# Cost confirmation
# ---------------------------------------------------------------------------

def confirm_generation(n_scenes: int, style: str, provider: str = "openrouter") -> bool:
    console.print(f"\n[bold]Cost estimate:[/]")
    if provider == "gemini_direct":
        console.print(f"  Images  ({n_scenes} scenes, Gemini direct): [green]free[/]")
        img_cost = 0.0
    else:
        img_cost = n_scenes * 0.02
        console.print(f"  Images  ({n_scenes} × ~$0.02 OpenRouter): [green]${img_cost:.2f}[/]")
    if style in ("aivid", "both"):
        runway_cost = n_scenes * 0.25
        console.print(f"  RunwayML ({n_scenes} × ~$0.25/clip):      [yellow]${runway_cost:.2f}[/]")
        img_cost += runway_cost
    console.print(f"  [bold]Total estimate: [green]${img_cost:.2f}[/][/bold]\n")
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Aborted.[/]")
        sys.exit(0)
    return answer in ("y", "yes")

# ---------------------------------------------------------------------------
# Prompt writing
# ---------------------------------------------------------------------------

def write_prompt(scene: dict, deepseek_key: str) -> str:
    user_msg = (
        f"Concept: {scene['concept']}\n"
        f"Emotional tone: {scene.get('emotional_tone', 'neutral')}\n"
        f"Visual description: {scene.get('visual_description', '')}\n"
        f"Script text: {scene['combined_text'][:400]}"
    )
    resp = httpx.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.4,
            "max_tokens": 300,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Prompt audit log
# ---------------------------------------------------------------------------

def log_prompt_used(scene_idx: int, tone: str, prompt: str) -> None:
    PROMPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[scene_{scene_idx:03d}] tone: {tone}\n")
        f.write(f"PROMPT: {prompt}\n")
        f.write("---\n")

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _generate_image_gemini_direct(prompt: str, gemini_key: str, retries: int = 2) -> bytes:
    url = f"{GEMINI_DIRECT_BASE}/models/{GEMINI_DIRECT_MODEL}:generateContent"
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(
                url,
                params={"key": gemini_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                },
                timeout=180,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code} - {resp.text[:300]}")
            parts = resp.json()["candidates"][0]["content"]["parts"]
            for part in parts:
                if "inlineData" in part:
                    return base64.b64decode(part["inlineData"]["data"])
            raise RuntimeError("No image in Gemini response")
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            if attempt < retries:
                console.print(f"    timeout, retrying ({attempt + 1}/{retries})...")
                time.sleep(10)
            else:
                raise RuntimeError(f"Gemini direct timed out after {retries + 1} attempts") from e


def _generate_image_openrouter(prompt: str, openrouter_key: str, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
                json={
                    "model":    GEMINI_IMG_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_modalities": ["IMAGE", "TEXT"],
                },
                timeout=180,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code} - {resp.text[:300]}")

            msg = resp.json()["choices"][0]["message"]

            for img_part in msg.get("images") or []:
                if img_part.get("type") == "image_url":
                    data_url = img_part["image_url"]["url"]
                    b64 = data_url.split(",", 1)[1]
                    return base64.b64decode(b64)

            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        data_url = part["image_url"]["url"]
                        b64 = data_url.split(",", 1)[1]
                        return base64.b64decode(b64)

            if isinstance(content, str) and content.startswith("data:image"):
                b64 = content.split(",", 1)[1]
                return base64.b64decode(b64)

            raise RuntimeError(f"No image in response. content={str(content)[:100]}")

        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            if attempt < retries:
                console.print(f"    timeout, retrying ({attempt + 1}/{retries})...")
                time.sleep(10)
            else:
                raise RuntimeError(f"Gemini image timed out after {retries + 1} attempts") from e


def generate_image(prompt: str, img_key: str, provider: str = "openrouter",
                   retries: int = 2, fallback_key: str = "") -> bytes:
    if provider == "gemini_direct":
        try:
            return _generate_image_gemini_direct(prompt, img_key, retries)
        except RuntimeError as e:
            if "429" in str(e) and fallback_key:
                console.print("  [yellow]Gemini quota hit — falling back to OpenRouter[/]")
                return _generate_image_openrouter(prompt, fallback_key, retries)
            raise
    return _generate_image_openrouter(prompt, img_key, retries)

# ---------------------------------------------------------------------------
# Paper grain overlay
# ---------------------------------------------------------------------------

def apply_paper_grain(img_bytes: bytes, strength: float = 0.03) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    rng = np.random.default_rng()
    noise = rng.normal(0, strength * 255, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(source_mp4: Path) -> Path:
    out = OUTPUT_DIR / "_audio_temp.aac"
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
# Ken Burns animation
# ---------------------------------------------------------------------------

def _apply_ken_burns_legacy(img_path: Path, duration: float, zoom: float = 1.08, direction: str = "in"):
    from moviepy.editor import VideoClip

    img_pil = Image.open(str(img_path)).convert("RGB")
    w, h = img_pil.size
    img_arr = np.array(img_pil)

    def make_frame(t):
        progress = t / max(duration, 0.001)
        scale = 1.0 + (zoom - 1.0) * (progress if direction == "in" else (1.0 - progress))
        crop_w = int(w / scale)
        crop_h = int(h / scale)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = img_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        resized = np.array(Image.fromarray(cropped).resize((w, h), Image.LANCZOS))
        return resized

    return VideoClip(make_frame, duration=duration).set_fps(FPS)

# ---------------------------------------------------------------------------
# Phase 2 — tone-driven Ken Burns (PIL/numpy, returns frames)
# ---------------------------------------------------------------------------

def _apply_ken_burns_phase2(img_path: Path, duration: float, tone: str = "neutral") -> list[np.ndarray]:
    """Return RGB numpy frames with tone-driven camera motion and per-frame film grain."""
    img = Image.open(str(img_path)).convert("RGB")
    W, H = img.size
    base_arr = np.array(img)
    n_frames = max(1, int(round(duration * FPS)))
    rng = np.random.default_rng(int(img_path.stat().st_mtime) % (2 ** 32))
    tone_key = tone.lower().strip()
    frames = []

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)

        if "trauma" in tone_key:
            zoom = 1.0 + 0.15 * t
            dx, dy = 0, int(H * 0.03 * t)
        elif "anxiety" in tone_key:
            zoom = 1.0 + 0.02 * math.sin(i * 0.5)
            win_rng = np.random.default_rng((i // 8) * 7919)
            dx = int(win_rng.integers(-3, 4))
            dy = int(win_rng.integers(-3, 4))
        elif "growth" in tone_key:
            zoom = 1.02
            dx = int(-W * 0.03 + W * 0.06 * t)
            dy = 0
        elif "breakthrough" in tone_key:
            zoom = 1.15 - 0.15 * t
            dx = int(W * 0.01 * t)
            dy = int(-H * 0.03 * t)
        elif "numbness" in tone_key:
            zoom = 1.05 - 0.05 * t
            dx, dy = 0, 0
        else:
            zoom = 1.02
            dx = int(-W * 0.01 + W * 0.02 * t)
            dy = int(-H * 0.01 + H * 0.02 * t)

        crop_w = max(1, int(W / zoom))
        crop_h = max(1, int(H / zoom))
        cx = W // 2 + dx
        cy = H // 2 + dy
        x1 = max(0, min(cx - crop_w // 2, W - crop_w))
        y1 = max(0, min(cy - crop_h // 2, H - crop_h))

        cropped = base_arr[y1:y1 + crop_h, x1:x1 + crop_w]
        frame = np.array(Image.fromarray(cropped).resize((W, H), Image.LANCZOS))

        noise = rng.normal(0, 6, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# Phase 2 — scene transitions
# ---------------------------------------------------------------------------

def shadow_fade(out_arr: np.ndarray, in_arr: np.ndarray, n: int = 18) -> list[np.ndarray]:
    """Fade outgoing image to paper tone then fade in incoming image."""
    half = n // 2
    paper = np.empty_like(out_arr, dtype=np.float32)
    paper[:] = PAPER_TONE
    out_f = out_arr.astype(np.float32)
    in_f  = in_arr.astype(np.float32)
    frames = []
    for i in range(half):
        alpha = 1.0 - (i + 1) / half
        frames.append(np.clip(out_f * alpha + paper * (1.0 - alpha), 0, 255).astype(np.uint8))
    for i in range(n - half):
        alpha = (i + 1) / (n - half)
        frames.append(np.clip(paper * (1.0 - alpha) + in_f * alpha, 0, 255).astype(np.uint8))
    return frames


def crack_spread(out_arr: np.ndarray, in_arr: np.ndarray, n: int = 12) -> list[np.ndarray]:
    """Red crack draws left to right; incoming image bleeds in behind it on later frames."""
    H, W = out_arr.shape[:2]
    step = max(1, W // 50)
    crack_pts: list[tuple[int, int]] = []
    y = H // 2
    crng = np.random.default_rng(42)
    for x in range(0, W + step, step):
        y = int(np.clip(y + crng.integers(-10, 11), H // 4, 3 * H // 4))
        crack_pts.append((min(x, W - 1), y))

    blend_start = n * 2 // 3  # incoming starts bleeding in at 2/3 through
    frames = []
    for i in range(n):
        x_end = int(W * (i + 1) / n)
        # blend incoming image behind the crack from blend_start onward
        if i >= blend_start:
            alpha = (i - blend_start + 1) / (n - blend_start)
            base = np.clip(
                out_arr.astype(np.float32) * (1 - alpha) + in_arr.astype(np.float32) * alpha,
                0, 255,
            ).astype(np.uint8)
        else:
            base = out_arr.copy()
        frame_img = Image.fromarray(base)
        draw = ImageDraw.Draw(frame_img)
        visible = [(x, y) for x, y in crack_pts if x <= x_end]
        if len(visible) >= 2:
            draw.line(visible, fill=(220, 20, 20), width=4)
        frames.append(np.array(frame_img))
    return frames


def make_transition(out_arr: np.ndarray, in_arr: np.ndarray,
                    out_tone: str, _in_tone: str) -> list[np.ndarray]:
    if "anxiety" in out_tone.lower():
        return crack_spread(out_arr, in_arr)
    return shadow_fade(out_arr, in_arr)


# ---------------------------------------------------------------------------
# Phase 2 — video assembly
# ---------------------------------------------------------------------------

def assemble_kburns_phase2(scenes: list[dict], image_paths: list[Path], audio_path: Path) -> Path:
    from moviepy.editor import VideoClip, AudioFileClip

    console.print("\n[bold]Assembling Ken Burns Phase 2 video...[/]")

    all_frames: list[np.ndarray] = []
    prev_arr: np.ndarray | None = None
    prev_tone = ""
    cam_log:   dict[str, int] = {}
    trans_log: dict[str, int] = {}

    for scene, img_path in zip(scenes, image_paths):
        tone     = scene.get("emotional_tone", "neutral").lower().strip()
        duration = float(scene["end"]) - float(scene["start"])
        curr_arr = np.array(Image.open(str(img_path)).convert("RGB"))

        if prev_arr is not None:
            kind = "crack_spread" if "anxiety" in prev_tone else "shadow_fade"
            trans_log[kind] = trans_log.get(kind, 0) + 1
            all_frames.extend(make_transition(prev_arr, curr_arr, prev_tone, tone))

        all_frames.extend(_apply_ken_burns_phase2(img_path, duration, tone))
        cam_log[tone] = cam_log.get(tone, 0) + 1
        prev_arr  = curr_arr
        prev_tone = tone

    if not all_frames:
        console.print("[red]ERROR:[/] No frames generated.")
        return OUT_KBURNS_P2

    total = len(all_frames)

    def make_frame(t: float) -> np.ndarray:
        return all_frames[min(int(t * FPS), total - 1)]

    video = VideoClip(make_frame, duration=total / FPS).set_fps(FPS)
    audio = AudioFileClip(str(audio_path))
    final_dur = min(video.duration, audio.duration)
    video = video.subclip(0, final_dur)
    audio = audio.subclip(0, final_dur)
    final = video.set_audio(audio)

    console.print(f"  Writing [cyan]{OUT_KBURNS_P2.name}[/] ...")
    final.write_videofile(
        str(OUT_KBURNS_P2),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_DIR / "_tmp_p2_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    final.close(); audio.close(); video.close()

    console.print(f"\n[bold green]Phase 2 Summary:[/]")
    console.print(f"  Scenes           : {len(scenes)}")
    console.print(f"  Total frames     : {total}")
    console.print(f"  Camera behaviors : {cam_log}")
    console.print(f"  Transitions      : {trans_log}")
    console.print(f"  [green]OK[/] {OUT_KBURNS_P2}")
    return OUT_KBURNS_P2


# ---------------------------------------------------------------------------
# Phase 3 — intensity scoring
# ---------------------------------------------------------------------------

def intensity_score(scene_index: int, total_scenes: int) -> float:
    """Return 0.0 → 1.0 based on position in video."""
    if total_scenes <= 1:
        return 0.0
    return (scene_index - 1) / (total_scenes - 1)


# ---------------------------------------------------------------------------
# Phase 3 — procedural overlays (PIL/numpy only)
# ---------------------------------------------------------------------------

def apply_overlays(img_pil: Image.Image, tone: str, intensity: float = 0.0,
                   motion: str = "normal") -> Image.Image:
    """Apply film grain, vignette, and crack glow pulse to a PIL image."""
    arr = np.array(img_pil, dtype=np.float32)
    H, W = arr.shape[:2]

    # Film grain
    grain_std = 8.0 + intensity * 6.0
    rng = np.random.default_rng()
    noise = rng.normal(0, grain_std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)

    # Vignette — cinematic forces 50%, otherwise intensity-scaled
    if motion == "cinematic":
        vignette_strength = 0.50
    else:
        vignette_strength = 0.35 + intensity * 0.20
    cx, cy = W / 2.0, H / 2.0
    Y_idx, X_idx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((X_idx - cx) / cx) ** 2 + ((Y_idx - cy) / cy) ** 2)
    vignette = 1.0 - vignette_strength * np.minimum(dist, 1.0)
    arr = arr * vignette[:, :, np.newaxis]
    arr = np.clip(arr, 0, 255)

    result = Image.fromarray(arr.astype(np.uint8))

    # Crack glow pulse: breakthrough/trauma always; cinematic mode: all tones
    tone_key = tone.lower().strip()
    if motion == "cinematic" or "breakthrough" in tone_key or "trauma" in tone_key:
        glow = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)
        gx, gy, radius = W // 2, H // 3, 40
        opacity = int(255 * 0.15)
        draw.ellipse(
            [gx - radius, gy - radius, gx + radius, gy + radius],
            fill=(216, 90, 48, opacity),
        )
        result = Image.alpha_composite(result.convert("RGBA"), glow).convert("RGB")

    return result


# ---------------------------------------------------------------------------
# Phase 3 — emotional Ken Burns (PIL, returns list of PIL Images)
# ---------------------------------------------------------------------------

def apply_ken_burns(img_pil: Image.Image, tone: str, duration_s: float,
                    fps: int = 24, intensity: float = 0.0,
                    motion: str = "normal") -> list:
    """Return list of PIL Images with tone-driven camera motion."""
    W, H = img_pil.size
    n_frames = max(1, int(round(duration_s * fps)))
    drift_scale = 1.0 + intensity * 0.3
    tone_key = tone.lower().strip()

    zoom_scale   = {"low": 0.4, "normal": 1.0, "cinematic": 1.4}.get(motion, 1.0)
    jitter_scale = {"low": 0,   "normal": 1,   "cinematic": 2  }.get(motion, 1)

    frames = []

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)

        if "trauma" in tone_key:
            zoom = 1.0 + 0.12 * zoom_scale * t
            dx, dy = 0, int(8 * t * drift_scale)
        elif "anxiety" in tone_key:
            zoom = 1.0 + 0.06 * zoom_scale * t
            if jitter_scale > 0:
                jitter_rng = random.Random((i // 6) * 7919)
                dx = jitter_rng.randint(-2, 2) * jitter_scale
                dy = jitter_rng.randint(-2, 2) * jitter_scale
            else:
                dx, dy = 0, 0
        elif "growth" in tone_key:
            zoom = 1.0
            dx = int((-6 + 12 * t) * drift_scale)
            dy = 0
        elif "breakthrough" in tone_key:
            zoom = 1.0 + 0.08 * zoom_scale - 0.08 * zoom_scale * t
            dx = 0
            dy = int(-10 * t * drift_scale)
        elif "numbness" in tone_key:
            zoom = 1.0 - 0.06 * zoom_scale * t
            dx, dy = 0, int(4 * t * drift_scale)
        else:  # neutral
            zoom = 1.0 + 0.05 * zoom_scale * t
            dx = int(8 * t * drift_scale)
            dy = int(4 * t * drift_scale)

        if zoom >= 1.0:
            crop_w = max(1, int(W / zoom))
            crop_h = max(1, int(H / zoom))
            cx = W // 2 + dx
            cy = H // 2 + dy
            x1 = max(0, min(cx - crop_w // 2, W - crop_w))
            y1 = max(0, min(cy - crop_h // 2, H - crop_h))
            frame = img_pil.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((W, H), Image.LANCZOS)
        else:
            # Zoom out: pad with paper tone then crop to simulate pulling back
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
# Phase 3 — shadow self post-process
# ---------------------------------------------------------------------------

def apply_shadow_self(img_pil: Image.Image) -> Image.Image:
    """Darken and blur the right half to simulate the shadow self."""
    W, H = img_pil.size
    right = img_pil.crop((W // 2, 0, W, H))
    overlay = Image.new("RGBA", right.size, (26, 26, 26, int(255 * 0.25)))
    composited = Image.alpha_composite(right.convert("RGBA"), overlay).convert("RGB")
    blurred = composited.filter(ImageFilter.GaussianBlur(radius=1))
    result = img_pil.copy()
    result.paste(blurred, (W // 2, 0))
    return result


# ---------------------------------------------------------------------------
# Phase 3 — scene transitions (PIL)
# ---------------------------------------------------------------------------

def shadow_fade_p3(out_pil: Image.Image, in_pil: Image.Image, n: int = 18) -> list:
    """Fade out → paper tone → fade in using PIL Image.blend()."""
    paper = Image.new("RGB", out_pil.size, PAPER_TONE)
    half = n // 2
    frames = []
    for i in range(half):
        frames.append(Image.blend(out_pil, paper, (i + 1) / half))
    for i in range(n - half):
        frames.append(Image.blend(paper, in_pil, (i + 1) / (n - half)))
    return frames


def crack_spread_p3(out_pil: Image.Image, in_pil: Image.Image, n: int = 12) -> list:
    """Red vertical line grows top-to-bottom over n-1 frames, then snaps to incoming image."""
    W, H = out_pil.size
    frames = []
    for i in range(n - 1):
        frame = out_pil.copy()
        draw = ImageDraw.Draw(frame)
        y_end = int(H * (i + 1) / (n - 1))
        draw.line([(W // 2, 0), (W // 2, y_end)], fill=(216, 90, 48), width=3)
        frames.append(frame)
    frames.append(in_pil.copy())
    return frames


def make_transition_p3(out_pil: Image.Image, in_pil: Image.Image, out_tone: str) -> list:
    if "anxiety" in out_tone.lower():
        return crack_spread_p3(out_pil, in_pil)
    return shadow_fade_p3(out_pil, in_pil)


# ---------------------------------------------------------------------------
# Phase 3 — video assembly
# ---------------------------------------------------------------------------

def assemble_kburns_phase3(scenes: list[dict], image_paths: list[Path], audio_path: Path) -> Path:
    from moviepy.editor import VideoClip, AudioFileClip

    console.print("\n[bold]Assembling Ken Burns Phase 3 video...[/]")

    all_pil_frames: list = []
    prev_last_frame = None
    prev_tone = ""
    total_scenes = len(scenes)
    summary_rows: list[tuple] = []

    for scene, img_path in zip(scenes, image_paths):
        scene_idx   = scene["scene_index"]
        tone        = scene.get("emotional_tone", "neutral").lower().strip()
        duration    = float(scene["end"]) - float(scene["start"])
        show_shadow = scene.get("show_shadow", False)
        intensity   = intensity_score(scene_idx, total_scenes)

        img_pil = Image.open(str(img_path)).convert("RGB")

        if show_shadow:
            img_pil = apply_shadow_self(img_pil)

        img_pil = apply_overlays(img_pil, tone, intensity)

        transition_used = "none"
        if prev_last_frame is not None:
            transition_frames = make_transition_p3(prev_last_frame, img_pil, prev_tone)
            all_pil_frames.extend(transition_frames)
            transition_used = "crack_spread" if "anxiety" in prev_tone else "shadow_fade"

        kb_frames = apply_ken_burns(img_pil, tone, duration, fps=FPS, intensity=intensity)
        all_pil_frames.extend(kb_frames)

        prev_last_frame = kb_frames[-1] if kb_frames else img_pil
        prev_tone = tone

        overlay_desc = (
            "crack_glow+grain+vignette" if ("breakthrough" in tone or "trauma" in tone)
            else "grain+vignette"
        )
        if show_shadow:
            overlay_desc += "+shadow_self"
        summary_rows.append((scene_idx, tone, round(intensity, 2), transition_used, overlay_desc))

    if not all_pil_frames:
        console.print("[red]ERROR:[/] No frames generated.")
        return OUT_KBURNS_P3

    total = len(all_pil_frames)
    np_frames = [np.array(f) for f in all_pil_frames]

    def make_frame(t: float) -> np.ndarray:
        return np_frames[min(int(t * FPS), total - 1)]

    video = VideoClip(make_frame, duration=total / FPS).set_fps(FPS)
    audio = AudioFileClip(str(audio_path))
    final_dur = min(video.duration, audio.duration)
    video = video.subclip(0, final_dur)
    audio = audio.subclip(0, final_dur)
    final = video.set_audio(audio)

    console.print(f"  Writing [cyan]{OUT_KBURNS_P3.name}[/] ...")
    final.write_videofile(
        str(OUT_KBURNS_P3),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_DIR / "_tmp_p3_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    final.close(); audio.close(); video.close()

    console.print(f"\n[bold green]Phase 3 Summary:[/]")
    console.print(f"  Scenes       : {len(scenes)}")
    console.print(f"  Total frames : {total}")
    for row in summary_rows:
        console.print(
            f"  scene_{row[0]:03d}  tone={row[1]}  intensity={row[2]:.2f}  "
            f"transition={row[3]}  overlay={row[4]}"
        )
    console.print(f"  [green]OK[/] {OUT_KBURNS_P3}")
    return OUT_KBURNS_P3


# ---------------------------------------------------------------------------
# Phase 4 — style validation
# ---------------------------------------------------------------------------

def validate_image(img_bytes: bytes, scene_idx: int = 0) -> tuple[bool, str]:
    """Run 5 checks. Logs all results to rejections.log. Returns (ok, first_fail_reason|'ok')."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape[:2]
    total = H * W
    checks: dict[str, str] = {}
    first_fail: str | None = None

    # Check 1 — overall brightness
    mean_b = float(arr.mean())
    bright_ok = mean_b >= 110
    checks["BRIGHT"] = f"{mean_b:.0f} {'PASS' if bright_ok else 'FAIL'}"
    if not bright_ok:
        first_fail = first_fail or f"too_dark (mean={mean_b:.1f})"

    # Check 2 — corner background (paper tone)
    cs = 5
    corners = np.concatenate([
        arr[:cs, :cs].reshape(-1, 3),
        arr[:cs, W - cs:].reshape(-1, 3),
        arr[H - cs:, :cs].reshape(-1, 3),
        arr[H - cs:, W - cs:].reshape(-1, 3),
    ])
    avg = corners.mean(axis=0)
    corner_ok = avg[0] >= 180 and avg[1] >= 160 and avg[2] >= 140
    checks["CORNER"] = f"[{avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f}] {'PASS' if corner_ok else 'FAIL'}"
    if not corner_ok:
        first_fail = first_fail or f"dark_background (corner_avg=[{avg[0]:.0f},{avg[1]:.0f},{avg[2]:.0f}])"

    # Check 3 — green/cyan detection (hue 60-180, sat > 60, val > 100)
    r, g, b = arr[:, :, 0] / 255, arr[:, :, 1] / 255, arr[:, :, 2] / 255
    maxc  = np.maximum(np.maximum(r, g), b)
    delta = maxc - np.minimum(np.minimum(r, g), b)
    hue   = np.zeros_like(r)
    mr = (maxc == r) & (delta > 0)
    mg = (maxc == g) & (delta > 0)
    mb = (maxc == b) & (delta > 0)
    hue[mr] = (60 * ((g[mr] - b[mr]) / delta[mr])) % 360
    hue[mg] = 60 * ((b[mg] - r[mg]) / delta[mg]) + 120
    hue[mb] = 60 * ((r[mb] - g[mb]) / delta[mb]) + 240
    sat255 = np.where(maxc > 0, delta / maxc, 0.0) * 255
    val255 = maxc * 255
    green_pct = float(((hue >= 60) & (hue <= 180) & (sat255 > 60) & (val255 > 100)).sum()) / total * 100
    green_ok  = green_pct <= 1.5
    checks["GREEN"] = f"{green_pct:.1f}% {'PASS' if green_ok else 'FAIL'}"
    if not green_ok:
        first_fail = first_fail or "off-brand green/cyan color detected"

    # Check 4 — dark patch (4×4 grid, any patch mean < 80 → 3D object)
    ph, pw = H // 4, W // 4
    gray   = arr.mean(axis=2)
    patch_min = min(
        float(gray[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw].mean())
        for r in range(4) for c in range(4)
    )
    patch_ok = patch_min >= 80
    checks["DARK_PATCH"] = f"{patch_min:.0f} {'PASS' if patch_ok else 'FAIL'}"
    if not patch_ok:
        first_fail = first_fail or "3D dark object detected — not hand-drawn style"

    # Check 5 — edge density (FIND_EDGES, white pixels > 18% → too detailed)
    edges    = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_pct = float((np.array(edges, dtype=np.float32) > 128).sum()) / total * 100
    edge_ok  = edge_pct <= 18.0
    checks["EDGES"] = f"{edge_pct:.1f}% {'PASS' if edge_ok else 'FAIL'}"
    if not edge_ok:
        first_fail = first_fail or "too much detail — not clean hand-drawn style"

    # Log every check (pass and fail) to rejections.log
    detail   = " | ".join(f"{k}: {v}" for k, v in checks.items())
    log_line = f"[scene_{scene_idx:03d}] {detail}\n"
    REJECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(log_line)
    console.print(f"  [dim]validation: {detail}[/]")

    return first_fail is None, first_fail or "ok"


def log_rejection(scene_idx: int, attempt: int, reason: str, kept: bool) -> None:
    REJECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    status = "KEPT_DESPITE_FAILURE" if kept else f"attempt_{attempt}"
    with REJECTIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(f"scene_{scene_idx:03d} | {status} | {reason}\n")


# ---------------------------------------------------------------------------
# Phase 4 — prompt + image cache
# ---------------------------------------------------------------------------

def _cache_key(combined_text: str, tone: str, concept: str = "") -> str:
    return hashlib.sha256(f"{combined_text}{tone}{concept}".encode()).hexdigest()[:16]


def _load_cache(no_cache: bool) -> dict:
    if no_cache:
        return {}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def write_prompt_cached(scene: dict, deepseek_key: str,
                        cache: dict, no_cache: bool) -> tuple[str, str]:
    """Return (prompt, 'HIT'|'MISS'). Updates cache in-place."""
    key = _cache_key(scene.get("combined_text", ""), scene.get("emotional_tone", "neutral"), scene.get("concept", ""))
    if not no_cache and key in cache and cache[key].get("prompt"):
        return cache[key]["prompt"], "HIT"
    prompt = write_prompt(scene, deepseek_key)
    cache.setdefault(key, {})["prompt"] = prompt
    return prompt, "MISS"


def generate_image_cached(
    prompt: str, scene: dict, img_path: Path,
    img_key: str, img_provider: str, img_fallback: str,
    cache: dict, no_cache: bool,
) -> tuple:
    """Return (png_bytes | None, 'HIT'|'MISS'). None means cached file already copied."""
    key = _cache_key(scene.get("combined_text", ""), scene.get("emotional_tone", "neutral"), scene.get("concept", ""))
    cached_path_str = cache.get(key, {}).get("image_path", "")
    if not no_cache and cached_path_str:
        cached_path = Path(cached_path_str)
        if cached_path.exists():
            if cached_path != img_path:
                shutil.copy2(cached_path, img_path)
            return None, "HIT"
    png_bytes = generate_image(prompt, img_key, provider=img_provider,
                               fallback_key=img_fallback or "")
    cache.setdefault(key, {})["image_path"] = str(img_path)
    return png_bytes, "MISS"


# ---------------------------------------------------------------------------
# Phase 4 — thumbnail generation
# ---------------------------------------------------------------------------

def generate_thumbnail(
    scenes: list[dict], image_paths: list[Path],
    deepseek_key: str, img_key: str, img_provider: str, img_fallback: str,
) -> None:
    total_scenes = len(scenes)
    best_idx, best_score = 0, -1.0
    for i, scene in enumerate(scenes):
        tone  = scene.get("emotional_tone", "neutral").lower()
        score = intensity_score(scene["scene_index"], total_scenes)
        if "breakthrough" in tone:
            score += 1.0
        if score > best_score:
            best_score, best_idx = score, i

    scene = scenes[best_idx]
    tone  = scene.get("emotional_tone", "neutral")
    console.print(f"\n[bold]Thumbnail:[/] scene {scene['scene_index']} ({tone}) ...")

    base_prompt   = write_prompt(scene, deepseek_key)
    boosted_prompt = (
        base_prompt.rstrip()
        + ", high contrast, bold lines, strong silhouette, "
          "crack prominently visible, maximum negative space"
    )
    png_bytes = generate_image(boosted_prompt, img_key, provider=img_provider,
                               fallback_key=img_fallback or "")

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape[:2]

    arr = np.clip(arr + np.random.default_rng().normal(0, 14, arr.shape), 0, 255)

    cx, cy   = W / 2.0, H / 2.0
    Y_i, X_i = np.mgrid[0:H, 0:W]
    dist     = np.sqrt(((X_i - cx) / cx) ** 2 + ((Y_i - cy) / cy) ** 2)
    arr      = np.clip(arr * (1.0 - 0.55 * np.minimum(dist, 1.0))[:, :, np.newaxis], 0, 255)

    OUT_THUMBNAIL.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(str(OUT_THUMBNAIL))
    console.print(
        f"  [green]Thumbnail saved → {OUT_THUMBNAIL}[/] "
        f"(scene {scene['scene_index']}, tone: {tone})"
    )


# ---------------------------------------------------------------------------
# Phase 4 — subtitle word extraction and rendering
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "and", "that", "this", "with", "have", "from", "they", "will",
    "would", "could", "should", "there", "their", "about", "which", "what",
    "when", "then", "than", "were", "been", "being", "into", "through",
    "because", "before", "after", "during", "between", "yourself", "himself",
    "herself", "ourselves", "themselves",
}


def extract_subtitle_words(scene: dict, segments: list[dict]) -> list[str]:
    """Return up to 3 long non-stop words from the scene's transcript window."""
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
    """Burn subtitle caption onto frames with fade-in. Returns new list of PIL Images."""
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
# Phase 4 — video assembly
# ---------------------------------------------------------------------------

def assemble_kburns_phase4(
    scenes: list[dict],
    image_paths: list[Path],
    audio_path: Path,
    segments: list[dict],
    motion: str = "normal",
    subtitles: bool = False,
) -> Path:
    console.print(f"\n[bold]Assembling Ken Burns Phase 4 video (motion={motion})...[/]")

    with Image.open(str(image_paths[0])) as _probe:
        W, H = _probe.size

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    OUT_KBURNS_P4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-vcodec", "libx264", "-preset", "fast", "-crf", "18",
        "-acodec", "aac", "-shortest",
        str(OUT_KBURNS_P4),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    prev_last_frame      = None
    prev_tone            = ""
    total_scenes         = len(scenes)
    total_frames         = 0
    summary_rows: list[tuple] = []

    try:
        for scene, img_path in zip(scenes, image_paths):
            scene_idx   = scene["scene_index"]
            tone        = scene.get("emotional_tone", "neutral").lower().strip()
            duration    = float(scene["end"]) - float(scene["start"])
            show_shadow = scene.get("show_shadow", False)
            intensity   = intensity_score(scene_idx, total_scenes)

            img_pil = Image.open(str(img_path)).convert("RGB")
            if show_shadow:
                img_pil = apply_shadow_self(img_pil)
            img_pil = apply_overlays(img_pil, tone, intensity, motion=motion)

            # Transition — stream frames immediately, no list accumulation
            transition_used = "none"
            if prev_last_frame is not None:
                use_crack = "anxiety" in prev_tone and motion != "low"
                transition_frames = (
                    crack_spread_p3(prev_last_frame, img_pil)
                    if use_crack
                    else shadow_fade_p3(prev_last_frame, img_pil)
                )
                for f in transition_frames:
                    proc.stdin.write(np.array(f, dtype=np.uint8).tobytes())
                    total_frames += 1
                transition_used = "crack_spread" if use_crack else "shadow_fade"

            kb_frames = apply_ken_burns(img_pil, tone, duration, fps=FPS,
                                        intensity=intensity, motion=motion)

            subtitle_words: list[str] = []
            if subtitles:
                subtitle_words = extract_subtitle_words(scene, segments)
                if subtitle_words:
                    kb_frames = render_subtitles(kb_frames, subtitle_words)

            for f in kb_frames:
                proc.stdin.write(np.array(f, dtype=np.uint8).tobytes())
                total_frames += 1

            prev_last_frame = kb_frames[-1] if kb_frames else img_pil
            prev_tone = tone

            overlay_desc = (
                "crack_glow+grain+vignette"
                if (motion == "cinematic" or "breakthrough" in tone or "trauma" in tone)
                else "grain+vignette"
            )
            if show_shadow:
                overlay_desc += "+shadow_self"
            summary_rows.append((
                scene_idx, tone, round(intensity, 2),
                transition_used, overlay_desc,
                ", ".join(subtitle_words) if subtitle_words else "—",
            ))
    finally:
        proc.stdin.close()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    console.print(f"\n[bold green]Phase 4 Summary (motion={motion}):[/]")
    console.print(f"  Scenes       : {len(scenes)}")
    console.print(f"  Total frames : {total_frames}")
    for row in summary_rows:
        console.print(
            f"  scene_{row[0]:03d}  tone={row[1]}  intensity={row[2]:.2f}  "
            f"transition={row[3]}  overlay={row[4]}  subtitles=[{row[5]}]"
        )
    console.print(f"  [green]OK[/] {OUT_KBURNS_P4}")
    return OUT_KBURNS_P4


# ---------------------------------------------------------------------------
# RunwayML AI video generation
# ---------------------------------------------------------------------------

def apply_runway_video(img_path: Path, scene_idx: int, runway_key: str) -> Path:
    out_path = CLIPS_DIR / f"scene_{scene_idx:03d}.mp4"
    if out_path.exists():
        console.print(f"    skip scene {scene_idx} (clip exists)")
        return out_path

    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    data_uri = f"data:image/png;base64,{img_b64}"

    headers = {
        "Authorization": f"Bearer {runway_key}",
        "X-Runway-Version": RUNWAY_VERSION,
        "Content-Type": "application/json",
    }

    resp = httpx.post(
        f"{RUNWAY_BASE}/image_to_video",
        headers=headers,
        json={
            "model":       "gen3a_turbo",
            "promptImage": data_uri,
            "duration":    5,
            "ratio":       "1280:768",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"RunwayML submit failed {resp.status_code}: {resp.text[:300]}")

    task_id = resp.json()["id"]
    console.print(f"    RunwayML task {task_id} submitted, polling...")

    for attempt in range(30):  # up to 150s
        time.sleep(5)
        poll = httpx.get(
            f"{RUNWAY_BASE}/tasks/{task_id}",
            headers=headers,
            timeout=15,
        )
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status", "PENDING")

        if status == "SUCCEEDED":
            video_url = data["output"][0]
            console.print(f"    Downloading clip...")
            video_bytes = httpx.get(video_url, timeout=60).content
            out_path.write_bytes(video_bytes)
            return out_path

        if status == "FAILED":
            raise RuntimeError(f"RunwayML task failed: {data.get('failure', 'unknown')}")

    raise TimeoutError(f"RunwayML task {task_id} did not complete in 150s")

# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def assemble_kburns(scenes: list[dict], image_paths: list[Path], audio_path: Path) -> Path:
    from moviepy.editor import concatenate_videoclips, AudioFileClip

    console.print("\n[bold]Assembling Ken Burns video...[/]")
    clips = []
    for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
        duration = float(scene["end"]) - float(scene["start"])
        direction = "in" if i % 2 == 0 else "out"
        clip = _apply_ken_burns_legacy(img_path, duration, zoom=1.08, direction=direction)
        if i > 0:
            clip = clip.crossfadein(0.5)
        clips.append(clip)

    padding = -0.5 if len(clips) > 1 else 0
    video = concatenate_videoclips(clips, padding=padding, method="compose")

    audio = AudioFileClip(str(audio_path))
    final_dur = min(video.duration, audio.duration)
    video = video.subclip(0, final_dur)
    audio = audio.subclip(0, final_dur)
    final = video.set_audio(audio)

    console.print(f"  Writing [cyan]{OUT_KBURNS.name}[/] ...")
    final.write_videofile(
        str(OUT_KBURNS),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_DIR / "_tmp_kburns_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    final.close(); audio.close(); video.close()
    for c in clips:
        c.close()

    console.print(f"  [green]OK[/] {OUT_KBURNS}")
    return OUT_KBURNS

def assemble_aivid(scenes: list[dict], clip_paths: list[Path], audio_path: Path) -> Path:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

    console.print("\n[bold]Assembling AI video...[/]")
    clips = []
    for scene, clip_path in zip(scenes, clip_paths):
        duration = float(scene["end"]) - float(scene["start"])
        raw = VideoFileClip(str(clip_path))
        if raw.duration < duration:
            n_loops = int(duration / raw.duration) + 1
            looped = concatenate_videoclips([raw] * n_loops)
            clip = looped.subclip(0, duration)
        else:
            clip = raw.subclip(0, duration)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(str(audio_path))
    final_dur = min(video.duration, audio.duration)
    video = video.subclip(0, final_dur)
    audio = audio.subclip(0, final_dur)
    final = video.set_audio(audio)

    console.print(f"  Writing [cyan]{OUT_AIVID.name}[/] ...")
    final.write_videofile(
        str(OUT_AIVID),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(OUTPUT_DIR / "_tmp_aivid_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    final.close(); audio.close(); video.close()
    for c in clips:
        c.close()

    console.print(f"  [green]OK[/] {OUT_AIVID}")
    return OUT_AIVID

# ---------------------------------------------------------------------------
# Fixed-interval scene chunking (alternative to DeepSeek grouping)
# ---------------------------------------------------------------------------

def chunk_by_interval(segments: list[dict], interval: float, max_seconds: float) -> list[dict]:
    """Slice [0, max_seconds] into fixed chunks of `interval` seconds each."""
    chunks = []
    start = 0.0
    idx = 1
    while start < max_seconds - 0.01:
        end = min(start + interval, max_seconds)
        text_parts = [
            s["text"] for s in segments
            if float(s["end"]) > start and float(s["start"]) < end
        ]
        chunks.append({
            "scene_index": idx,
            "start": round(start, 3),
            "end":   round(end, 3),
            "concept": f"{int(start)}s-{int(end)}s",
            "emotional_tone": "neutral",
            "combined_text": " ".join(text_parts) or "(no speech)",
            "visual_description": "",
        })
        start += interval
        idx += 1
    console.print(
        f"  [green]OK[/] {len(chunks)} chunks "
        f"({seconds_to_mmss(max_seconds)} @ {interval}s intervals)"
    )
    return chunks

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Phase 1-4 test: identity-locked character")
    parser.add_argument("--source",    default=None,   help="Source MP4 path")
    parser.add_argument("--seconds",   type=float, default=20.0, help="Test clip length in seconds")
    parser.add_argument("--style",     default="both", choices=["kburns", "aivid", "both"],
                        help="Animation style: kburns | aivid | both")
    parser.add_argument("--dry-run",   action="store_true", help="Preview scene plan, no images generated")
    parser.add_argument("--interval",  type=float, default=None,
                        help="Fixed image interval in seconds (e.g. 4). Default: use DeepSeek scene grouping.")
    parser.add_argument("--provider",  default="auto", choices=["auto", "gemini", "openrouter"],
                        help="Image provider: gemini (free direct), openrouter (~$0.02/image), auto (default)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Delete cached scene images before generating — forces fresh prompts and images")
    parser.add_argument("--thumbnail", action="store_true",
                        help="Generate thumbnail only (skips video assembly)")
    parser.add_argument("--subtitles", action="store_true",
                        help="Burn psychological keyword captions onto Ken Burns Phase 4 video")
    parser.add_argument("--motion",    default="normal", choices=["low", "normal", "cinematic"],
                        help="Ken Burns motion intensity: low | normal | cinematic")
    parser.add_argument("--no-cache",  action="store_true", dest="no_cache",
                        help="Bypass prompt and image cache")
    parser.add_argument("--full-run",  action="store_true", dest="full_run",
                        help="Process ALL transcript scenes; output to output/full_run/final.mp4")
    args = parser.parse_args()

    if args.full_run:
        global OUTPUT_DIR, IMAGES_DIR, CLIPS_DIR, PROMPTS_LOG, OUT_KBURNS_P4, REJECTIONS_LOG
        OUTPUT_DIR     = ROOT / "output" / "full_run"
        IMAGES_DIR     = OUTPUT_DIR / "frames"
        CLIPS_DIR      = OUTPUT_DIR / "aivid_clips"
        PROMPTS_LOG    = OUTPUT_DIR / "prompts_used.txt"
        OUT_KBURNS_P4  = OUTPUT_DIR / "final.mp4"
        REJECTIONS_LOG = OUTPUT_DIR / "rejections.log"

    source_mp4 = Path(args.source) if args.source else find_source_mp4()
    if not source_mp4.exists():
        console.print(f"[red]ERROR:[/] Source not found: {source_mp4}")
        sys.exit(1)

    transcript_path = find_transcript(source_mp4)

    deepseek_key = require_key("DEEPSEEK_API_KEY")
    runway_key   = require_key("RUNWAYML_API_KEY") if args.style in ("aivid", "both") and not args.dry_run else None

    if not args.dry_run:
        gemini_direct_key = _read_key_direct("GEMINI_API_KEY", "GOOGLE_API_KEY")
        openrouter_raw    = _read_key_direct("OPENROUTER_API_KEY")

        want_gemini    = args.provider in ("auto", "gemini")
        want_openrouter = args.provider in ("auto", "openrouter")

        if want_gemini and gemini_direct_key:
            img_key      = gemini_direct_key
            img_provider = "gemini_direct"
            img_fallback = openrouter_raw if args.provider == "auto" else ""
            console.print("  Using [green]Gemini API directly[/] (free tier, ~1500 req/day)")
            if img_fallback:
                console.print("  OpenRouter available as quota fallback")
        elif want_openrouter and openrouter_raw:
            img_key      = openrouter_raw
            img_provider = "openrouter"
            img_fallback = ""
            console.print("  Using [yellow]OpenRouter[/] (Gemini flash via OpenRouter, ~$0.02/image)")
        else:
            if args.provider == "gemini":
                console.print("[red]ERROR:[/] --provider gemini requires GEMINI_API_KEY in .env")
            elif args.provider == "openrouter":
                console.print("[red]ERROR:[/] --provider openrouter requires OPENROUTER_API_KEY in .env")
            else:
                console.print("[red]ERROR:[/] Set GEMINI_API_KEY (free) or OPENROUTER_API_KEY in .env")
            sys.exit(1)
    else:
        img_key = img_provider = img_fallback = None

    segments = load_transcript(transcript_path)
    console.print(f"[bold]Loaded[/] {len(segments)} segments from [cyan]{transcript_path.name}[/]")

    if args.interval:
        console.print(f"[bold]Step 1:[/] Fixed-interval mode — {args.interval}s chunks, no DeepSeek grouping")
        test_scenes = chunk_by_interval(segments, args.interval, args.seconds)
    else:
        all_scenes  = group_segments(segments, deepseek_key)
        all_scenes  = split_long_scenes(all_scenes)
        if args.full_run:
            test_scenes = all_scenes
            console.print(f"  [green]All {len(test_scenes)} scenes selected (full run)[/]")
        else:
            sample      = random.sample(all_scenes, min(3, len(all_scenes)))
            test_scenes = sorted(sample, key=lambda s: s["scene_index"])
            idxs        = ", ".join(str(s["scene_index"]) for s in test_scenes)
            console.print(f"  [green]Selected scenes:[/] {idxs} (random sample)")

    if not test_scenes:
        console.print("[red]ERROR:[/] No scenes generated for the requested duration.")
        sys.exit(1)

    if args.full_run:
        total_est_dur = sum(float(s["end"]) - float(s["start"]) for s in test_scenes)
        est_cost      = len(test_scenes) * 0.02
        console.print(f"\n[bold]Full Run Preview:[/]")
        console.print(f"  Total scenes      : {len(test_scenes)}")
        console.print(f"  Estimated cost    : ~${est_cost:.2f} ({len(test_scenes)} × $0.02/image)")
        console.print(f"  Estimated duration: ~{seconds_to_mmss(total_est_dur)}")

    if args.dry_run:
        dry_run_scenes(test_scenes, img_provider or "openrouter")
        return

    if not confirm_generation(len(test_scenes), args.style, img_provider):
        console.print("[yellow]Cancelled.[/]")
        sys.exit(0)

    # Set up output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if args.style in ("aivid", "both"):
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Purge cached images when --regenerate is set
    if args.regenerate and IMAGES_DIR.exists():
        stale = sorted(IMAGES_DIR.glob("scene_*.png"))
        for f in stale:
            f.unlink()
        if stale:
            console.print(f"  [yellow]--regenerate:[/] deleted {len(stale)} cached image(s)")

    # Reset logs for this run
    if PROMPTS_LOG.exists():
        PROMPTS_LOG.unlink()
    if REJECTIONS_LOG.exists():
        REJECTIONS_LOG.unlink()

    # Load prompt/image cache
    cache = _load_cache(args.no_cache)

    # Generate images
    src = "Gemini API (free)" if img_provider == "gemini_direct" else "Gemini via OpenRouter"
    console.print(f"\n[bold]Step 2:[/] Generating {len(test_scenes)} images with {src}...\n")
    image_paths       = []
    cache_hits        = 0
    prompt_cache_hits = 0
    img_api_calls     = 0
    total_rejections  = 0
    total_regen       = 0
    seen_hashes: dict = {}  # md5 -> scene_idx, for duplicate detection in full-run

    for scene in track(test_scenes, description="Generating images..."):
        idx      = scene["scene_index"]
        tone     = scene.get("emotional_tone", "neutral")
        img_path = IMAGES_DIR / f"scene_{idx:03d}.png"

        if img_path.exists() and not args.regenerate:
            if args.full_run:
                raw = img_path.read_bytes()
                ok, _ = validate_image(raw, idx)
                if ok:
                    img_md5 = hashlib.md5(raw).hexdigest()
                    if img_md5 in seen_hashes:
                        console.print(f"  [yellow]Duplicate of scene {seen_hashes[img_md5]} — regenerating scene {idx}[/]")
                        img_path.unlink()
                    else:
                        seen_hashes[img_md5] = idx
                        console.print(f"  skip scene {idx} (exists, validation PASS)")
                        image_paths.append(img_path)
                        cache_hits += 1
                        continue
                else:
                    console.print(f"  [yellow]Existing image failed validation — regenerating scene {idx}[/]")
            else:
                console.print(f"  skip scene {idx} (image exists)")
                image_paths.append(img_path)
                continue

        try:
            # Prompt (cache-aware)
            prompt, p_status = write_prompt_cached(scene, deepseek_key, cache, args.no_cache)
            if p_status == "HIT":
                prompt_cache_hits += 1
            console.print(f"  [dim][CACHE {p_status}][/] prompt scene {idx}")
            log_prompt_used(idx, tone, prompt)
            time.sleep(0.2)

            # Image generation + validation with up to 2 retries
            png_bytes = None
            img_cache_status = "MISS"
            for attempt in range(3):
                if attempt == 0:
                    raw_bytes, img_cache_status = generate_image_cached(
                        prompt, scene, img_path,
                        img_key, img_provider, img_fallback or "",
                        cache, args.no_cache,
                    )
                    if raw_bytes is None:
                        # cache hit — file already on disk
                        console.print(f"  [dim][CACHE HIT][/] image scene {idx}")
                        png_bytes = img_path.read_bytes()
                        cache_hits += 1
                        break
                    img_api_calls += 1
                    png_bytes = apply_paper_grain(raw_bytes)
                else:
                    raw_bytes = generate_image(
                        prompt, img_key, provider=img_provider,
                        fallback_key=img_fallback or "",
                    )
                    img_api_calls += 1
                    png_bytes = apply_paper_grain(raw_bytes)

                ok, reason = validate_image(png_bytes, idx)
                if ok:
                    if attempt > 0:
                        total_regen += 1
                    break
                # Validation failed
                kept = attempt == 2
                total_rejections += 1
                log_rejection(idx, attempt + 1, reason, kept)
                if kept:
                    console.print(f"  [yellow]WARN[/] scene {idx}: validation failed twice — keeping image")
                else:
                    console.print(f"  [red][REJECTED scene_{idx} attempt {attempt + 1}: {reason}][/] → regenerating")
                    time.sleep(1)

            img_path.write_bytes(png_bytes)
            _save_cache(cache)
            s_intensity = intensity_score(idx, len(test_scenes))
            console.print(
                f"  [green]OK[/] scene {idx} "
                f"[{seconds_to_mmss(scene['start'])}-{seconds_to_mmss(scene['end'])}] "
                f"[bold]{scene['concept']}[/] "
                f"[dim]tone={tone} intensity={s_intensity:.2f} [CACHE {img_cache_status}][/]"
            )
            image_paths.append(img_path)
        except Exception as e:
            err = str(e)
            if any(code in err for code in ("401", "403", "invalid_api_key")):
                console.print(f"  [red]FATAL[/] scene {idx}: auth error — {err}")
                sys.exit(1)
            console.print(f"  [yellow]SKIP[/] scene {idx}: {err} — continuing")

        time.sleep(0.5)

    if not image_paths:
        console.print("[red]ERROR:[/] All scenes failed to generate. Check your API key and network.")
        sys.exit(1)
    if len(image_paths) < len(test_scenes):
        skipped = len(test_scenes) - len(image_paths)
        console.print(f"  [yellow]Warning:[/] {skipped} scene(s) skipped — assembling video from {len(image_paths)} image(s)")

    # Align scene list to only scenes that produced an image
    generated_idxs = {int(p.stem.split("_")[1]) for p in image_paths}
    assembled_scenes = [s for s in test_scenes if s["scene_index"] in generated_idxs]

    # Extract audio
    console.print(f"\n[bold]Step 3:[/] Extracting audio...")
    audio_path = extract_audio(source_mp4)

    # Thumbnail mode — skip video assembly
    if args.thumbnail:
        generate_thumbnail(
            assembled_scenes, image_paths,
            deepseek_key, img_key, img_provider, img_fallback or "",
        )
        console.print("\n[bold green]Done (thumbnail mode).[/]")
        return

    # Ken Burns
    if args.style in ("kburns", "both"):
        if args.full_run:
            try:
                assemble_kburns_phase4(
                    assembled_scenes, image_paths, audio_path,
                    segments=segments,
                    motion=args.motion,
                    subtitles=args.subtitles,
                )
            except Exception as e:
                console.print(f"\n[red]Assembly failed: {e}[/]")
                console.print("[yellow]Generated images saved at:[/]")
                for p in image_paths:
                    console.print(f"  {p}")
                sys.exit(1)
        else:
            assemble_kburns_phase2(assembled_scenes, image_paths, audio_path)
            assemble_kburns_phase3(assembled_scenes, image_paths, audio_path)
            assemble_kburns_phase4(
                assembled_scenes, image_paths, audio_path,
                segments=segments,
                motion=args.motion,
                subtitles=args.subtitles,
            )

    # AI Video
    if args.style in ("aivid", "both"):
        console.print(f"\n[bold]Step 4:[/] Generating RunwayML video clips...\n")
        clip_paths = []
        for scene, img_path in zip(assembled_scenes, image_paths):
            console.print(f"  Scene {scene['scene_index']} [{seconds_to_mmss(scene['start'])}] "
                          f"[bold]{scene['concept']}[/]")
            clip_path = apply_runway_video(img_path, scene["scene_index"], runway_key)
            console.print(f"  [green]OK[/] {clip_path.name}")
            clip_paths.append(clip_path)

        assemble_aivid(assembled_scenes, clip_paths, audio_path)

    # Cleanup temp audio
    try:
        if audio_path.exists():
            audio_path.unlink()
    except PermissionError:
        pass

    console.print("\n[bold green]Done.[/]")
    if args.full_run:
        total_run_dur = sum(float(s["end"]) - float(s["start"]) for s in assembled_scenes)
        cost_spent    = img_api_calls * 0.02
        console.print(f"\n[bold]Full Run Summary:[/]")
        console.print(f"  Scenes generated  : {len(assembled_scenes)}")
        console.print(f"  From cache/skip   : {cache_hits}")
        console.print(f"  API calls made    : {img_api_calls}")
        console.print(f"  Prompt cache hits : {prompt_cache_hits}")
        console.print(f"  Validation fails  : {total_rejections}")
        console.print(f"  Successful regen  : {total_regen}")
        console.print(f"  Video duration    : ~{seconds_to_mmss(total_run_dur)}")
        console.print(f"  Cost spent        : ${cost_spent:.2f}")
        console.print(f"  Final video       : [cyan]{OUT_KBURNS_P4}[/]")
        console.print(f"  Images            : [cyan]{IMAGES_DIR}[/]")
        console.print(f"  Rejection log     : [cyan]{REJECTIONS_LOG}[/]")
    else:
        if args.style in ("kburns", "both"):
            console.print(f"  Ken Burns P2 -> [cyan]{OUT_KBURNS_P2}[/]")
            console.print(f"  Ken Burns P3 -> [cyan]{OUT_KBURNS_P3}[/]")
            console.print(f"  Ken Burns P4 -> [cyan]{OUT_KBURNS_P4}[/] (motion={args.motion})")
        if args.style in ("aivid", "both"):
            console.print(f"  AI Video  -> [cyan]{OUT_AIVID}[/]")
        console.print(f"  Scenes    : {len(assembled_scenes)}")
        console.print(f"  Images    : [cyan]{IMAGES_DIR}[/]")
        console.print(f"  Prompts   : [cyan]{PROMPTS_LOG}[/]")
        console.print("\n[bold]Scenes produced:[/]")
        for s in assembled_scenes:
            console.print(
                f"  scene_{s['scene_index']:03d}  "
                f"[cyan]{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}[/]  "
                f"[magenta]{s.get('emotional_tone', 'neutral')}[/]  "
                f"{s['concept']}"
            )


if __name__ == "__main__":
    main()
