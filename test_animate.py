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
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw
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

def apply_ken_burns(img_path: Path, duration: float, tone: str = "neutral") -> list[np.ndarray]:
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

        all_frames.extend(apply_ken_burns(img_path, duration, tone))
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

    parser = argparse.ArgumentParser(description="Phase 1 test: identity-locked character, 20s clip")
    parser.add_argument("--source",  default=None,   help="Source MP4 path")
    parser.add_argument("--seconds", type=float, default=20.0, help="Test clip length in seconds")
    parser.add_argument("--style",   default="both", choices=["kburns", "aivid", "both"],
                        help="Animation style: kburns | aivid | both")
    parser.add_argument("--dry-run",  action="store_true", help="Preview scene plan, no images generated")
    parser.add_argument("--interval", type=float, default=None,
                        help="Fixed image interval in seconds (e.g. 4). Default: use DeepSeek scene grouping.")
    parser.add_argument("--provider", default="auto", choices=["auto", "gemini", "openrouter"],
                        help="Image provider: gemini (free direct), openrouter (~$0.02/image), auto (default)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Delete cached scene images before generating — forces fresh prompts and images")
    args = parser.parse_args()

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
        sample      = random.sample(all_scenes, min(3, len(all_scenes)))
        test_scenes = sorted(sample, key=lambda s: s["scene_index"])
        idxs        = ", ".join(str(s["scene_index"]) for s in test_scenes)
        console.print(f"  [green]Selected scenes:[/] {idxs} (random sample)")

    if not test_scenes:
        console.print("[red]ERROR:[/] No scenes generated for the requested duration.")
        sys.exit(1)

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

    # Reset prompt audit log for this run
    if PROMPTS_LOG.exists():
        PROMPTS_LOG.unlink()

    # Generate images
    src = "Gemini API (free)" if img_provider == "gemini_direct" else "Gemini via OpenRouter"
    console.print(f"\n[bold]Step 2:[/] Generating {len(test_scenes)} images with {src}...\n")
    image_paths = []

    for scene in track(test_scenes, description="Generating images..."):
        idx      = scene["scene_index"]
        tone     = scene.get("emotional_tone", "neutral")
        img_path = IMAGES_DIR / f"scene_{idx:03d}.png"

        if img_path.exists():
            console.print(f"  skip scene {idx} (image exists)")
            image_paths.append(img_path)
            continue

        try:
            prompt    = write_prompt(scene, deepseek_key)
            log_prompt_used(idx, tone, prompt)
            time.sleep(0.2)
            png_bytes = generate_image(prompt, img_key, provider=img_provider, fallback_key=img_fallback or "")
            png_bytes = apply_paper_grain(png_bytes)
            img_path.write_bytes(png_bytes)
            console.print(
                f"  [green]OK[/] scene {idx} [{seconds_to_mmss(scene['start'])}] "
                f"[bold]{scene['concept']}[/] [dim]({tone})[/]"
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

    # Ken Burns
    if args.style in ("kburns", "both"):
        assemble_kburns_phase2(assembled_scenes, image_paths, audio_path)

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
    if args.style in ("kburns", "both"):
        console.print(f"  Ken Burns -> [cyan]{OUT_KBURNS_P2}[/]")
    if args.style in ("aivid", "both"):
        console.print(f"  AI Video  -> [cyan]{OUT_AIVID}[/]")
    console.print(f"  Scenes    : {len(test_scenes)}")
    console.print(f"  Images    : [cyan]{IMAGES_DIR}[/]")
    console.print(f"  Prompts   : [cyan]{PROMPTS_LOG}[/]")
    console.print("\n[bold]Scenes produced:[/]")
    for s in test_scenes:
        console.print(
            f"  scene_{s['scene_index']:03d}  "
            f"[cyan]{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}[/]  "
            f"[magenta]{s.get('emotional_tone', 'neutral')}[/]  "
            f"{s['concept']}"
        )


if __name__ == "__main__":
    main()
