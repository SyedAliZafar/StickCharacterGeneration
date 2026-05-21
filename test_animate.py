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
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
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
OUT_KBURNS  = OUTPUT_DIR / "kburns.mp4"
OUT_AIVID   = OUTPUT_DIR / "aivid.mp4"

OPENROUTER_BASE  = "https://openrouter.ai/api/v1"
GEMINI_IMG_MODEL = "google/gemini-2.5-flash-image"
DEEPSEEK_BASE    = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL   = "deepseek-chat"
RUNWAY_BASE      = "https://api.runwayml.com/v1"
RUNWAY_VERSION   = "2024-11-06"

FPS = 24

# ---------------------------------------------------------------------------
# Prompt file loader
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    path = ROOT / "prompts" / filename
    if not path.exists():
        console.print(f"[red]ERROR:[/] Missing prompt file: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8").strip()

GROUP_SYSTEM  = _load_prompt("group_system.txt")
PROMPT_SYSTEM = _load_prompt("prompt_system.txt")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        fallback = ROOT.parent / "videoSequence" / ".env"
        if fallback.exists():
            env_path = fallback
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

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run_scenes(scenes: list[dict]) -> None:
    from rich.table import Table
    console.print("\n[bold yellow]DRY RUN — No images will be generated[/]\n")
    table = Table(title=f"Scene Plan ({len(scenes)} scenes)", show_lines=True)
    table.add_column("#",       style="dim",     width=4)
    table.add_column("Time",    style="cyan",    width=13)
    table.add_column("Concept", style="bold",    width=26)
    table.add_column("Tone",    style="magenta", width=20)
    table.add_column("Visual",  width=55)
    total_s = 0.0
    for s in scenes:
        total_s += float(s["end"]) - float(s["start"])
        table.add_row(
            str(s["scene_index"]),
            f"{seconds_to_mmss(s['start'])}-{seconds_to_mmss(s['end'])}",
            s["concept"],
            s.get("emotional_tone", "neutral"),
            s.get("visual_description", "")[:100],
        )
    console.print(table)
    console.print(f"\n  Duration : {seconds_to_mmss(total_s)}")
    console.print(f"  Images   : {len(scenes)} × ~$0.02 (Gemini flash) ≈ [green]${len(scenes) * 0.02:.2f}[/]")
    console.print("\nRun without [yellow]--dry-run[/] to generate images.")

# ---------------------------------------------------------------------------
# Cost confirmation
# ---------------------------------------------------------------------------

def confirm_generation(n_scenes: int, style: str) -> bool:
    img_cost = n_scenes * 0.02
    console.print(f"\n[bold]Cost estimate:[/]")
    console.print(f"  Images  ({n_scenes} × ~$0.02 Gemini flash): [green]${img_cost:.2f}[/]")
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
# Image generation (Gemini via OpenRouter)
# ---------------------------------------------------------------------------

def generate_image(prompt: str, openrouter_key: str, retries: int = 2) -> bytes:
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
                timeout=120,
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
                time.sleep(5)
            else:
                raise RuntimeError(f"Gemini image timed out after {retries + 1} attempts") from e

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

def apply_ken_burns(img_path: Path, duration: float, zoom: float = 1.08, direction: str = "in"):
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
        clip = apply_ken_burns(img_path, duration, zoom=1.08, direction=direction)
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
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Phase 1 test: identity-locked character, 20s clip")
    parser.add_argument("--source",  default=None,   help="Source MP4 path")
    parser.add_argument("--seconds", type=float, default=20.0, help="Test clip length in seconds")
    parser.add_argument("--style",   default="both", choices=["kburns", "aivid", "both"],
                        help="Animation style: kburns | aivid | both")
    parser.add_argument("--dry-run", action="store_true", help="Preview scene plan, no images generated")
    args = parser.parse_args()

    source_mp4 = Path(args.source) if args.source else find_source_mp4()
    if not source_mp4.exists():
        console.print(f"[red]ERROR:[/] Source not found: {source_mp4}")
        sys.exit(1)

    transcript_path = find_transcript(source_mp4)

    deepseek_key   = require_key("DEEPSEEK_API_KEY")
    openrouter_key = require_key("OPENROUTER_API_KEY") if not args.dry_run else None
    runway_key     = require_key("RUNWAYML_API_KEY") if args.style in ("aivid", "both") and not args.dry_run else None

    segments = load_transcript(transcript_path)
    console.print(f"[bold]Loaded[/] {len(segments)} segments from [cyan]{transcript_path.name}[/]")

    all_scenes  = group_segments(segments, deepseek_key)
    test_scenes = select_test_scenes(all_scenes, args.seconds)

    if not test_scenes:
        console.print("[red]ERROR:[/] No scenes fit within the requested duration.")
        sys.exit(1)

    if args.dry_run:
        dry_run_scenes(test_scenes)
        return

    if not confirm_generation(len(test_scenes), args.style):
        console.print("[yellow]Cancelled.[/]")
        sys.exit(0)

    # Set up output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if args.style in ("aivid", "both"):
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate images
    console.print(f"\n[bold]Step 2:[/] Generating {len(test_scenes)} images with Gemini via OpenRouter...\n")
    image_paths = []

    for scene in track(test_scenes, description="Generating images..."):
        idx      = scene["scene_index"]
        img_path = IMAGES_DIR / f"scene_{idx:03d}.png"

        if img_path.exists():
            console.print(f"  skip scene {idx} (image exists)")
            image_paths.append(img_path)
            continue

        try:
            prompt    = write_prompt(scene, deepseek_key)
            time.sleep(0.2)
            png_bytes = generate_image(prompt, openrouter_key)
            png_bytes = apply_paper_grain(png_bytes)
            img_path.write_bytes(png_bytes)
            console.print(
                f"  [green]OK[/] scene {idx} [{seconds_to_mmss(scene['start'])}] "
                f"[bold]{scene['concept']}[/]"
            )
            image_paths.append(img_path)
        except Exception as e:
            console.print(f"  [red]FAIL[/] scene {idx}: {e}")
            sys.exit(1)

        time.sleep(0.5)

    # Extract audio
    console.print(f"\n[bold]Step 3:[/] Extracting audio...")
    audio_path = extract_audio(source_mp4)

    # Ken Burns
    if args.style in ("kburns", "both"):
        assemble_kburns(test_scenes, image_paths, audio_path)

    # AI Video
    if args.style in ("aivid", "both"):
        console.print(f"\n[bold]Step 4:[/] Generating RunwayML video clips...\n")
        clip_paths = []
        for scene, img_path in zip(test_scenes, image_paths):
            console.print(f"  Scene {scene['scene_index']} [{seconds_to_mmss(scene['start'])}] "
                          f"[bold]{scene['concept']}[/]")
            clip_path = apply_runway_video(img_path, scene["scene_index"], runway_key)
            console.print(f"  [green]OK[/] {clip_path.name}")
            clip_paths.append(clip_path)

        assemble_aivid(test_scenes, clip_paths, audio_path)

    # Cleanup temp audio
    try:
        if audio_path.exists():
            audio_path.unlink()
    except PermissionError:
        pass

    console.print("\n[bold green]Done.[/]")
    if args.style in ("kburns", "both"):
        console.print(f"  Ken Burns -> [cyan]{OUT_KBURNS}[/]")
    if args.style in ("aivid", "both"):
        console.print(f"  AI Video  -> [cyan]{OUT_AIVID}[/]")
    console.print(f"  Scenes    : {len(test_scenes)}")
    console.print(f"  Images    : [cyan]{IMAGES_DIR}[/]")


if __name__ == "__main__":
    main()
