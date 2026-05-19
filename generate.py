# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "Pillow",
#   "rich",
# ]
# ///

"""
stick_character_automation / generate.py

Reads a NotebookLM timestamped transcript, groups lines into visual scenes
using DeepSeek-V3, generates whiteboard stick figure images with DALL-E 3.

Usage:
  uv run generate.py                          # uses transcript.txt
  uv run generate.py --transcript my.txt      # custom path
  uv run generate.py --dry-run                # preview scenes, no images generated
"""

import base64
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent
OUTPUT_DIR  = ROOT / "output"
FRAMES_DIR  = OUTPUT_DIR / "frames"

OPENAI_BASE    = "https://api.openai.com/v1"
DEEPSEEK_BASE  = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DALLE_MODEL     = "gpt-image-1"
DALLE_SIZE      = "1536x1024"   # closest 16:9 for gpt-image-1

CHARACTER_STYLE = (
    "whiteboard animation, black marker on white background, "
    "simple stick figure with round head and thin limbs, "
    "clean hand-drawn educational illustration, no color, no shading, "
    "minimal line art, clear and simple composition, "
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        # try parent videoSequence .env as fallback
        parent_env = ROOT.parent / "videoSequence" / ".env"
        if parent_env.exists():
            env_path = parent_env
            console.print(f"[dim]Using .env from {parent_env}[/]")
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
# Transcript parsing — NotebookLM .txt format
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

def parse_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("segments", [])

def load_transcript(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return parse_json(path)
    return parse_txt(path)

# ---------------------------------------------------------------------------
# Step 1 — DeepSeek groups segments into visual scenes
# ---------------------------------------------------------------------------

GROUP_SYSTEM = """You are a whiteboard video producer for a psychology YouTube channel.

Given a full timestamped transcript, identify the distinct VISUAL SCENES needed.
Each scene = one whiteboard image that a viewer sees for several seconds.

Rules:
- Aim for 20–30 scenes for a video under 10 minutes
- Each scene covers ONE clear visual concept or metaphor
- Group consecutive segments that share the same visual idea
- Scenes must follow the narrative order of the script exactly
- Use the actual start/end timestamps from the segments

Respond with a JSON array only — no markdown, no explanation:
[
  {
    "scene_index": 1,
    "start": 0.0,
    "end": 36.2,
    "concept": "short label (3-5 words)",
    "combined_text": "full text of all segments in this scene combined",
    "visual_description": "what should be drawn — describe the metaphor or concept visually"
  }
]"""

def group_segments(segments: list[dict], deepseek_key: str) -> list[dict]:
    console.print("[bold]Step 1:[/] Grouping segments into visual scenes with DeepSeek-V3...")

    # Build a compact transcript string
    lines = [
        f"[{seconds_to_mmss(s['start'])}–{seconds_to_mmss(s['end'])}] {s['text']}"
        for s in segments
    ]
    transcript_text = "\n".join(lines)

    user_msg = f"Full transcript ({len(segments)} segments):\n\n{transcript_text}"

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
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 8192,
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        scenes = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        console.print(f"[red]ERROR:[/] DeepSeek returned truncated JSON ({e})")
        console.print(f"[dim]Last 200 chars: ...{raw[-200:]}[/]")
        sys.exit(1)
    console.print(f"  [green]✓[/] {len(scenes)} scenes identified")
    return scenes

# ---------------------------------------------------------------------------
# Step 2 — DeepSeek writes DALL-E prompt per scene
# ---------------------------------------------------------------------------

PROMPT_SYSTEM = """You write DALL-E 3 image generation prompts for whiteboard animation videos.

Given a scene concept and script text, write ONE image generation prompt.

Always start with:
"whiteboard animation, black marker on white background, simple stick figure with round head and thin limbs, clean hand-drawn educational illustration, no color, no shading, "

Then describe:
- The central visual: what stick figures are doing, any objects, any text labels
- The psychological metaphor made literal and visual
- Keep it under 200 words

Output the prompt text only — nothing else."""

def write_prompt(scene: dict, deepseek_key: str) -> str:
    user_msg = (
        f"Concept: {scene['concept']}\n"
        f"Visual description: {scene['visual_description']}\n"
        f"Script text: {scene['combined_text'][:400]}"
    )
    resp = httpx.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {deepseek_key}",
            "Content-Type": "application/json",
        },
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
# Step 3 — DALL-E 3 image generation
# ---------------------------------------------------------------------------

def generate_image(prompt: str, openai_key: str) -> bytes:
    resp = httpx.post(
        f"{OPENAI_BASE}/images/generations",
        headers={
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        },
        json={
            "model":   DALLE_MODEL,
            "prompt":  prompt,
            "n":       1,
            "size":    DALLE_SIZE,
            "quality": "medium",
        },
        timeout=120,
    )
    if resp.status_code != 200:
        err = resp.json().get("error", {})
        raise RuntimeError(f"{resp.status_code} — {err.get('code', '')} — {err.get('message', resp.text[:300])}")

    b64 = resp.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)

# ---------------------------------------------------------------------------
# Dry run — print scene table, no API image calls
# ---------------------------------------------------------------------------

def dry_run(scenes: list[dict], openrouter_key: str):
    console.print("\n[bold yellow]DRY RUN — No images will be generated[/]\n")

    table = Table(title=f"Scene Plan ({len(scenes)} scenes)", show_lines=True)
    table.add_column("#",        style="dim",  width=4)
    table.add_column("Time",     style="cyan", width=12)
    table.add_column("Concept",  style="bold", width=24)
    table.add_column("Visual",   width=50)

    for s in scenes:
        table.add_row(
            str(s["scene_index"]),
            f"{seconds_to_mmss(s['start'])}–{seconds_to_mmss(s['end'])}",
            s["concept"],
            s["visual_description"][:120],
        )

    console.print(table)
    est_cost = len(scenes) * 0.04
    console.print(f"\n[bold]Estimated cost:[/] {len(scenes)} images × $0.04 = [green]${est_cost:.2f}[/]")
    console.print("\nRun without [yellow]--dry-run[/] to generate images.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Generate whiteboard frames from transcript")
    parser.add_argument("--transcript", default="transcript.txt", help="Path to transcript file")
    parser.add_argument("--dry-run",    action="store_true",      help="Preview scenes only, no images")
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        console.print(f"[red]ERROR:[/] Transcript not found: {transcript_path}")
        console.print("Place your NotebookLM export as [cyan]transcript.txt[/] in this folder.")
        sys.exit(1)

    deepseek_key = require_key("DEEPSEEK_API_KEY")
    if not args.dry_run:
        openai_key = require_key("OPENAI_API_KEY")

    # Parse transcript
    segments = load_transcript(transcript_path)
    console.print(f"[bold]Loaded[/] {len(segments)} segments from [cyan]{transcript_path.name}[/]")

    # Group into scenes
    scenes = group_segments(segments, deepseek_key)

    if args.dry_run:
        dry_run(scenes, deepseek_key)
        return

    # Generate images
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    console.print(f"\n[bold]Step 2:[/] Writing prompts + generating {len(scenes)} images with DALL-E 3...\n")

    for scene in track(scenes, description="Generating..."):
        idx       = scene["scene_index"]
        start_s   = int(scene["start"])
        frame_name = f"{idx:03d}_{start_s}s"
        png_path   = FRAMES_DIR / f"{frame_name}.png"

        if png_path.exists():
            console.print(f"  [dim]skip {frame_name} (exists)[/]")
            results.append({**scene, "file": str(png_path), "skipped": True})
            continue

        try:
            # Write DALL-E prompt
            prompt = write_prompt(scene, deepseek_key)
            time.sleep(0.2)

            # Generate image
            png_bytes = generate_image(prompt, openai_key)
            png_path.write_bytes(png_bytes)

            console.print(
                f"  [green]✓[/] [{seconds_to_mmss(scene['start'])}] "
                f"[bold]{scene['concept']}[/]"
            )

            results.append({
                **scene,
                "prompt": prompt,
                "file":   str(png_path),
                "cost":   0.04,
            })

        except Exception as e:
            console.print(f"  [red]✗[/] {frame_name}: {e}")
            results.append({**scene, "error": str(e), "file": None})

        time.sleep(0.5)   # stay within rate limits

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    done    = sum(1 for r in results if r.get("file") and not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    errors  = sum(1 for r in results if r.get("error"))
    total_cost = done * 0.04

    console.print(f"\n[bold green]Done.[/]")
    console.print(f"  Generated : {done} images  (${total_cost:.2f})")
    console.print(f"  Skipped   : {skipped} (already existed)")
    console.print(f"  Errors    : {errors}")
    console.print(f"  Frames    : [cyan]{FRAMES_DIR}[/]")
    console.print(f"  Results   : [cyan]{results_path}[/]")


if __name__ == "__main__":
    main()
