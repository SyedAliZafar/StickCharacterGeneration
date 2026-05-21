# Stick Character Automation

## Project Goal
Turn a NotebookLM timestamped transcript into whiteboard-style stick figure PNG frames,
one image per visual concept — ready to sync with the NotebookLM audio in a video editor.

No duplicates. Consistent character style across every video. ~$1 per video.

Channel brand: **TheInnerWar** — psychology content. Every character has a bold red crack on the head.

---

## Pipeline

```
Step 1 — Parse NotebookLM .txt transcript → raw segments [{start, end, text}]
Step 2 — DeepSeek-V3 groups segments into 20–30 distinct visual scenes
Step 3 — DeepSeek-V3 writes a gpt-image-1 prompt for each scene
Step 4 — gpt-image-1 generates each image (1536×1024, whiteboard style)
Step 5 — Save as output/frames/001_0s.png, 002_36s.png, ...
Step 6 — Save output/results.json (scene, timestamp, prompt, cost, path)
```

---

## Tech Stack
- **Python 3.11+** with `uv run` (PEP 723 inline deps, no manual venv)
- **DeepSeek-V3** via `api.deepseek.com` directly — groups segments + writes prompts
- **gpt-image-1** via OpenAI — generates whiteboard stick figure images (returns base64)
- **httpx** — all API calls
- **Pillow** — listed as dep, not currently used in main path
- **rich** — terminal progress display

---

## Input

| File | Description |
|------|-------------|
| `transcript.txt` | NotebookLM export — timestamped lines `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text` |
| `.env` | API keys |

Accepts both:
- `.txt` — NotebookLM timestamped format (primary)
- `.json` — segment array `[{start, end, text}]` (also supported)

`.env` lookup order: `./env` → `../videoSequence/.env`

---

## Output

```
output/
├── frames/        ← 001_0s.png, 002_36s.png, ... (1536×1024 PNGs)
└── results.json   ← full log: scene, timestamp range, prompt, cost, file path
```

---

## Environment Variables

```
DEEPSEEK_API_KEY=...     ← required (DeepSeek-V3 scene grouping + prompt writing)
OPENAI_API_KEY=sk-proj-... ← required (gpt-image-1 image generation)
```

> Note: despite CLAUDE.md history saying `OPENROUTER_API_KEY`, the code calls
> `api.deepseek.com` directly and reads `DEEPSEEK_API_KEY`.

---

## Models

| Task | Model | Endpoint | Cost |
|------|-------|----------|------|
| Segment grouping + prompt writing | `deepseek-chat` | `api.deepseek.com` | ~$0.01 total |
| Image generation | `gpt-image-1` | `api.openai.com` | $0.04/image |
| **Typical video (~25 scenes)** | | | **~$1.00** |

---

## Running

```bash
# 1. Add API keys to .env
# 2. Place your NotebookLM transcript as transcript.txt

uv run generate.py

# Dry run — shows scenes DeepSeek would generate, no images, no cost:
uv run generate.py --dry-run

# Custom transcript path:
uv run generate.py --transcript path/to/my_script.txt
```

---

## Character Style (consistent across all videos)

Every image has:
- Whiteboard background, black marker line art
- Stick figure with a **perfectly round head**
- **Bold jagged RED crack** splitting from crown downward — mandatory in every frame
- Thin limbs, expressive posture
- No other color, no shading

The red crack intensity varies by emotional tone:
- **Heavy trauma** → deep, wide, jagged crack splitting far down the face
- **Anxiety/stress** → sharp branching crack, spiderweb fracture
- **Growth/healing** → faint crack with small stitching lines
- **Breakthrough** → crack glows at edges, light coming through
- **Numbness** → thin, barely visible, almost erased

---

## Key Design Decisions

- **Group first, generate second**: DeepSeek sees the full script before deciding scene
  boundaries — far better grouping than processing segment-by-segment
- **gpt-image-1 over FLUX**: Psychology metaphors need strong prompt understanding;
  gpt-image-1 follows complex metaphorical descriptions reliably
- **Dry run flag**: Review scene breakdown and prompts before spending any money
- **Skip existing frames**: Re-running is safe — already-generated PNGs are skipped
- **1536×1024 output**: Closest gpt-image-1 size to 1920×1080 (16:9 widescreen)
- **base64 response**: `gpt-image-1` returns `b64_json`; image bytes decoded before saving
