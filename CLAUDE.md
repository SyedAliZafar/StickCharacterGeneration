# Stick Character Automation

## Project Goal
Turn a NotebookLM timestamped transcript into whiteboard-style stick figure PNG frames,
one image per visual concept — ready to sync with the NotebookLM audio in a video editor.

No duplicates. Consistent character style across every image. ~$1 per video.

---

## Pipeline

```
Step 1 — Parse NotebookLM .txt transcript → raw segments [{start, end, text}]
Step 2 — DeepSeek-V3 groups segments into 20–30 distinct visual scenes
Step 3 — DeepSeek-V3 writes a DALL-E 3 prompt for each scene
Step 4 — DALL-E 3 generates each image (1792×1024, whiteboard style)
Step 5 — Save as output/frames/001_0s.png, 002_36s.png, ...
Step 6 — Save output/results.json (scene, timestamp, prompt, cost, path)
```

---

## Tech Stack
- **Python 3.11+** with `uv run` (PEP 723 inline deps, no manual venv)
- **DeepSeek-V3** via OpenRouter — groups segments + writes prompts
- **DALL-E 3** via OpenAI — generates whiteboard stick figure images
- **httpx** — all API calls
- **Pillow** — optional post-processing / resizing
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

---

## Output

```
output/
├── frames/        ← 001_0s.png, 002_36s.png, ... (final delivery PNGs 1792×1024)
└── results.json   ← full log: scene, timestamp range, prompt, cost, file path
```

---

## Environment Variables

```
OPENAI_API_KEY=sk-proj-...         ← required (DALL-E 3)
OPENROUTER_API_KEY=sk-or-v1-...   ← required (DeepSeek-V3)
```

---

## Models

| Task | Model | Cost |
|------|-------|------|
| Segment grouping + prompt writing | `deepseek/deepseek-chat` via OpenRouter | ~$0.01 total |
| Image generation | `dall-e-3` via OpenAI | $0.04/image |
| **Typical video (~25 scenes)** | | **~$1.00** |

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

Every DALL-E 3 prompt uses this prefix:
> "whiteboard animation, black marker on white background, simple stick figure with round head
> and thin limbs, clean hand-drawn educational illustration, no color, no shading,"

This ensures all images share the same visual identity across every video on the channel.

---

## Key Design Decisions

- **Group first, generate second**: DeepSeek sees the full script before deciding scene
  boundaries — far better grouping than processing segment-by-segment
- **DALL-E 3 over FLUX**: Psychology metaphors need strong prompt understanding; DALL-E 3
  follows complex metaphorical descriptions; FLUX Schnell does not
- **Dry run flag**: Review scene breakdown and prompts before spending any money
- **Skip existing frames**: Re-running is safe — already-generated PNGs are skipped
- **1792×1024 output**: Closest DALL-E 3 size to 1920×1080 (16:9 widescreen)
