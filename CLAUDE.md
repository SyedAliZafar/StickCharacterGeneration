# Stick Character Automation

## Project Goal
Turn a NotebookLM timestamped transcript into whiteboard-style stick figure PNG frames,
one image per visual concept — ready to sync with the NotebookLM audio in a video editor.

No duplicates. Consistent character style across every video. ~$1 per video.

Channel brand: **TheInnerWar** — psychology content. Every character has a bold red crack on the head.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `test_animate.py` | Phase 1 test — identity-locked character, tone-capped scene durations, Ken Burns / RunwayML video |
| `generate.py` | Production — gpt-image-1 image generation from transcript |
| `animate.py` | Code-rendered animation via matplotlib — zero image cost |

---

## test_animate.py Pipeline

```
Step 1 — Auto-detect first MP4 + transcript in video/
Step 2 — DeepSeek groups transcript into semantic scenes
Step 3 — split_long_scenes() caps each scene by emotional tone
Step 4 — DeepSeek writes a Gemini image prompt per scene
Step 5 — Gemini generates images (free via GEMINI_API_KEY, or ~$0.02 via OpenRouter)
Step 6 — apply_paper_grain() adds subtle paper texture overlay (PIL)
Step 7 — Assemble Ken Burns or RunwayML video
Step 8 — Save to output/phase1_test/
```

### Tone → Scene Duration Cap

| Tone | Max duration |
|------|-------------|
| anxiety/stress | 3s |
| breakthrough / neutral | 5s |
| growth/healing | 6s |
| heavy trauma / numbness | 8s |

Any scene exceeding its cap is split into equal sub-scenes automatically.

---

## generate.py Pipeline

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
- **DeepSeek-V3** via `api.deepseek.com` — groups segments + writes prompts
- **Gemini 2.0 Flash** via Google AI Studio (free) or OpenRouter (~$0.02/image)
- **gpt-image-1** via OpenAI — production image generation (generate.py only)
- **moviepy** — Ken Burns zoom/pan + video assembly
- **Pillow + numpy** — paper grain overlay
- **rich** — terminal progress display

---

## Input

| File | Description |
|------|-------------|
| `video/*.mp4` | Source video — auto-detected by test_animate.py |
| `video/*.txt` | NotebookLM transcript — same stem as MP4, preferred over .json |
| `.env` | API keys |
| `prompts/group_system.txt` | DeepSeek scene-grouping system prompt |
| `prompts/prompt_system.txt` | Identity-locked image generation prompt |

`.env` lookup: `./env` → `../videoSequence/.env`

---

## Output

```
output/
├── frames/           ← generate.py: 001_0s.png, ... (1536×1024)
├── results.json      ← generate.py: scene log
└── phase1_test/
    ├── images/       ← test_animate.py: scene_001.png, ...
    ├── aivid_clips/  ← RunwayML clips
    ├── kburns.mp4
    └── aivid.mp4
```

---

## Environment Variables

```
DEEPSEEK_API_KEY=...         ← required for all scripts (scene grouping)
GEMINI_API_KEY=AIza...       ← free tier, ~1500 req/day — test_animate.py prefers this
GOOGLE_API_KEY=AIza...       ← accepted as alias for GEMINI_API_KEY
OPENROUTER_API_KEY=sk-or-... ← fallback if no Gemini key (~$0.02/image)
OPENAI_API_KEY=sk-proj-...   ← required for generate.py (gpt-image-1)
RUNWAYML_API_KEY=...         ← only for --style aivid
```

`GEMINI_API_KEY` and `GOOGLE_API_KEY` are both accepted — whichever is in `.env`.

---

## Models

| Task | Model | Cost |
|------|-------|------|
| Scene grouping + prompt writing | `deepseek-chat` | ~$0.01/video |
| Image generation (test) | `gemini-2.0-flash-preview-image-generation` | free (direct) |
| Image generation (test fallback) | `gemini-2.5-flash-image` via OpenRouter | ~$0.02/image |
| Image generation (production) | `gpt-image-1` via OpenAI | ~$0.04/image |
| AI video clips | RunwayML Gen3 Turbo | ~$0.25/clip |

---

## Running

```bash
# test_animate.py — dry run first (zero cost)
uv run test_animate.py --dry-run
uv run test_animate.py --style kburns
uv run test_animate.py --style kburns --seconds 50

# generate.py — production
uv run generate.py --dry-run --transcript video/my.txt
uv run generate.py --transcript video/my.txt
```

---

## Character Style (identity-locked across all videos)

Every image has:
- **Warm paper background `#F5F1E8`** — never white
- Stick figure with a **perfectly round blank head** — no face features
- **Bold jagged RED crack** always starting at crown, splitting **downward-left** — same path every scene
- Thin uniform limbs, consistent line thickness
- No other color, no shading, no polished cartoon style

Crack intensity varies by emotional tone (deep/branching/faint/glowing/erased).
Composition: asymmetrical framing, character in left/right third, large negative space.

Prompt templates live in `prompts/` — edit the `.txt` files without touching Python.

---

## Key Design Decisions

- **Group first, generate second**: DeepSeek sees full script before deciding scene boundaries
- **Tone-aware duration caps**: anxiety scenes cut fast (3s), trauma lingers (8s)
- **Prompt files external**: `prompts/group_system.txt` + `prompts/prompt_system.txt` — iterate without code changes
- **Paper grain overlay**: PIL/numpy noise applied to every generated PNG before saving
- **Free image generation first**: GEMINI_API_KEY (direct Google API) tried before OpenRouter
- **Dry run + cost confirmation**: always preview before spending money
- **Phase 1 test isolated**: all test output in `output/phase1_test/` — never overwrites production
- **utf-8-sig encoding**: `.env` read with BOM-safe encoding to handle Windows editors
