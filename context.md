# Project Context

## Where This Fits

```
NotebookLM
  → generates audio podcast (.wav / .mp3)
  → exports timestamped transcript (.txt)   ← this pipeline reads this

stick_character_automation/
  → test_animate.py  → phase 1 test images + Ken Burns / AI video
  → generate.py      → production gpt-image-1 frames
  → animate.py       → code-rendered matplotlib animation (zero cost)
  → output/phase1_test/  ← test output, never overwrites production
  → output/frames/       ← production frames

Video editor (CapCut / Premiere / FFmpeg)
  → syncs frames to audio timeline
  → exports final YouTube video
```

---

## Sibling Projects

| Folder | Purpose |
|--------|---------|
| `videoSequence/` | Sequences & deduplicates frames from existing AI-generated videos |
| `whiteboard-gen/` | Experimental — SVG + FLUX Schnell hybrid |
| `stick_character_automation/` | THIS — production pipeline |
| `psych-youtube-automation/` | Script writing, ideation, NotebookLM prompts |
| `ThumbnailGeneration/` | YouTube thumbnail creation |

---

## Image Generation: Model Choices

### test_animate.py (Phase 1 test)
Uses **Gemini 2.0 Flash image generation** directly via Google AI Studio (free tier):
- Model: `gemini-2.0-flash-preview-image-generation`
- Endpoint: `generativelanguage.googleapis.com/v1beta`
- Key: `GEMINI_API_KEY` or `GOOGLE_API_KEY` (both accepted)
- Free tier: ~1500 requests/day at 15 RPM
- Falls back to OpenRouter (`OPENROUTER_API_KEY`) if no Gemini key is found

### generate.py (production)
Uses **gpt-image-1** via OpenAI:
- $0.04/image at medium quality, 1536×1024
- Psychology metaphors need strong prompt understanding — Gemini flash is sufficient for tests, gpt-image-1 for final production

### Why not FLUX Schnell
FLUX ($0.003/image) cannot reliably follow metaphorical prompts like
"anger is a heavily armored security guard protecting something soft."
gpt-image-1 / Gemini understand nuanced scene descriptions.

---

## Why DeepSeek for Grouping + Prompts

- Calls `api.deepseek.com` directly with `DEEPSEEK_API_KEY`
- DeepSeek-V3 excels at structured JSON output
- ~4× cheaper than Claude Sonnet for this text task
- Grouping cost: ~$0.01 for the full script

---

## Scene Strategy: Group First, Then Cap by Tone

A 518-second script has ~100 raw segments (3–8s each).
Generating one image per raw segment = 100 images = $4.00 + visual chaos.

**Step 1 — DeepSeek grouping**: understands narrative arc, produces 20–30 semantic scenes.

**Step 2 — Tone-aware duration cap** (`split_long_scenes()`):
Any scene longer than its emotional tone's cap is split into equal sub-scenes:

| Tone | Cap | Reasoning |
|------|-----|-----------|
| anxiety/stress | 3s | urgency → fast cuts |
| breakthrough / neutral | 5s | baseline pacing |
| growth/healing | 6s | contemplative but moving |
| heavy trauma / numbness | 8s | viewer needs to sit with it |

This matches TheInnerWar's retention-optimised editing rhythm:
- Talking head baseline: 4–6s per image
- Urgency/listing: 2–3s
- Heavy emotional truth: 7–10s

---

## Character Identity — TheInnerWar Brand

The character is a **persistent mascot**, not redesigned per scene:

- Warm paper background `#F5F1E8` — never white
- Stick figure, perfectly round blank head — no face features
- Bold jagged RED crack: always starts at crown, splits **downward-left**, same path every image
- Thin uniform limbs, consistent line weight
- Black marker line art only — no shading, no color except the crack
- Asymmetrical composition: character in left or right third, never centered
- Oppressive symbolic scale: objects dwarf the character

Prompt templates are in `prompts/` — edit without touching Python code.

---

## .env Loading

`.env` is read with `utf-8-sig` encoding (BOM-safe for Windows editors).
Direct assignment (`os.environ[k] = v`) — not `setdefault` — so `.env` always wins
over any stale system environment values.

`_read_key_direct(*names)` bypasses `os.environ` entirely and reads `.env` directly
as a final fallback, ensuring Gemini/OpenRouter keys are always found.

`.env` lookup order: `./env` → `../videoSequence/.env`

---

## Estimated Cost Per Video

### Test (test_animate.py, 20s clip)
| Item | Cost |
|------|------|
| DeepSeek grouping | ~$0.01 |
| Gemini image generation (direct) | free |
| **Test total** | **~$0.01** |

### Production (generate.py, full video)
| Item | Cost |
|------|------|
| DeepSeek grouping + prompt writing | ~$0.01 |
| gpt-image-1 × 25 images | ~$1.00 |
| **Total per video** | **~$1.01** |

At 4 videos/month ≈ $4/month.

---

## Code Map

### test_animate.py

| Function | Purpose |
|----------|---------|
| `_env_file_path()` | Finds `.env` — local first, then `../videoSequence/.env` |
| `load_env()` | Reads `.env` with utf-8-sig, direct assignment to os.environ |
| `_read_key_direct(*names)` | Reads key from os.environ then directly from .env as fallback |
| `_load_prompt(filename)` | Loads prompt template from `prompts/` — exits if missing |
| `chunk_by_interval(segments, interval, max_s)` | Fixed-interval slicing (--interval flag) |
| `split_long_scenes(scenes)` | Splits scenes exceeding tone cap into equal sub-scenes |
| `dry_run_scenes(scenes, provider)` | Prints scene table with Dur/Cap columns, zero cost |
| `confirm_generation(n, style, provider)` | Prints cost estimate, waits for y/N |
| `_generate_image_gemini_direct(prompt, key)` | Calls Google AI directly (free tier) |
| `_generate_image_openrouter(prompt, key)` | Calls OpenRouter Gemini flash fallback |
| `generate_image(prompt, key, provider)` | Dispatcher — routes to gemini_direct or openrouter |
| `apply_paper_grain(img_bytes)` | Adds PIL/numpy noise for paper texture |
| `apply_ken_burns(img_path, duration)` | moviepy zoom/pan clip |
| `apply_runway_video(img_path, idx, key)` | RunwayML Gen3 image-to-video |
| `assemble_kburns(scenes, images, audio)` | Concatenates Ken Burns clips with crossfade |
| `assemble_aivid(scenes, clips, audio)` | Concatenates RunwayML clips |

### generate.py

| Function | Purpose |
|----------|---------|
| `load_env()` | Same pattern — local .env, videoSequence fallback |
| `group_segments(segments, key)` | DeepSeek scene grouping |
| `write_prompt(scene, key)` | DeepSeek → gpt-image-1 prompt |
| `generate_image(prompt, key)` | OpenAI gpt-image-1, returns PNG bytes |
| `dry_run(scenes)` | Scene table + cost estimate |

---

## Known Issues / Notes

- `GROUP_SYSTEM` in `generate.py` (line ~133) still contains a placeholder stub — production grouping quality improvements should start there
- `CHARACTER_STYLE` constant in `generate.py` is dead code — style is embedded in `PROMPT_SYSTEM`
- RunwayML `gen3a_turbo` outputs 1280×768; Ken Burns uses original image resolution
- `moviepy<2.0` pinned — v2 has breaking API changes
