# Run

## Prerequisites
- `uv` installed
- `.env` with API keys (see below)

## API Keys (.env)

```
DEEPSEEK_API_KEY=...         # required for scene grouping (both scripts)
GEMINI_API_KEY=AIza...       # free tier ~1500 req/day — used first if set
# OR (Google AI Studio names it this):
GOOGLE_API_KEY=AIza...       # accepted as alias for GEMINI_API_KEY
OPENROUTER_API_KEY=sk-or-... # fallback if GEMINI_API_KEY not set (~$0.02/image)
OPENAI_API_KEY=sk-proj-...   # required for generate.py (gpt-image-1)
RUNWAYML_API_KEY=...         # only needed for --style aivid
```

---

## test_animate.py — Phase 1 test (identity-locked character)

Outputs to `output/phase1_test/`. Never overwrites production frames.

```bash
# Dry run — preview scene plan, zero cost
uv run test_animate.py --dry-run

# Dry run for 50s clip
uv run test_animate.py --dry-run --seconds 50

# Generate Ken Burns video (cost prompt appears before spending)
uv run test_animate.py --style kburns

# Generate 50s Ken Burns video
uv run test_animate.py --style kburns --seconds 50

# Fixed 4s intervals instead of scene grouping (50s = 13 images)
uv run test_animate.py --interval 4 --seconds 50 --style kburns

# Both Ken Burns + RunwayML AI video
uv run test_animate.py --style both --seconds 50

# Custom source MP4
uv run test_animate.py --source video/my_video.mp4 --style kburns
```

### Scene strategy (default — no --interval flag)
DeepSeek groups the transcript into semantic scenes, then each scene is
capped by emotional tone before generating images:

| Tone | Max duration |
|------|-------------|
| anxiety/stress | 3s |
| breakthrough / neutral | 5s |
| growth/healing | 6s |
| heavy trauma / numbness | 8s |

### Output
```
output/phase1_test/
├── images/          ← scene_001.png, scene_002.png, ...
├── aivid_clips/     ← RunwayML clips (--style aivid only)
├── kburns.mp4       ← Ken Burns video
└── aivid.mp4        ← AI video
```


---

## animate.py — Code-rendered animation (zero image cost)

Uses matplotlib — no API calls for image generation.

```bash
# Dry run — preview scene plan
uv run animate.py --dry-run

# Render full animation
uv run animate.py

# Custom source
uv run animate.py --source video/my_video.mp4
```

### Output
```
output/
└── animation.mp4
```

---

## Notes
- Re-running is safe — existing images/frames are skipped
- Dry-run only calls DeepSeek for grouping (~$0.01), no image cost
- `test_animate.py` auto-detects the first `.mp4` + matching `.txt` in `video/`
- Prompt templates live in `prompts/` — edit without touching Python







---

## generate.py — Production image generation

Uses gpt-image-1 via OpenAI. Requires `OPENAI_API_KEY`.

```bash
# Dry run — preview scenes, no images
uv run generate.py --dry-run --transcript video/clean_How_To_Never_Get_Angry.txt

# Generate all images (~$1.00 for ~25 scenes)
uv run generate.py --transcript video/clean_How_To_Never_Get_Angry.txt

# Uses transcript.txt in root by default
uv run generate.py --dry-run
```

### Output
```
output/
├── frames/          ← 001_0s.png, 002_36s.png, ... (1536×1024)
└── results.json     ← scene log with prompts, timestamps, costs
```
