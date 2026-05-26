# Run

## .env keys
```
DEEPSEEK_API_KEY=...         # required
GEMINI_API_KEY=AIza...       # free, ~1500/day (preferred)
OPENROUTER_API_KEY=sk-or-... # fallback, ~$0.02/image
OPENAI_API_KEY=sk-proj-...   # generate.py only
RUNWAYML_API_KEY=...         # --style aivid only
```

---

## test_animate.py — test run (3 random scenes)

```bash
uv run test_animate.py --dry-run                          # preview, zero cost
uv run test_animate.py --style kburns                     # Ken Burns (P2+P3+P4)
uv run test_animate.py --style kburns --motion cinematic  # cinematic motion
uv run test_animate.py --style kburns --subtitles         # burn keyword captions
uv run test_animate.py --thumbnail                        # thumbnail only
uv run test_animate.py --style kburns --no-cache          # force fresh generation
uv run test_animate.py --style kburns --regenerate        # delete + redo images
uv run test_animate.py --interval 4 --seconds 50          # fixed 4s intervals
```

Output → `output/phase1_test/` (kburns_phase2/3/4.mp4)

---

## test_animate.py — full production run (all scenes)

```bash
uv run test_animate.py --full-run --dry-run --provider openrouter   # preview
uv run test_animate.py --full-run --style kburns --motion cinematic --provider openrouter
```

Output → `output/full_run/final.mp4` · Restarts safe (validated frames reused)

---

## animate.py — code-rendered (zero cost)

```bash
uv run animate.py --dry-run
uv run animate.py
```

Output → `output/animation.mp4`

---

## generate.py — production images (gpt-image-1)

```bash
uv run generate.py --dry-run --transcript video/my.txt
uv run generate.py --transcript video/my.txt
```

Output → `output/frames/` · `output/results.json`

---

## Tone duration caps

| Tone | Cap |
|---|---|
| anxiety/stress | 3s |
| breakthrough / neutral | 5s |
| growth/healing | 6s |
| heavy trauma / numbness | 8s |
