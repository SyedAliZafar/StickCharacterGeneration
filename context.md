# Project Context

## Where This Fits

```
NotebookLM
  → generates audio podcast (.wav / .mp3)
  → exports timestamped transcript (.txt)   ← this pipeline reads this

stick_character_automation/
  → reads transcript.txt
  → generates ~25 whiteboard PNG frames
  → output/frames/001_0s.png ...

Video editor (CapCut / Premiere / FFmpeg)
  → syncs frames to audio timeline
  → exports final YouTube video
```

---

## Sibling Projects

| Folder | Purpose |
|--------|---------|
| `videoSequence/` | Sequences & deduplicates frames extracted from existing AI-generated videos |
| `whiteboard-gen/` | Experimental — SVG + FLUX Schnell hybrid (cheaper, lower quality) |
| `stick_character_automation/` | THIS — production pipeline, DALL-E 3, consistent character style |
| `psych-youtube-automation/` | Script writing, ideation, NotebookLM prompts |
| `ThumbnailGeneration/` | YouTube thumbnail creation |

---

## Why DALL-E 3 (not FLUX Schnell)

The psychology scripts are metaphor-heavy:
- "anger is a heavily armored security guard protecting something soft"
- "dry wood you've been carrying around"
- "watching the storm from behind a window"
- "lifeguard throwing a life preserver"

FLUX Schnell (cheap, $0.003/image) cannot follow metaphorical prompts reliably.
DALL-E 3 ($0.04/image) understands nuanced scene descriptions.

For ~25 images per video: DALL-E 3 = **$1.00 total**. Not worth compromising quality.

---

## Why DeepSeek for Grouping + Prompts

- Already have OpenRouter key (routes to DeepSeek-V3)
- DeepSeek-V3 is excellent at structured JSON output
- ~4× cheaper than Claude Sonnet for this text task
- Grouping cost: negligible (~$0.01 for the full script)
- Claude reserved for vision tasks (frame matching in videoSequence)

---

## Why Group First

A 518-second script has ~100 raw segments (3–8 seconds each).
Most segments continue the same visual idea across 10–20 seconds.

Generating one image per raw segment = 100 images = $4.00 + visual chaos.
Grouping into scenes first = 25 images = $1.00 + clean visual storytelling.

DeepSeek sees the entire script before deciding boundaries — it understands
narrative arc, not just individual lines.

---

## Character Style Decision

All prompts share the same style prefix:
> "whiteboard animation, black marker on white background, simple stick figure
> with round head and thin limbs, clean hand-drawn educational illustration, no color, no shading,"

This creates a consistent visual identity — viewers recognize the channel's style
across every video without any extra work.

---

## API Keys Available (from videoSequence/.env)

```
OPENAI_API_KEY       ← DALL-E 3 (already have this)
OPENROUTER_API_KEY   ← DeepSeek-V3 (already have this)
DEEPSEEK_API_KEY     ← direct DeepSeek fallback (already have this)
```

Copy `.env` from `videoSequence/` — all keys are already there.

---

## Estimated Cost Per Video

| Item | Cost |
|------|------|
| DeepSeek grouping + prompt writing (~100 segments) | ~$0.01 |
| DALL-E 3 × 25 images (standard, 1792×1024) | ~$1.00 |
| **Total per video** | **~$1.01** |

At 4 videos/month = ~$4/month for all images.

---

## Future: When to Add Supabase

Add Supabase when:
- Managing 5+ videos simultaneously
- Want to avoid re-generating already-paid-for images
- Want a web-based review UI (not just local files)
- Going multi-channel

Until then: JSON files + local folders is sufficient.
