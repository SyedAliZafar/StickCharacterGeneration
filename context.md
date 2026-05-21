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

## Why gpt-image-1 (not FLUX Schnell)

The psychology scripts are metaphor-heavy:
- "anger is a heavily armored security guard protecting something soft"
- "dry wood you've been carrying around"
- "watching the storm from behind a window"
- "lifeguard throwing a life preserver"

FLUX Schnell (cheap, $0.003/image) cannot follow metaphorical prompts reliably.
gpt-image-1 ($0.04/image at medium quality) understands nuanced scene descriptions.

For ~25 images per video: gpt-image-1 = **$1.00 total**. Not worth compromising quality.

---

## Why DeepSeek for Grouping + Prompts

- Calls `api.deepseek.com` directly with `DEEPSEEK_API_KEY`
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

## Character Style Decision — TheInnerWar Brand

Every frame shares a consistent identity:
- Whiteboard background, black marker line art
- Stick figure with **perfectly round head**
- **Bold jagged RED crack** from crown downward — mandatory in every image
- Thin limbs, expressive posture; red is the only color, no shading

The crack intensity maps to emotional tone (deep/branching/faint/glowing/erased).
This is a brand requirement for TheInnerWar channel — do not remove it.

The style is embedded in `PROMPT_SYSTEM` inside `write_prompt()`. The `CHARACTER_STYLE`
constant at module level is currently unused (dead code).

---

## API Keys Required

```
OPENAI_API_KEY       ← gpt-image-1 image generation
DEEPSEEK_API_KEY     ← DeepSeek-V3 grouping + prompt writing (api.deepseek.com)
```

`.env` lookup order: `./env` first, then `../videoSequence/.env` as fallback.
All keys should already be in the `videoSequence/.env` shared file.

---

## Estimated Cost Per Video

| Item | Cost |
|------|------|
| DeepSeek grouping + prompt writing (~100 segments) | ~$0.01 |
| gpt-image-1 × 25 images (medium, 1536×1024) | ~$1.00 |
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

---

## Code Map (`generate.py`)

| Function | Purpose |
|----------|---------|
| `load_env()` | Reads `.env` — checks local first, falls back to `../videoSequence/.env` |
| `require_key(name)` | Exits with a clear message if an env var is missing |
| `parse_txt(path)` | Parses `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text` lines into `{start, end, text}` |
| `parse_json(path)` | Parses a JSON segment array (alternative input format) |
| `load_transcript(path)` | Dispatches to `parse_txt` or `parse_json` by file extension |
| `group_segments(segments, key)` | Calls DeepSeek to group all segments into scenes (one API call) |
| `write_prompt(scene, key)` | Calls DeepSeek to write a single gpt-image-1 prompt for a scene |
| `generate_image(prompt, key)` | Calls OpenAI gpt-image-1, decodes base64, returns PNG bytes |
| `dry_run(scenes, key)` | Prints scene table + cost estimate, no image API calls |
| `main()` | Orchestrates everything; skips existing PNGs |

---

## Known Issues / Incomplete Parts

- **`GROUP_SYSTEM` prompt is a placeholder**: Line ~133 contains `"...same rules..."` — this
  is a stub. The actual production prompt needs to specify grouping rules (scene count,
  boundary logic, etc.). Scene quality improvements should start here.

- **`CHARACTER_STYLE` constant is dead code**: Defined at module level but `write_prompt()`
  embeds the style directly in `PROMPT_SYSTEM`. The constant is never used.

- **`dry_run()` parameter name is misleading**: Signature is `dry_run(scenes, openrouter_key)`
  but the caller passes `deepseek_key`. Harmless — the key isn't used in dry run — but
  the parameter name is wrong.

- **Sibling project references in context.md are aspirational**: `whiteboard-gen/`,
  `psych-youtube-automation/`, and `ThumbnailGeneration/` may not exist yet.
