# Run

## Prerequisites
- `uv` installed
- `.env` contains `DEEPSEEK_API_KEY` and `OPENAI_API_KEY`

## Commands

```bash
# Preview scenes only — no images, ~$0.01
uv run generate.py --dry-run --transcript video/clean_How_To_Never_Get_Angry.txt

# Generate all images (~$1.00)
uv run generate.py --transcript video/clean_How_To_Never_Get_Angry.txt

# Default transcript path (transcript.txt in root)
uv run generate.py --dry-run
```

## Output
- `output/frames/` — PNG images (`001_0s.png`, `002_36s.png`, ...)
- `output/results.json` — scene log with prompts, timestamps, costs

## Notes
- Re-running is safe — existing frames are skipped
- Dry-run calls DeepSeek only (no DALL-E, no image cost)
