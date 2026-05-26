# How to Run

## Files needed before you start

```
video/
├── your_video.mp4        ← source video
└── your_video.txt        ← transcript (same filename stem as MP4)

.env                      ← API keys (DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY)
```

---

## Step 1 — Generate images + save scenes

```bash
uv run test_animate.py --full-run
```

- Calls DeepSeek to group transcript into scenes
- Generates one image per scene via Gemini (free) or OpenRouter (~$0.02/image)
- Saves frames to `output/full_run/frames/`
- Saves `output/full_run/scenes.json` (used by Step 2)
- **If frames already exist, they are skipped — no extra cost**

---

## Step 2 — Assemble video

```bash
uv run assemble_video.py
```

- Reads frames from `output/full_run/frames/`
- Reads tone data from `output/full_run/scenes.json`
- Auto-detects audio from `video/*.mp4`
- Outputs `output/full_run/final_still.mp4`

### Options

```bash
uv run assemble_video.py --subtitles              # burn keyword captions onto frames
uv run assemble_video.py --motion smooth          # add gentle zoom/pan
uv run assemble_video.py --motion cinematic       # dramatic zoom
uv run assemble_video.py --out output/my.mp4      # custom output path
uv run assemble_video.py --thumbnail              # generate thumbnail only
```

---

## Step 3 — Generate captions for CapCut

```bash
uv run caption.py
```

- Auto-detects `video/*.txt` transcript
- Outputs `video/your_video.srt` next to the transcript
- **In CapCut:** Captions → Import → select the `.srt` file

### Options

```bash
uv run caption.py --merge 1.5    # merge captions shorter than 1.5s (stops fast flashing)
uv run caption.py --out output/caps.srt   # custom output path
```

---

## Output

```
video/
└── your_video.srt        ← import into CapCut

output/
├── full_run/
│   ├── frames/           ← scene_001.png … scene_086.png
│   ├── scenes.json       ← scene metadata (tone, timing)
│   └── final_still.mp4  ← your video
└── thumbnail.png         ← if --thumbnail was used
```
