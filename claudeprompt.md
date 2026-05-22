
phase 2 prompt 

CONTEXT: test_animate.py is the production pipeline. Phase 1 is complete. Read test_animate.py fully before editing. Test output goes to output/phase1_test/. Source files are in video/. Use only first 20 seconds of transcript for test runs.
TASK — Phase 2: Motion, transitions, emotional camera
1. Fix crack consistency
The red crack is changing shape too much between scenes. Lock it down:

Define ONE canonical crack path in prompts/prompt_system.txt with exact language: "a single bold jagged red crack that starts at the top-center of the head, splits slightly left at the midpoint, and ends just below the chin — this exact crack shape appears in every single frame, never branches, never changes path"
Remove any prompt language that says "branching" or "spiderweb" — those are causing the variation
Crack intensity (width, glow) can vary by tone but the PATH never changes

2. Emotional Ken Burns camera behavior
Replace the current generic zoom with tone-driven camera logic using PIL/numpy. Each scene gets a camera behavior based on its emotional_tone:

heavy trauma → slow push-in, start wide, end 15% closer, move slightly downward
anxiety/stress → micro jitter, 2-3px random frame offset every 8 frames, slight zoom instability
growth/healing → gentle horizontal drift left to right, neutral zoom
breakthrough → start close, drift upward and slightly out, opening feeling
numbness → very slow zoom-out, start slightly cropped, end wide, barely moving
neutral → slow diagonal drift, low intensity

Implement this in a apply_ken_burns(frames, tone, duration_s) function that takes the static image and returns a list of PIL frames with the motion applied.
3. Scene transitions
Add transitions between scenes. Implement these two first (simplest, most effective):

shadow_fade — outgoing image fades to paper tone #F5F1E8, incoming fades in. 18 frames total (0.75s at 24fps)
crack_spread — a red crack line draws across the frame left to right over 12 frames, then incoming image appears behind it. Use PIL ImageDraw to draw the line frame by frame

Pick transition based on tone change:

trauma → anything: shadow_fade
anxiety → anything: crack_spread
everything else: shadow_fade

4. Fix scene concept labels
The grouping is producing identical concept labels when one transcript section gets split by tone. Fix this: when a scene is split into sub-scenes, append a direction to each label — e.g. "Storm of frustration intro → rising", "Storm of frustration intro → peak", "Storm of frustration intro → breaking"
5. Add procedural film grain per frame
Apply subtle grain to every frame using numpy before saving:

Generate gaussian noise array, mean=0, std=6, same size as frame
Add to frame, clip to valid range
This should be barely visible but make it feel hand-made

6. Test run

Same as Phase 1: first 20 seconds only, output/phase1_test/, cost estimate + confirmation before generating
After generating, print a summary: scene count, transitions used, camera behaviors applied, total frames rendered
Save the output video as output/phase1_test/kburns_phase2.mp4 so you can compare side by side with Phase 1

CONSTRAINTS:

Do not break --dry-run, --source, --transcript, --style, --provider flags
All motion must be done with PIL/numpy — do not add opencv as a dependency
Keep uv run inline deps format
If a tone isn't in the camera behavior map, default to neutral