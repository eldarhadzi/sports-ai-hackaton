# Sports Intelligence — CV prototype (Karate Kumite)

You already know Klub.ba (React + Supabase SaaS for sports clubs). This task is a **new, separate module**: an offline computer-vision prototype that analyzes karate kumite videos. It will integrate with Klub.ba much later — not now.

## Where to build

- Create this as a **separate project folder/repo called `sports-ai/`, outside the Klub.ba app code.**
- **Do not modify any existing Klub.ba code, Supabase schema, edge functions or config.**
- If you are currently running inside the Klub.ba repo, stop and ask me where to create `sports-ai/` before writing anything.

## Goal of V1

Take a normal recorded kumite video and produce:

1. an annotated output video with two tracked athletes (persistent IDs "Athlete 1" / "Athlete 2"), bounding boxes, pose skeletons and movement trails;
2. structured per-frame data (JSONL) usable for analytics without re-running the video;
3. basic movement statistics per athlete;
4. per-athlete heatmaps.

Everything runs **offline on my local machine**. No cloud, auth, payments, frontend, database or Klub.ba integration.

Target command:

```
python src/main.py --input videos/input/match.mp4 --output videos/output/match_analyzed.mp4
```

## Step 1 — Inspect the environment, then STOP

Before writing any real code, check and report:

- OS and CPU architecture (I'm most likely on a Mac with Apple Silicon).
- Python version (need 3.11+), and whether a venv exists.
- Available acceleration: Apple **MPS**, NVIDIA **CUDA**, or CPU only. Don't assume a GPU.
- Whether `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib` are installed.
- Free disk space.

Then write a short `README.md` (setup + how to run) and **stop and wait for my OK** before continuing.

## Tech stack

- Python 3.11+, OpenCV, Ultralytics YOLO, NumPy, Pandas, Matplotlib.
- Device selection: `cuda` if available, else `mps`, else `cpu`. It must be configurable in `config.yaml`. If MPS gives errors or wrong results for pose, fall back to CPU and **document it**.
- Keep the code portable to Linux/ARM: the long-term target is an **NVIDIA Jetson**, so no macOS-only dependencies.

## Pipeline design (important, keep it simple)

- Use **one YOLO pose model with Ultralytics' built-in tracking** (`model.track(..., persist=True)` with BoT-SORT or ByteTrack). One pass gives bbox + confidence + 17 COCO keypoints + track ID. Don't run a separate detector.
- Start with a small model (e.g. `yolo11n-pose`), and make the model name configurable so I can compare `n` / `s` / `m`.
- Wrap the model behind a thin interface (`PoseTracker`), so the model or library can be swapped later.

### Picking the two athletes

A kumite video contains **other people**: the referee is usually on the tatami, plus judges, coaches and the audience. Handle this explicitly:

- Filter out small or short-lived detections.
- Choose the two most persistent, largest tracks as athletes, and log how the choice was made.
- Once tatami calibration exists (below), prefer people whose ground point is inside the tatami. The referee will still be inside, so log when there are more than two candidates.
- Tracker IDs will change after occlusions. Keep a separate mapping from tracker ID to Athlete 1/2 and re-associate by last known position. Count and log every re-association or ID switch.
- **Investigate, but only build if it is simple:** in kumite the athletes wear red (aka) and blue (ao) belts or gear. Belt color could make ID assignment and recovery after crossing much more reliable. Report whether it looks feasible.

### Position = ground point, not bbox center

For position, movement and heatmaps, use the athlete's **ground point**: the midpoint of both ankles if their keypoint confidence is high enough, else the bottom-center of the bbox. The bbox center floats during kicks and breaks the homography later. Apply light smoothing (e.g. a moving average) and make the window configurable.

## Data export

- Write **JSONL**, one record per athlete per frame: frame, timestamp, athlete_id, tracker_id, bbox, confidence, keypoints (x, y, conf), ground_point, and `tatami_point` (null until calibrated).
- Also write a `run_meta.json` with: model, tracker, device, config snapshot, input fps/resolution, frames processed, processing time and effective FPS, and the ID switch count.
- Keep raw videos, extracted data and rendered outputs in separate folders.

## Movement analysis

Per athlete, compute:

- total distance, average and peak speed;
- significant direction changes, with the threshold defined in config;
- time visible in frame;
- position over time.

Before calibration, **label all values as image-space (pixels)**. Never present pixels as meters.

## Tatami calibration (manual)

- Let me click the 4 corners of the competition area on a frame, in an OpenCV window, or type them into config.
- Save the calibration per video to JSON.
- Tatami dimensions are configurable, with the WKF competition area of 8 × 8 m as default.
- Compute a homography and fill `tatami_point` in approximate meters. Metric stats get recomputed from it and labeled "approximate".
- No automatic tatami detection in V1.

## Heatmaps

- One heatmap image per athlete.
- With calibration: on a top-down tatami grid. Without calibration: overlaid on a video frame, in image space.

## Testing — honesty over demos

- I will provide 3–5 short real clips. Ask me for them when needed.
- Label each clip by scenario: normal exchange, athletes crossing, clinch/overlap, fast kicks, athlete partly out of frame, referee between athletes.
- For each clip, report: ID switches, frames with missing pose, and where the skeleton visibly fails (fast kicks, overlap).
- Put all of this in a **"Known limitations"** section in the README. Do not claim something works if you haven't run it on a real clip.

## Data & privacy

These videos may contain **minors**:

- Add `videos/`, `models/` and all outputs to `.gitignore`. Never commit video or extracted data.
- Never upload videos or data anywhere. Everything stays local.

## Licensing note

Ultralytics YOLO is **AGPL-3.0**. That's fine for a local prototype, but it matters before any commercial use inside Klub.ba. Note this in the README, and keep the model behind the `PoseTracker` interface so alternatives with permissive licenses can be tested later.

## Suggested structure

Adapt as needed; don't add abstractions we don't use yet.

```
sports-ai/
  videos/input/  videos/output/  models/  data/
  src/
    main.py
    pose_tracker.py      # YOLO pose + tracking wrapper
    athletes.py          # choosing the 2 athletes, ID mapping
    calibration.py       # tatami corners + homography
    analysis/karate.py   # movement stats (sport-specific layer)
    visualization.py     # boxes, skeletons, trails, heatmaps
    export.py            # JSONL + run_meta
  tests/
  config.yaml
  requirements.txt
  README.md
```

The detection, tracking and pose layers should be shared across sports. Only `analysis/` becomes sport-specific later (karate, taekwondo, football).

## Out of scope for V1 — do NOT build

Technique classification, scoring, referee decisions, foul detection, LLM reports or AI coaching, cloud, frontend, auth, payments, database, Klub.ba integration, multi-camera, sensors, real-time streaming.

## How to work

1. Step 1 (environment), then stop for my OK.
2. Then go milestone by milestone:
   1. video I/O;
   2. pose + tracking;
   3. athlete selection + ID mapping;
   4. rendering;
   5. export;
   6. movement stats;
   7. calibration;
   8. heatmaps.
3. After each milestone:
   - run it on a real clip;
   - commit;
   - tell me briefly what works, what doesn't, and the next technical bottleneck.
4. Don't move on to technique recognition until tracking and pose are stable.
5. Prefer simple, readable Python. Log processing time. Make failures visible.

## Definition of done (V1)

I put a real kumite clip in `videos/input/` and run the target command. I get:

- an annotated video: two athletes with stable IDs, boxes, skeletons and trails;
- a JSONL data file and `run_meta.json`;
- a stats summary (image-space, or approximate meters if calibrated);
- two heatmaps;
- a README with honest known limitations.
