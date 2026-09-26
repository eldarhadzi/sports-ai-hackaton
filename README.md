# Sports Intelligence — CV Prototype (Karate Kumite)

Offline computer-vision prototype that analyzes karate kumite videos: tracks two
athletes, extracts pose + movement data, and produces stats/heatmaps.

This is a **separate project**, independent from Klub.ba. Nothing here touches
the Klub.ba codebase, Supabase, or any cloud/auth/frontend.

## Environment (checked 2026-09-26)

| Item | Result |
|---|---|
| OS | macOS 26.5.1 |
| CPU | Apple M1 (arm64, Apple Silicon) |
| Project Python | 3.11.16 (`.venv`, via Homebrew `python@3.11`) |
| Acceleration | Apple **MPS available** (confirmed via `torch.backends.mps.is_available()`); no CUDA (no NVIDIA GPU) — device order is `cuda` → `mps` → `cpu` |
| Packages installed (in `.venv`) | `ultralytics 8.4.163`, `opencv-python 5.0.0`, `numpy 2.4.6`, `pandas 3.0.6`, `matplotlib 3.11.2`, `torch 2.14.0`, `pytest 9.1.1` |
| Free disk space | 23 GiB free of 228 GiB (42% used) — fine for now, but videos/models/outputs can add up |

## Setup

```bash
# from the sports-ai/ folder
python3.11 -m venv .venv        # already done
source .venv/bin/activate
pip install -r requirements.txt  # already done
```

Device selection (`cuda` / `mps` / `cpu`, or `auto`) is configurable in
`config.yaml`. On this Mac, `auto` resolves to `mps`; if MPS misbehaves for
pose estimation, `PoseTracker` falls back to `cpu` automatically and logs it
(see Known limitations — this fallback path itself hasn't been triggered by
real testing yet).

## How to run

```bash
source .venv/bin/activate
python src/main.py --input videos/input/match.mp4 --output videos/output/match_analyzed.mp4
```

Add `--calibrate` to click the 4 tatami corners on the first frame before
processing (saved per-video to `data/calibration/<name>.json` and reused on
later runs of the same clip).

Outputs land in `videos/output/` (annotated video) and `data/` (`<name>.jsonl`,
`<name>_run_meta.json`, `<name>_stats.json`, `<name>_athlete_{1,2}_heatmap.png`).

## Running tests

```bash
source .venv/bin/activate
pytest tests/
```

Covers the pure logic only (movement stats math, athlete selection and
tracker-ID re-association) — no model/video required. Currently 5/5 passing.

## Project layout

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

## Licensing note

Ultralytics YOLO is **AGPL-3.0**. Fine for this local prototype; matters before
any commercial use inside Klub.ba. The model sits behind a `PoseTracker`
interface so a permissively-licensed alternative can be swapped in later.

## Data & privacy

These clips may contain minors. `videos/`, `models/`, and all generated outputs
are git-ignored. Nothing is ever uploaded — everything stays local.

## Known limitations

**Nothing below has been validated on a real kumite clip yet.** The pipeline
has only been smoke-tested end-to-end on a synthetic random-noise video (to
catch crashes in video I/O, model loading, export, and heatmap code) — that
tells us the code runs, not that tracking/athlete-selection/stats are
accurate. I need 3–5 real clips (see "Testing" below) before I can report
honest per-clip numbers here (ID switches, missing-pose frames, where the
skeleton visibly fails).

Known, not-yet-tested design tradeoffs to watch for once real clips arrive:

- **`tracking.min_track_len` (currently 10 frames)** delays how fast a new
  tracker ID becomes eligible to replace a lost athlete after an occlusion —
  there will be a short gap (a few frames) right after a crossing/clinch
  where an athlete has no ID assigned. This is intentional (filters
  audience/spurious detections) but the gap length needs tuning against real
  footage.
- **Belt color (red/blue) as a re-association signal was investigated, not
  integrated.** `athletes.estimate_belt_hue()` exists as a proof of concept
  (samples HSV in the lower third of the bbox) but is not wired into
  `AthleteSelector` — it hasn't been validated against real lighting/tatami
  color/belt visibility during grappling, and a wrong color read could make
  re-association worse, not better. Feasibility verdict pending real footage.
- **MPS vs CPU for pose**: `PoseTracker` falls back from MPS to CPU
  automatically on error and logs it, but this fallback path itself hasn't
  been exercised (no MPS failure has occurred yet in testing).
- Heatmap alignment when overlaid on a video frame (no calibration) is
  approximate — it uses the image bounding box of observed points, not exact
  pixel-for-pixel alignment with the source frame's coordinate grid.

## Status

**Milestones 1–6 built** (video I/O, pose+tracking, athlete selection with ID
re-association logging, rendering, JSONL/run_meta export, movement stats).
**Milestones 7–8 built** (manual tatami calibration + homography, heatmaps)
but calibration's interactive click flow hasn't been exercised on a real
frame yet. Unit tests cover the pure logic (movement stats, athlete
selection/re-association) and pass; `pytest` in the venv runs them.

**Next step: I need 3–5 real clips in `videos/input/`**, labeled by scenario
(normal exchange, athletes crossing, clinch/overlap, fast kicks, athlete
partly out of frame, referee between athletes), so I can run the pipeline for
real, report actual ID switches / missing-pose frames / skeleton failures per
clip, and commit per milestone as intended.
