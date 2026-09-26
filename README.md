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
| System Python | 3.9.6 (`/usr/bin/python3`) — **too old, need 3.11+** |
| Python 3.11+ available | **No** — not via Homebrew, pyenv, or uv |
| Existing venv | None |
| Acceleration | No GPU. Apple MPS should be available once PyTorch is installed; no CUDA (no NVIDIA GPU) |
| Packages installed (system Python) | `numpy` only — none of `ultralytics`, `opencv-python`, `pandas`, `matplotlib`, `torch` |
| Free disk space | 23 GiB free of 228 GiB (42% used) |

**Action needed before any code runs:** install Python 3.11+ (e.g. via Homebrew:
`brew install python@3.11`) and create a project-local virtual environment. The
system Python 3.9.6 will not be used for this project.

## Setup (once Python 3.11+ is installed)

```bash
# from the sports-ai/ folder
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Device selection (`cuda` / `mps` / `cpu`) is configurable in `config.yaml`.
On this Mac, `mps` will be the default; if MPS misbehaves for pose estimation
it will fall back to `cpu` (documented in Known limitations below, once tested).

## How to run

```bash
python src/main.py --input videos/input/match.mp4 --output videos/output/match_analyzed.mp4
```

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

Not yet applicable — no pipeline exists yet. Will be filled in honestly after
each milestone is run on a real clip (per-clip ID switches, missing-pose
frames, skeleton failure cases).

## Status

**Step 1 (environment check) complete.** Waiting for OK before building the
pipeline. See table above — Python 3.11+ needs to be installed first.
