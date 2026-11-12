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
later runs of the same clip). Since clicking needs a real display, there's
also a non-interactive path:

```bash
python src/main.py --input videos/input/match.mp4 --output videos/output/match_analyzed.mp4 \
  --corners "x1,y1,x2,y2,x3,y3,x4,y4"   # top-left,top-right,bottom-right,bottom-left, pixels
```

**Calibration matters a lot, not just for meters**: without it, seated
corner coaches/officials can outrank the actual athletes in the "largest,
most persistent track" heuristic and get wrongly labeled Athlete 1/2 (see
Known Limitations). Skip it only for a moving/panning camera, where a single
homography can't hold anyway (stats stay in pixel space).

Outputs land in `videos/output/` (annotated video) and `data/` (`<name>.jsonl`,
`<name>_run_meta.json`, `<name>_stats.json`, `<name>_athlete_{1,2}_heatmap.png`).

## Running tests

```bash
source .venv/bin/activate
pytest tests/
```

Covers the pure logic only (movement stats math, athlete selection and
tracker-ID re-association) — no model/video required. Currently 6/6 passing.

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

Tested on 6 real clips from what looks like the same competition broadcast
(`videos/input/kumite_0{1..6}_*.mp4`, all excluded from git). Five are a
static wide shot (`siroki_kadar`, 640x360, ~24fps) sharing one camera
position — calibrated with the same manually-eyeballed tatami corners
(`--corners "185,195,460,195,495,335,150,335"`). The sixth is a closer,
panning camera (`medium_kamera_prati`) — left uncalibrated (see below), pixel
space only.

### Bugs this testing round found and fixed

Running on real footage (not just the earlier synthetic smoke test) surfaced
four real, confirmed bugs — this is exactly why the plan called for real
clips before trusting the pipeline:

1. **Without tatami calibration, seated corner coaches got locked in as
   "Athlete 1/2" instead of the real fighters**, for most of clip 1 — visibly
   wrong (screenshotted and confirmed). The "largest, most persistent track"
   heuristic alone isn't enough; a stationary seated official can easily
   out-outrank a fighter's track right after an occlusion resets it. Fixed by
   requiring tatami calibration and always restricting candidates to
   ground-points-inside-the-mat once calibrated (see #2).
2. **A related bug**: even with calibration, if fewer than 2 people currently
   had a ground point inside the tatami (e.g. one athlete briefly undetected),
   the code fell back to the *unfiltered* candidate list — silently
   reintroducing the same seated-coach problem. Fixed: once calibrated, only
   ever consider people currently inside the tatami, even if that leaves 0 or
   1 valid candidates that frame.
3. **`tracking.min_bbox_area_frac` (was `0.01`) excluded both real athletes
   entirely** in clip 4 (a more zoomed-out "pauza" framing) while a seated
   coach's larger box passed — 0% athlete visibility, confirmed by inspecting
   actual bbox areas in the exported JSONL. Lowered to `0.002`. Across all 5
   wide-shot clips, YOLO never produced a persistent track for a background
   audience member at all (they're too small/dense to detect as individuals
   at this resolution) — so this filter isn't doing meaningful anti-audience
   work in this footage; athlete-vs-official separation relies on tatami
   calibration and the size/persistence ranking, not this threshold.
4. **`direction_changes` was silently always 0 once calibrated** — the
   pixel-space movement threshold (`min_move_speed_px: 40`) was being reused
   against meter-space speeds (~1-2 m/s), which never cleared it. Fixed by
   adding a separate `min_move_speed_mps` used only when calibrated.

A fifth issue was found but only mitigated, not fully fixed: re-associating a
lost athlete to a new tracker ID used to force the *nearest available* match
even when it was clearly too far (>15% of the frame diagonal) to trust —
which is exactly how bug #1 locked onto a coach. Now a distance-gated match
is required; if nothing is close enough, the athlete is left unassigned that
frame instead of confidently wrong. Tradeoff: a genuinely fast repositioning
during a longer occlusion could also get blocked and show up as a longer gap
— hasn't been separately isolated from other causes of low visibility below.

### Per-clip results (after the fixes above)

| Clip | Frames | Calibrated | ID switches | Visible (A1/A2) | Notes |
|---|---|---|---|---|---|
| kumite_01_siroki_kadar | 665 | yes | 1 | 72% / 72% | Frames ~590-664 (~11% of clip): the two fighters are so close YOLO emits only **one** bounding box for both — neither athlete slot has 2 valid candidates, both correctly left unassigned rather than guessing. |
| kumite_02_siroki_kadar | 868 | yes | 8 | 62% / 52% | Brief unassigned gaps around a fast exchange / penalty call (~frame 162); more ID churn than clip 1. |
| kumite_03_siroki_kadar | 312 | yes | 7 | 73% / 43% | At the exact frame of a scoring Ippon kick (~frame 176), **both** athletes lost their assignment simultaneously for a few frames — fast, blurred motion breaks tracking right when it matters most. |
| kumite_04_siroki_kadar_pauza | 352 | yes | 2 | 81% / 47% | Mostly static stand-off; was 0%/0% before bug #3 above was fixed. |
| kumite_05_siroki_kadar | 944 | yes | 20 | 43% / 58% | Longest, highest-action bout tested — most ID churn of the 5 calibrated clips, consistent with more exchanges/occlusions. |
| kumite_06_medium_kamera_prati | 737 | no (panning camera) | 14 | 97% / 80% | Closer framing gives YOLO bigger, clearer boxes — best visibility of all 6 clips — but a moving camera can't use a single static homography, so this one stays in pixel space by design. |

Skeletons themselves (pose quality) looked visually solid whenever a person
was detected at all, across all 6 clips, including during kicks — the
failures above are tracking/assignment failures (who is this?), not pose
failures (where are their joints?).

### Still-open / by-design limitations

- **`peak_speed` in meters is noisy and can be implausible** (10-21 m/s
  observed — faster than a sprinter) even after excluding the "teleport"
  jump across a tracker-ID re-association. Root cause: the single-homography
  tatami mapping amplifies pixel-level jitter unevenly — the same few pixels
  of smoothing residual translate to a much bigger real-world distance in the
  part of the mat farther from the camera (perspective compresses more real
  space into fewer pixels there) than close to it. `avg_speed` and
  `total_distance` are far more stable since they average over many frames;
  treat `peak_speed` as an upper bound / noisy signal, not a measurement, for
  V1.
- **`tracking.min_track_len` (10 frames)** delays how fast a reappearing
  athlete gets reassigned after an occlusion — a short, real "blind gap"
  after every occlusion, by design (filters spurious/short-lived detections).
- **Belt color (red/blue) as a re-association signal was investigated, not
  integrated.** `athletes.estimate_belt_hue()` exists as a proof of concept
  but isn't wired into `AthleteSelector`, and wasn't validated this round
  either — still an open feasibility question, not a "yes" or "no".
- **MPS**: ran all 6 real clips plus the synthetic smoke test on Apple MPS
  with zero fallback triggers — the MPS→CPU fallback path in `PoseTracker`
  itself remains untested (no MPS failure has occurred yet).
- **Clip 6's camera pans/zooms**, so tatami calibration fundamentally can't
  apply there (a static homography assumes a static camera) — not a bug, a
  structural limit of the manual single-homography approach for V1.
- Heatmap alignment when overlaid on a video frame (no calibration) is
  approximate — it uses the image bounding box of observed points, not exact
  pixel-for-pixel alignment with the source frame's coordinate grid.

## Status

**All 8 milestones built and run on real footage**: video I/O, pose+tracking,
athlete selection with ID re-association logging, rendering, JSONL/run_meta
export, movement stats, manual tatami calibration + homography, heatmaps.
6/6 unit tests passing. Interactive click-to-calibrate is implemented but
untested (no display in this environment) — used `--corners` instead, which
covers the same code path minus the mouse callback.

**Next step**: your call — more clips (especially closer/harder angles like
clip 6, or the crossing/clinch/out-of-frame scenarios not clearly represented
yet), or move on to tuning (e.g. the reassociation distance threshold, or
`min_track_len`) against what's already here.
