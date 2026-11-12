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

## Athlete selection rework (round 2, following visual review of all 6 clips)

You reviewed the round-1 output and flagged that detection/pose were fine but
**athlete selection and ID assignment** were the real problem: referee/coach
mislabeled as an athlete (clip 03 ~2s, clip 04 ~10s, clip 06), standing
athletes undetected at the start of clips 02/04, lost athletes never
recovering (clip 05 ~26s), AKA/AO not matching actual belt colors and
swapping (clip 06), broadcast graphics detected as people, and trails drawing
straight lines across ID jumps. Below is what changed, in the order you asked
for (A→E), with before/after numbers — not just claims.

### A. Detection recall

Built a separate probe (`model.predict`, no tracking) over sampled frames
from all 6 clips, sweeping `yolo11{n,s,m}-pose` × `conf {0.10, 0.25}` ×
`imgsz {640, 1280}`, counting raw person-detection coverage per frame:

| Setting | avg % frames with ≥2 people detected | avg fps (MPS) |
|---|---|---|
| nano, conf 0.25, imgsz 640 (round-1 default) | 79.7% | 82 |
| **nano, conf 0.10, imgsz 640 (chosen)** | **100%** | **~80** |
| nano/small/medium, conf 0.10, imgsz 1280 | 100% | 5-30 |

The problem was the confidence threshold, not the model or resolution:
**conf 0.10 alone gets 100% coverage on all 6 clips** at full nano-model
speed — bigger models and imgsz 1280 weren't needed and cost real speed
(relevant for the eventual Jetson target). Also raised ByteTrack's
`track_buffer` 30→60 frames (`bytetrack_tuned.yaml`) so a ~1-2s miss doesn't
immediately force a new tracker ID. Kept: `yolo11n-pose`, imgsz 640.
Changed: `conf: 0.10` (was implicitly Ultralytics' default 0.25, never set
explicitly before).

### B. Athlete classifier (replaces "largest/most persistent track" alone)

Added `athletes.passes_athlete_filter()`, checked every frame per detection,
*before* size/persistence/tatami ranking ever runs:

- **Seated rejection** (`is_seated()`): compares thigh length (hip→knee) to
  torso length (shoulder→hip) in image space; a seated person's knees sit
  ~level with their hips instead of well below. Returns "don't know" (never
  rejects) when keypoints aren't confident enough.
- **Dark-clothing rejection** (`torso_brightness()`): mean V-channel
  brightness of the shoulder-to-hip region; a suited referee/coach is much
  darker than a white gi. Threshold: `classifier.dark_brightness_threshold`
  (90).
- **Ignore-zones** (`in_ignore_zone()`): config-defined pixel rects (for
  broadcast overlays); a detection mostly inside one is dropped. Left empty
  by default — a wrong guess here could just as easily hide a real athlete
  near the mat edge as hide a graphic, so it needs a real observed instance's
  coordinates, not a guess.

A detection failing any check is excluded from candidacy *that frame*, and a
slot whose current tracker fails is cleared immediately (not left stale) —
closing a real bug where a slot that lost its live detection could still
render/export as an athlete for extra frames because nothing had overwritten
the mapping.

### C. AKA/AO identity from belt/glove color

`athlete_id` values are now `"aka"`/`"ao"` (red/blue), not `"athlete_1"/"2"` —
drawn in matching colors. `classify_belt_color()` samples small HSV patches
around both hip keypoints (belt) and both wrist keypoints (gloves),
classifies each red/blue/unknown by hue, majority-votes across patches, and
that per-frame read feeds a 15-frame sliding-window vote per tracker
(`TrackStats.color_verdict`, needs ≥4 confident reads and ≥60% agreement).

Re-association now tries a color match **first**, bypassing the distance
gate entirely (a color match is accepted regardless of how far it is from
the slot's last known position); only when color is unknown does it fall
back to the old distance-gated nearest-match. If both slots' current
trackers read confidently as the *other* color, they're swapped outright
(`_maybe_correct_color_swap`) — this is what fixes the "labels don't match
belts" complaint, including cases where the initial bootstrap assignment
guessed wrong before any color evidence existed. Every assignment,
reassignment, and swap is logged with frame number and reason (`color match
(red)`, `distance fallback (color unknown)`, `bootstrap`, `color
correction`).

### D. Trails

A ground-point jump over `visualization.trail_break_dist_px` (80px) now
breaks the trail line instead of drawing a straight connector across an ID
jump. Trail length changed from a fixed frame count to
`visualization.trail_seconds` (2.0s), computed against each clip's actual fps.

### E. Per-clip metrics (`<name>_metrics.json`, also printed)

`analysis/metrics.py` computes, from the exported JSONL: % of frames with
both/one/zero slots assigned, ID switch count, color-swap count, and
**`filter_violations`** — frames where an assigned slot's tracker failed the
athlete filter that same frame. This should structurally always be 0 now
(assignment requires passing the filter that frame); it's a standing
correctness check, not just a diagnostic.

### Results: before (round 1) vs after (this round)

| Clip | ID switches before → after | Visible/assigned before → after | Color swaps | Filter violations |
|---|---|---|---|---|
| kumite_01 | 1 → 2 | 72%/72% (per-athlete) → 57.6% both, 13.5% zero | 0 | 0 |
| kumite_02 | 8 → 6 | 62%/52% → 46.0% both, 35.6% zero | 0 | 0 |
| kumite_03 | 7 → 6 | 73%/43% → 43.6% both, 26.0% zero | 0 | 0 |
| kumite_04 | 2 → 1 | 81%/47% → 41.5% both, 16.5% zero | 0 | 0 |
| kumite_05 | 20 → 24 | 43%/58% → 29.1% both, 25.6% zero | 0 | 0 |
| kumite_06 | 14 → **7** | 97%/80% → 81.8% both, 1.2% zero | **1** | 0 |

Note the "visible" metric isn't apples-to-apples: round 1 measured each
athlete's visibility independently; round 2's "both/zero assigned" is a
stricter joint measure (both slots correct at once), so a drop there doesn't
necessarily mean regression — it's a harder bar. **ID switches and color
swaps are directly comparable and are the primary signal**, plus the visual
spot-checks below.

**Confirmed by screenshot** (not just numbers):
- Clip 04 frame 240 (~10s): referee is now correctly unassigned (grey `id
  44`), both real athletes correctly AKA/AO. Round 1 had the referee as
  `athlete_2` here.
- Clip 06: at frame 655 AKA (red box) was on the athlete wearing the blue
  belt (wrong). At frame 659 `color_swap_count` fired
  (`AKA/AO swapped - color evidence says they're reversed`), and from frame
  670 onward AKA correctly tracks the red-belt athlete. One clean,
  logged, verifiable self-correction.
- `filter_violations: 0` on every clip confirms the "slot stuck on a
  filtered-out detection" bug class is closed, structurally, not just in the
  cases we happened to look at.

**Two clips got worse, and here's why, not just the number**:
- **Clip 01** (1→2 switches): minor, not investigated further — small sample.
- **Clip 05** (20→24 switches): a real new failure mode. Around frames
  537-573, three candidates sit inside the tatami at once (logged as
  "referee is likely among them") during a scored exchange (a real
  "referee between/near athletes" moment, visually confirmed at frame 553) —
  AO's tracker ID and the color-match logic bounce between two nearby
  candidates several times in that window. This is a genuinely hard,
  acknowledged scenario (dense pileup near a score), not a regression in the
  classifier itself — but it shows color-matching can add its own churn when
  three candidates are all momentarily plausible, which round 1's pure
  distance-based approach didn't have this specific pattern of. Not yet
  fixed; flagging honestly rather than tuning further without more clips to
  validate against.

**Clip 06 (moving camera) also showed a distinct issue**: long stretches
(dozens of consecutive frames) where AKA's color read stayed "unknown" and
the nearest candidate was beyond `reassoc_max_dist` (which is a *fraction of
the frame diagonal* — the same fraction is a much stricter absolute pixel
budget on this closer/zoomed framing than on the wide-shot clips, since
normal motion covers more pixels when the camera is closer). Per your
instruction, trails/heatmaps aren't being evaluated for this clip beyond
pose/ID debugging (`--no-heatmap` used); the distance-threshold-vs-zoom
interaction is worth revisiting if clip 06-like footage becomes a priority.

### Still-open / by-design limitations (carried over, still true)

- `peak_speed` in meters can still be implausible (homography amplifies
  pixel jitter unevenly across the mat) — unchanged this round, not
  re-investigated.
- `tracking.min_track_len` (10 frames) still delays recovery after an
  occlusion before a new tracker is even eligible - independent of the color
  matching added in C, which only applies once a track exists at all.
  Clip 05's "both athletes grey" complaint traced substantially to this plus
  the pre-color-match distance-only logic; color-match should help most such
  cases going forward but wasn't isolated as its own before/after here.
- MPS: all 6 real clips ran clean, zero fallback triggers; the MPS→CPU
  fallback path itself is still untested.
- Clip 06's moving camera still structurally can't use tatami calibration.
- Ignore-zones mechanism exists but ships with zero configured zones - the
  broadcast-graphics-detected-as-people complaint isn't fixed by default,
  it's *fixable* once a real instance's pixel coordinates are supplied.

### Round 1 results (earlier pass, for reference)

The before/after table above compares against this round. Kept here since it
has detail (the specific failure notes per clip) the numbers alone don't
capture.

Tested on 6 real clips from what looks like the same competition broadcast
(`videos/input/kumite_0{1..6}_*.mp4`, all excluded from git). Five are a
static wide shot (`siroki_kadar`, 640x360, ~24fps) sharing one camera
position — calibrated with the same manually-eyeballed tatami corners
(`--corners "185,195,460,195,495,335,150,335"`). The sixth is a closer,
panning camera (`medium_kamera_prati`) — left uncalibrated (see below), pixel
space only.

#### Round 1: bugs found and fixed

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

#### Round 1 per-clip results

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

#### Round 1 still-open / by-design limitations

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
- ~~Belt color (red/blue) as a re-association signal was investigated, not
  integrated~~ — **done in round 2** (section C above): `classify_belt_color()`
  replaced the old proof-of-concept and is wired into `AthleteSelector`.
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

**Round 1** (video I/O, pose+tracking, rendering, JSONL/run_meta export,
movement stats, manual tatami calibration + homography, heatmaps) and
**round 2** (steps A-E: detection recall tuning, athlete classifier, AKA/AO
color identity, trail breaking, per-clip metrics) are both built and run on
all 6 real clips. 11/11 unit tests passing. Interactive click-to-calibrate is
implemented but untested (no display in this environment) — `--corners`
covers the same code path minus the mouse callback.

**Not done this round**: no calibration or heatmap changes, per your
instruction to focus only on selection/ID. Open items if you want another
pass: clip 05's color-match oscillation near a dense scoring pileup, and the
zoom-dependent distance threshold on clip 06 (a fixed fraction-of-diagonal
budget is stricter in pixel terms on a closer camera).
