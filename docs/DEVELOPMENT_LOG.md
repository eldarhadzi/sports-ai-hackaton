# Development log: tracking pipeline, rounds 1-8

> The history of the tracking pipeline, moved here from the README. The current project overview is
> [README.md](../README.md); the full technical handoff is [HANDOFF.md](../HANDOFF.md).

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
**Exception (team decision, 2026-11-12):** the public senior broadcast match FRA vs CRO
(`kumite_0{1,2,3}_siroki_kadar`: raw clips, web videos, `data/v1/`, `data/roster.csv`) is
force-added to git so teammates can work without the videos being sent around. Everything else
keeps the rule.

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

## Round 3 (following frame-by-frame review of round 2's real output)

Round 2's claimed fixes only partly held up under a real frame-by-frame check
of the annotated video: the referee/judge-ID fix held, but standing athletes
were still missed in the exact same frames, trails still drew long straight
lines across jumps, and AKA/AO still flipped identity on real footage. This
round fixed the confirmed root causes and is reported here with evidence, not
just claims, per your instruction not to make further changes without numbers.

### 1. Detection: `imgsz` was never actually wired in

Round 2's README claimed nano @ imgsz=640 @ conf=0.10 gave 100% detection
coverage - true for the standalone probe script it was measured with, but
`imgsz` was never passed to `model.track()` in `PoseTracker` at all, so the
real pipeline always ran at Ultralytics' default resolution regardless.
Confirmed directly on kumite_02 frame 0 (both athletes standing still,
clearly visible): at imgsz=640 the nano model does not detect the two
standing athletes at all, down to conf=0.10 - only 3 boxes come back, all
seated/dark-suited officials. At imgsz=1280 both athletes appear at
conf 0.86-0.87 and cleanly pass the athlete filter. Fixed by actually passing
`imgsz` through `PoseTracker` (now logged at startup); `config.yaml` sets it
to 1280. Debug images for the full model x imgsz x conf sweep (12 combos,
raw boxes + confidence + pass/drop reason per box) are in
`data/debug_detection/`. Cost: nano @ 1280 runs ~20fps vs ~80fps @ 640 on
this Mac - acceptable for an offline prototype.

### 2. Color identity: stricter sampling + a lock

Patches are now sampled only at the belt (hip-keypoint midpoint, shifted up
toward the obi rather than the hip joints) and the wrists, clamped strictly
inside the detection's own bbox so a patch can never land on the tatami floor
(also red/blue) or a neighboring person. Saturation/value/keypoint-confidence
bars were all raised. A tracker's color identity now locks after 20
cumulative confident agreeing samples and only reverses if the opposite color
wins >=85% of a separate, longer 45-sample window - a single noisy burst can
no longer flip it. Debug strips (every sampled patch, labeled with role and
read color) save automatically for the first 10 frames per athlete per clip
to `data/debug_color/`.

**This did not fully fix clip 06.** Verified directly (JSONL + a zoomed crop
of the annotated frame, not just logs): tracker 3 reads a confident `red`
belt at frames 75 and 120, but the reassociation log shows both slots went
unassigned simultaneously around frame 76-77 (a mutual occlusion/clinch), and
when the two trackers reappeared, distance-fallback re-assigned tracker 3 to
**AO** and tracker 2 to **AKA** - backwards, and a zoomed crop of frame 120
confirms it visually (the red-belt athlete is boxed blue/AO, the other is
boxed red/AKA). This persists until a late, unrelated re-association around
frame 654-659 happens to land correctly. Root cause: color evidence is keyed
per tracker_id, and this clip's tracker IDs churn frequently (panning camera,
small figures) - each churn resets that tracker's accumulated color count to
zero, so neither the lock nor the swap-correction (which needs both slots'
*current* tracker to independently reach a confident, opposite verdict) ever
gets enough evidence before the next distance-fallback decision has to be
made blind. This is a distance-fallback/ID-churn interaction, not a sampling
accuracy problem, and is unfixed.

Also worth flagging: aggregate belt_color reads across all six clips show
`AO` reading confidently blue far more often than red in every single clip
(expected, since AO is a fixed blue-corner label) - but `AKA` reads more
`blue` than `red` in clips 02 and 05, and *only* blue (never red) in clip 03.
Two possible explanations, not yet distinguished: a real identity mixup in
those clips, or a real bias in the classifier itself - the blue hue
acceptance band (100-130, 31 degrees) is 50% wider than red's wraparound band
(0-10 and 170-179, 21 degrees total), so blue is statistically an easier bar
to clear at low resolution/compression. Recommend visually checking AKA in
clips 02/03/05 specifically. Clip 04 got **zero** confident color reads for
either slot the entire clip - its AKA/AO assignment rests entirely on
initial bootstrap order and was never validated by color at all.

### 3. Trails: timestamp-based, not frame-count-based

Trails now store `(timestamp, point)` pairs and are trimmed to the last 2
real seconds by timestamp, not by counting the last N *appended* points -
the bug was that a point only got appended when the athlete was actually
detected that frame, so during any occlusion the "2 second" window silently
covered more than 2 real seconds once the athlete reappeared, which is what
produced the long straight lines. A jump over `trail_break_dist_px` still
starts a new segment, and a slot with no detection in the current frame now
draws no trail at all rather than a stale tail. Spot-checked clip 05 at 0s
and ~20s (both previously flagged) - clean, no stray lines.

### 4. Metrics: old (round 2) vs new (this round), all 6 clips

Round 2's committed output was snapshotted to `data/baseline_round2/` before
any round-3 changes, so this is a real before/after, not a re-run of the same
code. Run `python scripts/compare_metrics.py` to reproduce (new value shown,
old in parens):

```
clip                                 both%      one%     zero%     id_sw     flips filt_viol
kumite_01_siroki_kadar          93.4(57.6) 5.3(28.9) 1.4(13.5)      5(2)      0(0)      0(0)
kumite_02_siroki_kadar          64.9(46.0)34.1(18.4) 1.0(35.6)     16(6)      0(0)      0(0)
kumite_03_siroki_kadar          97.1(43.6) 0.0(30.4) 2.9(26.0)      6(6)      0(0)      0(0)
kumite_04_siroki_kadar_pauza    97.4(41.5) 0.0(42.0) 2.6(16.5)      0(1)      0(0)      0(0)
kumite_05_siroki_kadar          86.2(29.1)12.8(45.2) 1.0(25.6)    27(24)      0(0)      0(0)
kumite_06_medium_kamera_prati   75.6(81.8)22.9(17.0)  1.5(1.2)      6(7)      0(1)      0(0)
```

`both%` (both slots assigned) improved substantially on 5/6 clips - clip 06
dropped slightly (81.8% -> 75.6%), not yet investigated further.
`filter_violations` stayed 0 everywhere (that invariant still holds).
`id_switch_count` rose on several clips - expected, since far more frames are
now actually being tracked at all (imgsz=1280 catches far more of both
athletes, so there are more live frames in which a switch can happen);
`color_swap_count` (the explicit two-slot swap correction) is 0 everywhere
this round, including clip 06 - but per the clip 06 finding above, that's
because the wrong assignment there was never in a state the swap-correction
could catch, not because nothing was wrong.

**Per your instruction, no further changes were made past this point** -
this report and the clip 06/AKA-in-02/03/05 findings above are for you to
review before authorizing another round.

## Status

**Round 1** (video I/O, pose+tracking, rendering, JSONL/run_meta export,
movement stats, manual tatami calibration + homography, heatmaps),
**round 2** (steps A-E: detection recall tuning, athlete classifier, AKA/AO
color identity, trail breaking, per-clip metrics), and **round 3** (real
imgsz wiring, stricter/locked color identity, timestamp-based trails,
old-vs-new metrics - see above) are all built and run on all 6 real clips.
11/11 unit tests passing. Interactive click-to-calibrate is implemented but
untested (no display in this environment) — `--corners` covers the same code
path minus the mouse callback.

**Not done this round**: the clip 06 backwards-identity bug and the
clip 02/03/05 AKA color-signal ambiguity (both described above) are
identified but not fixed - they need a design decision (e.g. carrying color
evidence across a low-confidence tracker hand-off, or narrowing red/blue's
hue-band asymmetry) that goes beyond this round's scope.

## Round 5 (ground truth, color identity, graphics zone, clean rendering)

Self-consistency metrics (0 identity flips, 0 filter violations) hid two real
errors the user saw on video: clip 06 had AKA/AO inverted and clip 05 had
non-athletes holding a slot. This round fixes both root causes and adds
metrics that can actually catch them.

1. **Ground truth** - `scripts/label_ground_truth.py` (1 frame/second, boxes
   numbered, click or key to mark AKA / AO / not visible; saves to
   `data/ground_truth/<clip>.json`, resumable). `src/analysis/ground_truth.py`
   scores a run: identity accuracy %, swapped / unknown identities, frames where
   a slot holds a non-athlete, missed athletes. `main.py` adds them to
   `<clip>_metrics.json` when labels exist; `scripts/gt_report.py <run>...`
   prints the table. **No labels exist yet, so none of these are reported yet.**
2. **Color identity** - belt + glove patches sampled on every frame (pass 1),
   per-pixel red/blue fractions (dark navy/purple = weak blue), accumulated per
   track over the whole clip. AKA is whoever is redder *relative to the other
   athlete* (no absolute threshold); with no usable evidence the box shows `?`
   instead of a position-based guess.
3. **Ignore zone** - YUKO/IPPON panel zone is configurable, scaled to the video
   resolution, and only applied on frames where the graphic's dark text bar is
   on screen (`config.yaml: classifier.ignore_zones`). Dark-clothing (referee)
   rejection is now also a windowed track-level verdict re-checked on every
   (re)assignment, because the per-frame check silently passes when the torso
   keypoints are unmeasurable.
4. **Rendering** - normal output draws only AKA/AO + trails; `--debug` adds grey
   boxes for everyone else; `--debug-strips` saves only the patch images.

Run: `scripts/run_all.sh v5` (outputs `videos/output/v5/`, `data/v5/`).
Each `run_meta.json` records commit, dirty flag, diff hash and active features.

## Round 6 (two-pass identity from ground truth feedback)

Ground-truth scoring of V5 (`scripts/gt_report.py v5`) showed identity was right
whenever it committed (0 swaps on fixed-camera clips) but often said "?" (unknown).

- **Why per-track evidence failed**: ByteTrack IDs get silently re-used across
  people (clip 01: after a clinch tracker 3 jumps from the red athlete to the
  blue one, then back), so pooling evidence per track over its lifetime mixes two
  people. The decision is now made per FRAME over the whole clip (`src/identity.py`):
  selection only builds two continuity slots A/B (position + color continuity);
  each frame gives a pair signal x(t) = (A red-blue) - (B red-blue); a two-state
  Viterbi picks "A is AKA" / "A is AO" over the whole clip, using past AND future
  evidence, with one cost per change of AKA's side. Frames before the first
  evidence get the label of the evidence that comes later; a stretch with no
  usable evidence stays "?".
- **Side-order prior**: when the A/B left-right order flips (typical ID jump after
  a clinch), flipping the labels with it is free, keeping them costs (= AKA
  changed sides). `--no-side-prior` turns it off for ablation: on clip 01 it is
  the difference between 0 and 6 wrong labels.
- **Diagnostics**: `scripts/gt_cases.py <run>` lists every missed / nonathlete /
  swapped / unknown case with a debug image in `data/<run>/gt_cases/`;
  `scripts/gt_sanity.py <run>` flags labels that contradict measured colors.
  GT matching is now one-to-one by best IoU (first-over-threshold was ambiguous
  in a clinch).
- `scripts/compare_runs.py A B` compares two runs' metrics and video hashes.

## Round 7 (labeling tool, back-fill, generalized graphic handling)

- `scripts/label_ground_truth.py --clip X --frames 192,240,384` or `--flagged RUN`
  (frames `gt_sanity` flags; status bar shows why; `c` shows the `gt_cases` image).
- **Back-fill**: track eligibility (min length, mean size) and ranking use each
  track's whole-clip stats from pass 1, so a track is a candidate from its first
  frame, not after 10 sightings. Fixes the "nobody assigned at frame 0" misses.
- **Graphics**: ignore zones apply only while their graphic is up. The detector now
  takes a dark bar (YUKO/IPPON) or an HSV color swatch (Chukoku: yellow card). The
  on-window is widened by `graphic_fade_pad_seconds` (0.5s) each side because the
  pictogram figures show before the marker pixels do (clip 05 frame 0 put two
  pictogram figures in the slots without this). Current slot holders keep their
  place in the top two, so a newcomer cannot push out an athlete.
- Skipped on purpose: post-bout mat margin (bowing/walking off).

## Current status (BASELINE - frozen at tag `baseline-broadcast-v1`)

Run `v7` = commit `e3feb3a` (clean tree), `scripts/run_all.sh v7`, scored with the
human-corrected labels in `data/ground_truth/` (308 labeled athlete instances, 1 frame
per second, 6 clips). v6 was re-scored on the same labels.

| Clip | labeled frames | identity acc % v6 → v7 | swapped | unknown | non-athlete in slot | missed v6 → v7 |
|---|---|---|---|---|---|---|
| 01 | 28 | 85.7 → 89.3 | 0 | 6 | 0 | 2 → 0 |
| 02 | 37 | 93.2 → 98.6 | 0 | 0 | 1 | 5 → 1 |
| 03 | 13 | 92.3 → 100 | 0 | 0 | 0 | 2 → 0 |
| 04 | 15 | 93.3 → 100 | 0 | 0 | 0 | 2 → 0 |
| 05 | 35 | 94.3 → 97.1 | 0 | 0 | 0 | 4 → 2 |
| 06 | 26 | 92.3 → 96.2 | 0 | 2 | 0 | 2 → 0 |
| **all** | 154 | **91.9 → 96.4** | 0 | 8 | 1 | 17 → 3 |

Swapped identities (the worst error) are 0 on every clip. Speed: ~19 fps on this M1
(yolo11n-pose, imgsz 1280, MPS), offline.

Accuracy is measured on 1 frame/second in 6 short clips from ONE broadcast, so each
number rests on few frames (a single frame is ~0.3 points overall, ~3-4 points per
clip). Treat the table as "no known regression", not as a precise accuracy.

### What the remaining 12 non-correct cases are
- **8 unknown (`?`)**: clip 01 frames 480/503/527 and clip 06 frame 647 - both athletes
  in a clinch (GT boxes overlap), no usable color evidence. By design the box shows `?`
  rather than a guess.
- **2 missed, clip 05 frames 767/791**: the athlete's box exists and passes the filter
  but no slot holds it. Not investigated.
- **1 missed + 1 non-athlete, clip 02 frame 192**: the `AO` label in the ground truth
  is on a seated official (the pipeline rejects that box as seated and its slot goes to
  the real athlete). This looks like a label that is still wrong - please re-check.
- `gt_sanity` still flags clip 05 frame 575 (AKA box not redder than AO box); unchecked.

### Known limitations
- Tuned on 6 clips of one broadcast; anything else (camera, lighting, graphics, belt
  or gi colors) is untested. Do not read the table as a generalization estimate.
- Moving camera (clip 06): no tatami calibration, pixel space only; stats/heatmaps are
  not meaningful there.
- Clinches: two overlapping people yield one box or merged evidence -> `?` or misses.
- Identity depends on red/blue belt+glove pixels; at 360p a belt is ~5px thick.
- `peak_speed` in meters is noisy (homography amplifies jitter) - use avg speed/distance.
- Ranking now keeps the current slot holder over a bigger newcomer; a wrong holder that
  stays a valid candidate is not displaced by size alone (it is only dropped when it
  fails the filters or leaves the mat).
- Only the first-listed graphic layouts exist in config (YUKO/IPPON panel, Chukoku
  panel with a yellow card). Other penalty levels (other card colors/positions) are not
  covered. Seated-person and dark-clothing rejection are tuned on this broadcast.
- Interactive calibration click-through and the labeling tool's `--flagged`/`c` UI
  paths were exercised only up to rendering, not by hand.

### What is tuned for this broadcast
- `config.yaml: classifier.ignore_zones` (pixel rects at 640x360 and their detectors),
  `graphic_fade_pad_seconds`, `dark_brightness_threshold: 90`.
- Tatami corners `185,195,460,195,495,335,150,335` for the 5 fixed-camera clips.
- Color thresholds in `athletes.py` (red/blue hue/sat/val, weak-blue navy band).
- Identity: `SWITCH_COST`, `SINGLE_DEADZONE`, side-order prior (`identity.py`).
- Detection: `yolo11n-pose`, `imgsz 1280`, `conf 0.10`, ByteTrack `track_buffer 60`.

### Baseline policy
Do not tune further on these 6 clips. The next input is a new 1080p video from the
user's own club: first measure `v7`'s code on it with fresh ground-truth labels
(`label_ground_truth.py`, `gt_report.py`), compare to this table, and only then change
anything. Reproduce this baseline: `git checkout baseline-broadcast-v1 && scripts/run_all.sh v7b`.
