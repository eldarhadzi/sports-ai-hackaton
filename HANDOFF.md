# HANDOFF: Karate Kumite Match Analysis (hackathon state, 2026-11-12)

This document covers what exists, what has been checked, what doesn't work yet, and what to
do next, with exact files and commands. Read sections 1-3, then pick a task from section 9.

**Status in one paragraph:** the whole flow works end to end on one real match. A raw video goes
through pose tracking and AKA/AO identity, then karate metrics (strike rate, initiative, range
control, tempo), then athlete profiles in the team's JSON format, a coach report, and a dashboard.
One match is fully processed (FRA vs CRO, 3 clips). The main open items are: validate the strike
detector against a human count, add more matches/athletes (the cohort is only 2), agree on what
`standing_pct` means, and polish the UI for the demo.

---

## 1. What the product does

Input: a recorded kumite (karate sparring) video.
Output, per athlete:

```json
{
  "id": "F-07", "age_group": "U14", "belt": "green",
  "n_bouts": 2, "reliability": 0.78,
  "metrics": {"strike_rate": 14.2, "initiative_share": 0.61, "range_control": 0.55, "tempo": 3.1},
  "skill_pct": 82, "standing_pct": 35, "gap": 47,
  "confidence": "medium", "cohort_n": 7,
  "timeline": [{"t": 0.0, "initiative": 0.4, "strike": false}]
}
```

These numbers are shown in a dashboard. An LLM turns them into a coach report: what's good, what's
bad, and where to focus.

---

## 2. How it works (the flow)

```
videos/input/<clip>.mp4
   │  src/main.py            YOLO11-pose + ByteTrack (Ultralytics) + OpenCV, plus this repo's athlete logic
   ▼
data/<run>/<clip>.jsonl      one row per person per frame: 17 keypoints, box, AKA/AO label, graphic_on…
videos/output/<run>/<clip>_analyzed.mp4   boxes + skeletons drawn on the video
   │  src/analysis/features.py   karate metrics from keypoints (no extra model)
   ▼
data/<run>/features/<clip>.json   per clip: both athletes' metrics, strike events, exchanges, timeline
   │  scripts/build_profiles.py  + data/roster.csv (who is who, age group, belt, score)
   ▼
data/<run>/profiles.json     per athlete: the team JSON format + extras, cohort percentiles
   │  scripts/coach_report.py  (Claude API; reads numbers only, never video)
   ▼
data/<run>/reports/<id>.json   headline, summary, strengths, weaknesses, focus, drills, caveats
   │  dashboard/app.py  (Streamlit + Plotly)
   ▼
http://localhost:8501        Athlete card · Bout view (video + timeline) · Scouting board
```

`scripts/run_match.py` runs the first steps (pipeline, features, web video, profiles) for a list of clips in one command.

### How the video analysis works, step by step (src/main.py)
1. **Detect + pose:** `yolo11n-pose` finds every person in each frame and gives 17 body keypoints
   (COCO: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), at `imgsz 1280`, `conf 0.10`.
2. **Track:** ByteTrack (`bytetrack_tuned.yaml`, `track_buffer 60`) keeps an ID per person across frames.
3. **Pick the 2 athletes:** `src/athletes.py` rejects seated people, dark suits and dark trousers
   (referee/judges), broadcast graphics (`config.yaml: classifier.ignore_zones`), and anyone outside
   the mat (tatami calibration).
4. **AKA/AO identity:** red vs blue pixels on belt and gloves, decided over the whole clip with a
   Viterbi pass (`src/identity.py`). AKA = red, AO = blue. `?` when there's no evidence.
5. **Export:** JSONL + annotated video + run_meta. With `--det-cache`, steps 1-2 are cached, so a
   re-run takes seconds instead of minutes.

### How the metrics work (src/analysis/features.py)
All distances are divided by the athlete's **torso length** (shoulder-to-hip), so they work on any
zoom and in pixel space. Each threshold is in `DEFAULTS` at the top of the file.

| Concept | Definition |
|---|---|
| **live time** | both athletes tracked, **no** score graphic on screen, and **not idle** (idle = both nearly still and >4 torso apart for ~1 s: walk-back, waiting for "hajime") |
| **strike** | wrist (punch) or ankle (kick) speed peak relative to own hips ≥ 5 torso/s, aimed at the opponent, with the arm going from bent to extended (or the foot rising) within 0.3 s, at 1.2-6 torso distance (not in a clinch); max 1 per 0.35 s. **Kick** if a knee or ankle is raised at that moment |
| **advance** | moving toward the opponent > 1.5 torso/s for ≥ 0.2 s |
| **exchange** | strikes from either athlete within 1 s of each other; **initiator** = whoever advanced or struck first (looking 0.8 s back) |
| **strike_rate** | strikes per minute of live time |
| **initiative_share** | exchanges started / exchanges with a known initiator |
| **range_control** | share of moving frames where this athlete presses forward more than the opponent |
| **tempo** | (strikes + advances) per 10 s of live time |
| **timeline** | every 0.5 s: initiative = 0.5 + 0.5·tanh(Δ forward speed / 1.5); `null` while paused |
| **reliability** | tracked share of live time × identity-known share × keypoint quality |

### Profiles (scripts/build_profiles.py)
- Metrics are weighted by live time across the athlete's clips; initiative is pooled over exchanges.
- **Cohort** = same `age_group`. `metric_pct` = percentile per metric, `skill_pct` = percentile of the
  mean metric percentile.
- `standing_pct` = percentile of **bout points share** from the scoreboard. **This is a placeholder:
  the team hasn't confirmed what "standing" means.** `gap = skill_pct - standing_pct`.
- Confidence: **high** if reliability ≥ 0.85 and ≥ 3 bouts and cohort ≥ 5; **medium** if reliability ≥ 0.7; otherwise **low**.
- Clips of the same bout share a `bout_id` in the roster, so they count as one bout. The score is
  cumulative, so the max per bout is used.

---

## 3. Setup (Windows, CPU only; also works on Mac/Linux)

```bash
git clone git@github.com:eldarhadzi/sports-ai-hackaton.git && cd sports-ai-hackaton
python -m venv .venv                       # Python 3.11-3.13 tested (3.13 on Windows)
.venv\Scripts\pip install -r requirements.txt      # Mac/Linux: .venv/bin/pip
.venv\Scripts\python -m pytest tests -q            # expect 40 passed
```

**Data is not in git, on purpose.** `videos/`, `data/`, `models/` and `*.pt` are git-ignored
(README "Data & privacy": clips may contain minors; never commit or upload them). Get
`handoff_v1.zip` from Vedad and unzip it into the repo root. It contains:

```
data/roster.csv
data/v1/profiles.json
data/v1/features/kumite_0{1,2,3}_siroki_kadar.json
data/v1/reports/FRA-1.json, CRO-1.json
videos/output/v1/kumite_0{1,2,3}_siroki_kadar_web.mp4
```

To re-run the analysis itself, you also need the raw clips in `videos/input/` (ask Vedad).
The pose model downloads itself on first run (`yolo11n-pose.pt`, ~6 MB).

### Commands
```bash
# dashboard
.venv\Scripts\streamlit run dashboard/app.py                # http://localhost:8501

# full analysis of clips (CPU: ~1.7 frames/s, so a 30 s clip takes ~7 min; re-runs use the cache)
.venv\Scripts\python scripts/run_match.py --run v1 --clips kumite_01_siroki_kadar kumite_02_siroki_kadar kumite_03_siroki_kadar --corners "185,195,460,195,495,335,150,335"

# only recompute metrics/profiles after changing features.py or the roster (seconds)
.venv\Scripts\python scripts/run_match.py --run v1 --clips kumite_01_siroki_kadar kumite_02_siroki_kadar kumite_03_siroki_kadar --skip-pipeline

# coach reports through the Claude API (needs ANTHROPIC_API_KEY; Claude Pro chat does NOT include API access)
.venv\Scripts\python scripts/coach_report.py --run v1
```

`--corners` are the tatami corners **for this broadcast camera only** (top-left, top-right,
bottom-right, bottom-left, in pixels). New footage needs its own corners. Open a frame and read
the pixel positions, or use `--calibrate` to click them. A moving camera gets no corners at all.
For non-broadcast footage (club videos), add `--config config_club.yaml` (no TV-graphic zones).

---

## 4. Data formats

### `data/<run>/profiles.json`: a list of athletes
```jsonc
{
  "id": "FRA-1", "label": "FRA (AKA)", "age_group": "senior_f", "belt": "black",
  "n_bouts": 1, "reliability": 0.93,
  "metrics": {"strike_rate": 12.92, "initiative_share": 0.4, "range_control": 0.5, "tempo": 3.59},
  "skill_pct": 25, "standing_pct": 75, "gap": -50, "confidence": "medium", "cohort_n": 2,
  "timeline": [{"t": 0.0, "initiative": 0.61, "strike": true, "clip": "kumite_01_siroki_kadar"}],
  // extras beyond the team format
  "metric_pct": {"strike_rate": 25, "initiative_share": 25, "range_control": 50, "tempo": 25},
  "standing_basis": "bout points share (placeholder)",
  "n_clips": 3, "active_s": 55.8, "points_for": 4, "points_against": 0,
  "counts": {"strikes": 12, "punches": 11, "kicks": 1, "advances": 8, "exchanges_initiated": 8},
  "events": [{"t": 0.21, "clip_t": 0.21, "clip": "kumite_01_siroki_kadar", "type": "punch", "limb": "left_hand", "speed": 6.5, "frame": 5}],
  "clips": [{"clip": "kumite_01_siroki_kadar", "slot": "aka", "offset": 0.0, "duration_s": 27.74,
             "active_s": 27.03, "metrics": {}, "counts": {}, "reliability": 0.88}]
}
```
- `timeline` / `events` put the clips **back to back** (`t` = clip offset + time in clip). Use
  `clip_t` (seconds into that clip) to seek its video.
- `initiative` is `null` while the bout is paused, so draw it as a gap.

### `data/<run>/features/<clip>.json`: one clip, both athletes
`T, fps, duration_s, active_s, paused_s, idle_s, clip,`
`exchanges: [{t0, t1, n_strikes, initiator: "aka"|"ao"|null}],`
`athletes: {aka|ao: {metrics, counts, events, timeline, reliability, coverage, identity_known, keypoint_quality}},`
`config` (the thresholds used).

### `data/<run>/reports/<id>.json`
`athlete_id, model, headline, summary, strengths[], weaknesses[], focus[], drills[], caveats[]`

### `data/roster.csv`
`clip,slot,athlete_id,label,age_group,belt,bout_id,points_for,points_against`
(one row per clip × slot; `slot` = aka|ao; points = scoreboard at the **end** of that clip).

### `data/<run>/<clip>.jsonl`: raw, per person per frame
`frame, timestamp, athlete_id (aka|ao|null), tracker_id, bbox, confidence, keypoints[17×(x,y,conf)],
ground_point, tatami_point, belt_color, graphic_on, color_evidence, identity_known,
passed_athlete_filter, reject_reason, on_mat, leg_brightness, glove, official`

---

## 5. Current results: FRA vs CRO (senior female, broadcast, 640×360, 24 fps)

| Clip | Length | Tracking (both athletes) | Identity known | Score during clip |
|---|---|---|---|---|
| kumite_01 | 27.7 s | 97.4 % | 89.5 % (clinch at the end → `?`) | 0-0 |
| kumite_02 | 36.2 s | 99.1 % | 100 % | FRA yuko → 1-0 |
| kumite_03 | 13.0 s | 99.7 % | 100 % | FRA ippon just before the clip → 4-0 |

| | FRA (AKA, red) | CRO (AO, blue) |
|---|---|---|
| strike_rate | 12.9 /min | 33.4 /min |
| initiative_share | 40 % | 60 % |
| range_control | 50 % (69 % in clip 03) | 50 % |
| tempo | 3.6 /10 s | 7.7 /10 s |
| reliability / confidence | 0.93 / medium | 0.93 / medium |

**Demo story:** CRO does most of the fighting (2.6× the strikes, starts 60 % of exchanges) but
doesn't score. FRA is selective and efficient. Clip 02: at 18.8 s FRA initiates an exchange
(both strike at the same moment), 19.8-22.9 s walk-back, 22.9-28.3 s idle, **27.7 s the
referee awards FRA the YUKO**. The system tied the point to an exchange FRA started.

### What was checked by eye (frames compared with detections)
- ✅ Score-graphic pauses are excluded. At clip 03, 6.9-7.7 s, the IPPON pictogram animation was
  being counted as strikes before the fix.
- ✅ Walk-back / idle periods are excluded and correctly have 0 strikes (clip 03, 0-5.8 s).
- ✅ The clip 03 exchange at 9.5-10.3 s is real: CRO initiates (high kick) and FRA counters.
- ✅ The clip 02 exchange at 18.8 s (FRA lunge) is the one that produced her yuko.
- ⚠️ Clip 01, 9.6-13.2 s: 9 CRO "strikes" in close-range hand-fighting. Some of them are probably
  guard or grip movements, not punches. **No human count exists yet.**
- ⚠️ Punch vs kick: CRO's high kick at clip 03, 9.75 s, was lost (the pose model dropped her for
  that frame) and counted as punches around it.

The two current coach reports in `data/v1/reports/` were written by Claude (in Claude Code)
**from profiles.json only**, under the same rules as `scripts/coach_report.py`. The API script
itself hasn't been run yet (no API key on the dev laptop).

---

## 6. What is DONE (file by file)

### Before this hackathon session (rounds 1-8, see README for details and measured history)
| File | What |
|---|---|
| `src/pose_tracker.py` | YOLO pose + ByteTrack wrapper, device auto (cuda→mps→cpu), detection cache |
| `src/athletes.py` | athlete filter (seated, dark clothing, legs/gloves official check, graphic zones), slot selection, color evidence |
| `src/identity.py` | whole-clip AKA/AO decision (pair signal + Viterbi + side-order prior) |
| `src/calibration.py`, `src/mat_region.py`, `scripts/mark_mat.py` | tatami homography (meters), optional active-mat region that follows a moving camera |
| `src/visualization.py`, `src/export.py` | annotated video, trails, heatmaps, JSONL, run_meta |
| `src/analysis/karate.py` | movement stats (distance, speed, direction changes) |
| `src/analysis/ground_truth.py`, `scripts/label_ground_truth.py`, `gt_*.py` | GT labeling tool + identity accuracy scoring (**96.4 % identity accuracy, 0 swaps** on 154 labeled frames of the 6 broadcast clips, round 7) |
| `src/slots.py` | **v8 WIP: written and unit-tested, NOT wired into main.py yet** (offline global slot assignment) |

### This session
| File | What |
|---|---|
| `src/analysis/features.py` | **new**: karate metrics, strikes, exchanges, idle/pause detection, timeline, reliability |
| `src/main.py` | + `graphic_on` per JSONL row (score/penalty graphic on screen = stoppage) |
| `scripts/build_profiles.py` | **new**: roster + features → `profiles.json` (team format, cohort percentiles, gap, confidence) |
| `scripts/coach_report.py` | **new**: Claude API coach report (`claude-opus-5-5`, structured output, numbers only) |
| `scripts/run_match.py` | **new**: one cross-platform command: pipeline → features → web video → profiles |
| `dashboard/app.py`, `.streamlit/config.toml` | **new**: Streamlit dashboard (Athlete card, Bout view, Scouting board), light theme, usage stats off |
| `tests/test_features.py` | **new**: synthetic bout tests (punch, exchange initiator, range control, pause exclusion) |
| `requirements.txt` | + streamlit, plotly, imageio-ffmpeg, anthropic, pydantic |

---

## 7. Known limitations (keep them visible in the UI and the pitch)
1. **Strike counts are not validated** against a human count. Treat them as estimates.
2. **Punch vs kick is unreliable at 360p.** The pose model often loses an athlete during a fast kick.
3. **Cohort = 2 athletes** (one bout), so percentiles and the gap are illustrative only.
4. **`standing_pct` is a placeholder** (bout points share).
5. **CPU speed:** ~1.7 frames/s on a Ryzen 7 4700U (no NVIDIA GPU). A CUDA GPU would be ~10× faster.
6. **Tuned for one broadcast.** Graphic zones, tatami corners and dark-clothing thresholds come
   from this camera. New footage: use `config_club.yaml` and new corners, and check the output
   video before trusting the numbers.
7. **Clinch:** overlapping athletes give `?` identity or a missed athlete. Strikes inside 1.2 torso are ignored.
8. **Idle detection** is a heuristic (both still and far apart). A slow tactical stand-off at long
   range could be counted as idle.
9. **AGPL:** Ultralytics YOLO is AGPL-3.0. Fine for a prototype, but it matters before any commercial use.

### Gotchas
- OpenCV writes `*_analyzed.mp4` as **mp4v, which browsers can't play**. The dashboard and
  `run_match.py` make `*_web.mp4` (H.264) with the bundled ffmpeg (`imageio-ffmpeg`).
- `scripts/run_all.sh`, `run_club.sh`, `eval_v8.sh` are **zsh/Mac only**. Use `run_match.py` on Windows.
- `--corners` is only used when `data/<run>/calibration/<clip>.json` doesn't exist yet.
- The detection cache key includes model/conf/imgsz/tracker config. Change any of them and the slow pass runs again.
- `features.py` undoes the pipeline's 0.5 s graphic "fade pad" (`PAUSE_UNPAD_S`). Keep it equal to
  `config.yaml: classifier.graphic_fade_pad_seconds`.

---

## 8. What SHOULD be done next (prioritized)

### P0: before the demo
| # | Task | How | Done when |
|---|---|---|---|
| 1 | **UI polish** for the demo | see the prompt in section 10 | 3 polished views + downloadable report, AppTest passes |
| 2 | **Validate strikes** | Watch clip 01 at 9-14 s and clip 02 (whole) at 0.25× speed. Write down `time, athlete (red/blue), punch/kick` in a CSV. Compare with `events` in `data/v1/features/*.json` and tune `DEFAULTS` in `features.py` (main knobs: `punch_speed`, `punch_delta`, `min_range`, `refractory_s`). Re-run with `run_match.py --skip-pipeline`. | precision/recall per athlete written down, e.g. "detector matches 8 of 10 human-counted punches, 2 extra" |
| 3 | **Decide `standing_pct`** | Ask whoever wrote the example JSON. If it isn't "points share", change `build_profiles.py` (search `standing_pct`) | definition agreed and documented in this file |
| 4 | **Demo script + pitch** | Flow: raw video → annotated video (Bout view) → athlete card → coach report → limitations slide | 3-minute run-through rehearsed |

### P1: makes the numbers meaningful
| # | Task | How |
|---|---|---|
| 5 | **More matches / athletes** (cohort ≥ 5, same age group) | Fixed camera, whole mat, red/blue belts visible, 720p+. Put clips in `videos/input/`, add roster rows, run `run_match.py`. New camera = new `--corners`. Club footage: `--config config_club.yaml`. Check the annotated video first |
| 6 | **Coach reports from the API** | Get an API key at console.anthropic.com (separate from Claude Pro). `set ANTHROPIC_API_KEY=...` then `python scripts/coach_report.py --run v1`. Reports are cached files, so the demo doesn't need the network |
| 7 | **Better kick detection** | Kicks are lost when the pose disappears for 1-2 frames. Ideas: interpolate keypoints over 1-2 frame gaps before computing speeds; treat "athlete disappeared next to the opponent with a raised knee just before" as a kick candidate |
| 8 | **Higher resolution source** | The same match in 720p/1080p would improve wrist/ankle keypoints a lot. No code change needed |

### P2: after the hackathon
- A GPU machine and a bigger pose model (`yolo11s-pose` / `yolo11m-pose`: change `config.yaml: model.name`).
- Wire `src/slots.py` into `main.py` (v8 WIP, see the round 8 commits) and re-score with `scripts/gt_report.py`.
- Technique classification (gyaku-zuki, kizami-zuki, mawashi-geri…) from labeled strike clips.
- Read the score from the scoreboard (OCR) instead of typing it in the roster.
- License: test a permissively-licensed pose model (RTMPose/RTMO via MMPose, Apache-2.0) behind `PoseTracker`.
- Klub.ba integration (out of scope for V1 by design).

---

## 9. Who does what (suggested split for 3 people)
| Person | Track | Section |
|---|---|---|
| UI teammate | dashboard + downloadable report | 8 #1, prompt in 10 |
| Metrics teammate | human strike count + tuning, kick detection | 8 #2, #7 |
| Vedad | `standing_pct` decision, more footage, API key, pitch | 8 #3, #4, #5, #6 |

---

## 10. Prompt for the UI work (paste into your AI coding assistant)

> You are working in the repo `sports-ai-hackaton` (Python). Read `HANDOFF.md` (sections 2, 4,
> 5, 7) and `dashboard/app.py` first. The video analysis is DONE. Do not modify `src/`,
> `scripts/` or anything under `data/`, and do not regenerate data. Your job is the presentation
> layer for a hackathon demo, in about 2 hours.
>
> Inputs (read-only): `data/v1/profiles.json`, `data/v1/features/*.json`,
> `data/v1/reports/*.json`, `videos/output/v1/*_web.mp4`. Schemas are in HANDOFF.md section 4.
>
> Turn the existing Streamlit app `dashboard/app.py` (keep Streamlit + Plotly) into a polished demo:
>
> 1. **Match overview** (new first page): title "FRA vs CRO, Senior Female Kumite", final score
>    4-0, the 4 metrics side by side for both athletes (FRA red `#D9483B`, CRO blue `#2F6FD6`,
>    every bar labeled with its value), and each athlete's report `headline`.
> 2. **Athlete card**: header (label, belt, age group, bouts, reliability, confidence badge),
>    4 metric tiles with a definition tooltip (HANDOFF.md section 2) and cohort percentile, the
>    initiative timeline with strike markers (gaps where `initiative` is null, thin vertical lines
>    at clip boundaries), and the coach report (headline, summary, strengths, weaknesses, focus;
>    drills in an expander; caveats in small text).
> 3. **Bout view**: the clip's video next to a time-aligned chart. Exchanges are grey bands
>    labeled with the initiator, strikes for both athletes are markers, and a "jump to" list of
>    events seeks the video with `st.video(path, start_time=int(clip_t - 1))`.
> 4. A **"Download report"** button per athlete that exports a clean single-page HTML (header,
>    metrics table, timeline image or table, coach report, limitations) for a coach to keep.
>
> Rules:
> - Use the AKA/AO colors above everywhere. Light theme, no dual-axis charts, and a legend
>   whenever a chart shows 2 series.
> - Always show the limitations from HANDOFF.md section 7 (items 1-4) as a small caption or info
>   box. Never present strike counts as exact.
> - Everything stays local: no uploads, no external APIs, no analytics.
> - Don't rename JSON fields, since other scripts produce them.
> - Test with `streamlit.testing.v1.AppTest` (every page, no exceptions), then click through in the browser.
> - Work on your own branch and commit with clear messages.
