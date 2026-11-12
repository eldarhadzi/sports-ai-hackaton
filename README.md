
# Kumite Analytics

Computer-vision match analysis for karate kumite. It turns an ordinary match video into
per-athlete performance metrics, an interactive dashboard and an AI coach report, all
processed locally.

> The video tracks both athletes, labels them AKA (red) and AO (blue), detects strikes and
> exchanges, and measures who takes the initiative and controls the distance. A coach gets
> numbers they can click back to the exact moment in the video.

---

## Features

- **Athlete tracking.** Pose estimation and multi-object tracking follow both fighters and
  ignore referees, judges, coaches and TV graphics.
- **AKA / AO identity.** Belt and glove color, decided over the whole clip, keep each athlete's
  label stable through clinches and tracker ID changes.
- **Karate metrics.** Strike rate, initiative share, range control and tempo, measured only
  during live fighting (score-graphic pauses and walk-backs are excluded).
- **Video-linked evidence.** Every strike and exchange has a timestamp; click one in the
  dashboard and the video jumps to it.
- **Athlete profiles.** Metrics combined across clips and bouts, with percentiles within the
  age-group cohort and a confidence level.
- **Coach report.** An LLM (Claude) writes strengths, weaknesses, focus areas and drills from the
  numbers only. No video is ever sent.
- **Shareable report.** Each athlete card exports a self-contained HTML report.

## How it works

```
match video ──► pose + tracking ──► karate metrics ──► athlete profiles ──► coach report
               (YOLO11-pose,         (strikes,          (cohort            (Claude API,
                ByteTrack, OpenCV)    exchanges,         percentiles,       numbers only)
                                      initiative)        confidence)
                                                │
                                                ▼
                                   dashboard (React + FastAPI)
```

| Stage | Code | Output |
|---|---|---|
| Pose, tracking, athlete selection, identity | `src/main.py` | `data/<run>/<clip>.jsonl`, annotated video |
| Metrics per clip | `src/analysis/features.py` | `data/<run>/features/<clip>.json` |
| Profiles per athlete | `scripts/build_profiles.py` + `data/roster.csv` | `data/<run>/profiles.json` |
| Coach report | `scripts/coach_report.py` | `data/<run>/reports/<id>.json` |
| Read-only API | `api/server.py` (FastAPI) | JSON and video streaming |
| Dashboard | `web/` (React, TypeScript, Vite) | Match overview, Athlete card, Bout view, Scouting board |

All distances are normalized by the athlete's torso length, so the metrics work at any camera zoom.

| Metric | Meaning |
|---|---|
| `strike_rate` | strikes per minute of live fighting |
| `initiative_share` | share of exchanges this athlete started |
| `range_control` | share of moving time this athlete pressed forward more than the opponent |
| `tempo` | strikes plus forward attacks per 10 s of live fighting |
| `reliability` | tracking coverage × identity certainty × keypoint quality |

Full definitions and thresholds are in [HANDOFF.md, section 2](HANDOFF.md#2-how-it-works-the-flow).

## Quick start

**Requirements:** Python 3.11-3.13 and Node.js 20+. Runs on CPU; no GPU needed.

```bash
git clone git@github.com:eldarhadzi/sports-ai-hackaton.git
cd sports-ai-hackaton
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # macOS/Linux: .venv/bin/pip
.venv\Scripts\python -m pytest tests -q           # 52 tests
```

The repository includes one fully analyzed sample match (FRA vs CRO), so the dashboard works
right after cloning.

### Run the dashboard

Two terminals, from the repository root:

```bash
# terminal 1: API on http://127.0.0.1:8765
.venv\Scripts\python -m api.server

# terminal 2: dashboard on http://127.0.0.1:5199
cd web
npm install
npm run dev
```

For a demo, a single process serves both the built app and the API:

```bash
cd web && npm run build && cd ..
.venv\Scripts\python -m api.server                # open http://127.0.0.1:8765
```

### Analyze a match

1. Put the clips in `videos/input/<clip>.mp4`.
2. Add one row per clip and corner to `data/roster.csv` (athlete id, age group, belt, bout id, score).
3. Run the pipeline, metrics, browser videos and profiles in one command:

```bash
.venv\Scripts\python scripts/run_match.py --run v1 --clips <clip_1> <clip_2> \
    --corners "x1,y1,x2,y2,x3,y3,x4,y4"
```

`--corners` are the four tatami corners in pixels (top-left, top-right, bottom-right,
bottom-left) for a fixed camera; to click them on a frame instead, run `src/main.py --calibrate`
once per camera. Use `--config config_club.yaml` for footage without TV graphics. On CPU the first run processes
about 2 frames per second; detections are cached, so re-runs take seconds. Add
`--skip-pipeline` to recompute only the metrics and profiles.

4. Optionally, generate coach reports (needs an Anthropic API key in `ANTHROPIC_API_KEY`):

```bash
.venv\Scripts\python scripts/coach_report.py --run v1
```

## Sample match: FRA vs CRO

A public senior female broadcast match (640×360, 24 fps), split into 3 clips. Both athletes were
tracked in 97-99 % of frames, with a reliability of 0.93.

| | FRA (AKA) | CRO (AO) |
|---|---|---|
| Strike rate | ~12.9 /min | ~33.4 /min |
| Initiative share | 40 % | 60 % |
| Range control | 50 % | 50 % |
| Tempo | 3.6 /10 s | 7.7 /10 s |
| Score | 4 | 0 |

CRO does most of the attacking but does not score; FRA is selective and efficient. The dashboard
ties FRA's yuko (clip 02, 27.7 s) to the exchange FRA started at 18.8 s.

## Limitations

- Strike counts are estimates from 2D pose and have not yet been checked against a human count.
- At 360p the punch/kick split is unreliable, and fast kicks can be missed when the pose model
  briefly loses an athlete.
- `reached` (the hand or foot got close to the opponent) is a 2D proxy, not contact detection.
- The sample cohort has 2 athletes, so percentiles move in large steps. `standing_pct` is a
  placeholder (bout points share).
- Tuned on one broadcast. New cameras, lighting or graphics need their own calibration and checks.
  A moving camera cannot use tatami calibration.

## Repository layout

```
src/                 video pipeline (pose, tracking, athlete selection, identity, export)
src/analysis/        karate metrics (features.py) and tracking-quality metrics
scripts/             run_match.py, build_profiles.py, coach_report.py, ground-truth tools
api/                 read-only FastAPI service for the dashboard
web/                 React dashboard (see web/README.md)
dashboard/           earlier Streamlit prototype of the dashboard
tests/               unit tests (synthetic data, no model or video needed)
data/, videos/       sample match only; other footage is git-ignored
docs/                development log of the tracking pipeline
config.yaml          pipeline settings (detection, tracking, broadcast graphic zones)
```

## Documentation

- [HANDOFF.md](HANDOFF.md): complete technical handoff (metric definitions, data formats, results,
  limitations and the prioritized to-do list).
- [web/README.md](web/README.md): dashboard architecture, API routes and code map.
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md): measured history of the tracking pipeline
  (rounds 1-8, ground-truth accuracy).

## Data and privacy

Match clips may show minors. `videos/`, `data/` and `models/` are git-ignored, and nothing is
uploaded: all video processing runs locally, and the coach report sends only aggregated numbers to
the LLM. The one exception is the public senior broadcast match FRA vs CRO, committed on purpose
as sample data.

## License note

The pose model, Ultralytics YOLO, is licensed under **AGPL-3.0**. This is fine for a local
prototype but must be reviewed before any commercial use. The model sits behind the `PoseTracker`
interface, so a permissively licensed alternative can be swapped in.
