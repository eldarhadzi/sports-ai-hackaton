# Kumite dashboard (React)

React + TypeScript (Vite) frontend over a small read-only FastAPI service (`api/server.py`).
It shows what the pipeline already produced. It never writes to `data/` or regenerates anything.

| View | Route | Data |
|---|---|---|
| Match overview | `/r/<run>` | profiles, bouts (from `data/roster.csv`), reports, clips |
| Athlete card | `/r/<run>/athlete/<id>` | profile + report; "Download report" exports a single-page HTML |
| Bout view | `/r/<run>/bout/<clip>` | `features/<clip>.json` + annotated video; chart and video are time-aligned (click to seek, playhead follows the video) |
| Scouting board | `/r/<run>/scouting` | profiles (skill vs standing) |

The known limitations (HANDOFF.md section 7, items 1-4) are in the downloadable HTML report. By team decision
they are not shown on the dashboard pages; strike counts are still marked as estimates (`~`).

## Run it (development)

Two terminals, from the repo root:

```bash
.venv/bin/python -m api.server          # API on http://127.0.0.1:8765   (Windows: .venv\Scripts\python -m api.server)
cd web && npm install && npm run dev    # app on http://127.0.0.1:5199, proxies /api to :8765
```

The ports are deliberately uncommon. Change them with `API_PORT=...` (both commands) and `WEB_PORT=...`.
API docs: http://127.0.0.1:8765/api/docs

## Run it (demo, one process)

```bash
cd web && npm run build && cd ..
.venv/bin/python -m api.server          # serves the built app and the API on http://127.0.0.1:8765
```

## API

All routes are `GET` and read-only. JSON field names are the pipeline's own (HANDOFF.md section 4).

| Route | Returns |
|---|---|
| `/api/runs` | run names that have `profiles.json`, newest first |
| `/api/runs/{run}/profiles` | `data/<run>/profiles.json` |
| `/api/runs/{run}/clips` | clips with both athlete ids, `has_features`, `has_video` |
| `/api/runs/{run}/bouts` | bouts from `roster.csv` (`bout_id`), final score = max over the bout's clips, title |
| `/api/runs/{run}/features/{clip}` | `data/<run>/features/<clip>.json` |
| `/api/runs/{run}/reports` / `/reports/{id}` | coach reports |
| `/api/runs/{run}/videos/{clip}` | `videos/output/<run>/<clip>_web.mp4` with HTTP Range (seeking) |

JSON reads are cached by file mtime, so a re-run of `scripts/run_match.py` shows up on reload without
restarting. Tests: `.venv/bin/python -m pytest tests/test_api.py`.

## Code map

```
web/src/lib/api.ts          typed fetch + TanStack Query hooks (one per route)
web/src/lib/types.ts        schemas from HANDOFF.md section 4
web/src/lib/constants.ts    AKA/AO colors, metric names and definitions, limitations, formatting
web/src/lib/report.ts       downloadable HTML report (inline CSS + SVG, no scripts)
web/src/components/         charts (Recharts), video player, shared UI
web/src/pages/              the four views
```

Colors: AKA `#D9483B`, AO `#2F6FD6` (validated as a pair for color-vision deficiency; punch/kick
also differ by marker shape). Light theme, no dual-axis charts.
