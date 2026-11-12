"""Read-only JSON + video API for the React dashboard (web/).

    .venv/bin/python -m api.server            # Windows: .venv\\Scripts\\python -m api.server
    (port 8765, or API_PORT=...; the React dev server proxies /api here)

Serves what the pipeline already produced. It never writes and never regenerates anything:
  data/<run>/profiles.json, data/<run>/features/<clip>.json, data/<run>/reports/<id>.json,
  data/roster.csv, videos/output/<run>/<clip>_web.mp4 (H.264, with HTTP Range so seeking works).

Schemas: HANDOFF.md section 4. Field names are passed through unchanged.
If web/dist exists (npm run build), it is served at / so one process runs the whole demo.
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS = ROOT / "videos" / "output"
WEB_DIST = ROOT / "web" / "dist"
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")  # run / clip / athlete ids; blocks path traversal

AGE_GROUPS = {"senior_f": "Senior Female", "senior_m": "Senior Male"}

app = FastAPI(title="Kumite Analysis API", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5199", "http://127.0.0.1:5199"],
                   allow_methods=["GET"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers
def _safe(name: str) -> str:
    if not NAME.match(name) or name.startswith("."):
        raise HTTPException(400, f"invalid name: {name!r}")
    return name


@lru_cache(maxsize=64)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path):
    if not path.is_file():
        raise HTTPException(404, f"not found: {path.relative_to(ROOT).as_posix()}")
    return _read_json(str(path), path.stat().st_mtime)  # mtime in the key: edits show up without restart


def run_dir(run: str) -> Path:
    d = DATA / _safe(run)
    if not (d / "profiles.json").is_file():
        raise HTTPException(404, f"unknown run: {run}")
    return d


def roster() -> list[dict]:
    p = DATA / "roster.csv"
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def _short(label: str) -> str:
    return re.sub(r"\s*\((AKA|AO)\)\s*$", "", label or "").strip() or label


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/runs")
def runs():
    return sorted((p.parent.name for p in DATA.glob("*/profiles.json")), reverse=True)


@app.get("/api/runs/{run}/profiles")
def profiles(run: str):
    return load_json(run_dir(run) / "profiles.json")


@app.get("/api/runs/{run}/clips")
def clips(run: str):
    """Clips of this run with both athletes' ids and whether the web video exists."""
    d = run_dir(run)
    out: dict[str, dict] = {}
    for p in load_json(d / "profiles.json"):
        for c in p.get("clips", []):
            e = out.setdefault(c["clip"], {"clip": c["clip"], "duration_s": c.get("duration_s"), "athletes": {}})
            e["athletes"][c["slot"]] = p["id"]
    for e in out.values():
        e["has_features"] = (d / "features" / f"{e['clip']}.json").is_file()
        e["has_video"] = (VIDEOS / run / f"{e['clip']}_web.mp4").is_file()
    return sorted(out.values(), key=lambda e: e["clip"])


@app.get("/api/runs/{run}/features/{clip}")
def features(run: str, clip: str):
    return load_json(run_dir(run) / "features" / f"{_safe(clip)}.json")


@app.get("/api/runs/{run}/reports")
def reports(run: str):
    """All coach reports of the run, keyed by athlete id (athletes without a report are absent)."""
    d = run_dir(run) / "reports"
    return {p.stem: load_json(p) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}


@app.get("/api/runs/{run}/reports/{athlete_id}")
def report(run: str, athlete_id: str):
    return load_json(run_dir(run) / "reports" / f"{_safe(athlete_id)}.json")


@app.get("/api/runs/{run}/bouts")
def bouts(run: str):
    """Bouts of this run from data/roster.csv (bout_id), with the final score per side.

    The roster score is cumulative per clip, so the final score is the max over the bout's clips.
    Only clips that are in this run's profiles are included.
    """
    run_clips = {c["clip"] for c in clips(run)}
    by_bout: dict[str, dict] = {}
    for r in roster():
        if r.get("clip") not in run_clips:
            continue
        b = by_bout.setdefault(r["bout_id"], {"bout_id": r["bout_id"], "age_group": r.get("age_group"),
                                              "clips": [], "sides": {}})
        if r["clip"] not in b["clips"]:
            b["clips"].append(r["clip"])
        s = b["sides"].setdefault(r["slot"], {"athlete_id": r["athlete_id"], "label": r["label"],
                                             "name": _short(r["label"]), "belt": r.get("belt"), "points": 0})
        s["points"] = max(s["points"], _num(r.get("points_for")) or 0)
    for b in by_bout.values():
        b["clips"].sort()
        aka, ao = b["sides"].get("aka", {}), b["sides"].get("ao", {})
        group = AGE_GROUPS.get(b["age_group"] or "", (b["age_group"] or "").replace("_", " ").title())
        b["title"] = f"{aka.get('name', 'AKA')} vs {ao.get('name', 'AO')}"
        b["subtitle"] = f"{group} Kumite".strip()
    return list(by_bout.values())


@app.get("/api/runs/{run}/videos/{clip}")
def video(run: str, clip: str):
    """Annotated H.264 video. Starlette's FileResponse answers Range requests (206), so <video> can seek."""
    p = VIDEOS / _safe(run) / f"{_safe(clip)}_web.mp4"
    if not p.is_file():
        raise HTTPException(404, "web video not found (run scripts/run_match.py to create *_web.mp4)")
    return FileResponse(p, media_type="video/mp4")


# Production: serve the built React app from the same origin (after the /api routes).
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, f"no such API route: /{path}")  # don't answer API typos with index.html
        f = (WEB_DIST / path).resolve()
        if path and f.is_file() and WEB_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(WEB_DIST / "index.html")  # client-side routes


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=int(os.environ.get("API_PORT", 8765)), reload=True)
