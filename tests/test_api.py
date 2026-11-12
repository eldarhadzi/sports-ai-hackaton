"""API for the React dashboard (api/server.py): read-only routes over data/<run>/ and videos/output/<run>/."""
import json
import os
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api import server  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    run = tmp_path / "data" / "r1"
    (run / "features").mkdir(parents=True)
    (run / "reports").mkdir()
    clips = [{"clip": c, "slot": s, "offset": 0, "duration_s": 10.0} for c in ("k_01", "k_02") for s in ("aka",)]
    profiles = [
        {"id": "A-1", "label": "AAA (AKA)", "age_group": "senior_f", "clips": clips},
        {"id": "B-1", "label": "BBB (AO)", "age_group": "senior_f",
         "clips": [dict(c, slot="ao") for c in clips]},
    ]
    (run / "profiles.json").write_text(json.dumps(profiles))
    (run / "features" / "k_01.json").write_text(json.dumps({"clip": "k_01", "duration_s": 10.0}))
    (run / "reports" / "A-1.json").write_text(json.dumps({"athlete_id": "A-1", "headline": "h"}))
    (tmp_path / "data" / "roster.csv").write_text(
        "clip,slot,athlete_id,label,age_group,belt,bout_id,points_for,points_against\n"
        "k_01,aka,A-1,AAA (AKA),senior_f,black,b1,1,0\n"
        "k_01,ao,B-1,BBB (AO),senior_f,black,b1,0,1\n"
        "k_02,aka,A-1,AAA (AKA),senior_f,black,b1,3,2\n"
        "k_02,ao,B-1,BBB (AO),senior_f,black,b1,2,3\n"
        "other,aka,X,X (AKA),senior_f,black,b2,9,0\n")
    vids = tmp_path / "videos" / "output" / "r1"
    vids.mkdir(parents=True)
    (vids / "k_01_web.mp4").write_bytes(bytes(range(256)) * 4)
    monkeypatch.setattr(server, "DATA", tmp_path / "data")
    monkeypatch.setattr(server, "VIDEOS", tmp_path / "videos" / "output")
    monkeypatch.setattr(server, "ROOT", tmp_path)
    return TestClient(server.app)


def test_runs_and_profiles(client):
    assert client.get("/api/runs").json() == ["r1"]
    assert [p["id"] for p in client.get("/api/runs/r1/profiles").json()] == ["A-1", "B-1"]
    assert client.get("/api/runs/nope/profiles").status_code == 404


def test_clips_flags(client):
    c = {x["clip"]: x for x in client.get("/api/runs/r1/clips").json()}
    assert c["k_01"]["athletes"] == {"aka": "A-1", "ao": "B-1"}
    assert c["k_01"]["has_features"] and c["k_01"]["has_video"]
    assert not c["k_02"]["has_features"] and not c["k_02"]["has_video"]


def test_bouts_final_score_is_max_and_only_this_run(client):
    (b,) = client.get("/api/runs/r1/bouts").json()
    assert b["title"] == "AAA vs BBB" and b["subtitle"] == "Senior Female Kumite"
    assert b["sides"]["aka"]["points"] == 3 and b["sides"]["ao"]["points"] == 2
    assert b["clips"] == ["k_01", "k_02"]


def test_features_reports(client):
    assert client.get("/api/runs/r1/features/k_01").json()["duration_s"] == 10.0
    assert client.get("/api/runs/r1/features/k_02").status_code == 404
    assert list(client.get("/api/runs/r1/reports").json()) == ["A-1"]
    assert client.get("/api/runs/r1/reports/B-1").status_code == 404


def test_video_supports_range(client):
    r = client.get("/api/runs/r1/videos/k_01", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
    assert client.get("/api/runs/r1/videos/k_02").status_code == 404


@pytest.mark.parametrize("name", ["..", ".hidden", "a b"])
def test_rejects_bad_names(client, name):
    assert client.get(f"/api/runs/r1/features/{name}").status_code in (400, 404)
    assert client.get(f"/api/runs/r1/reports/{name}").status_code in (400, 404)


def test_does_not_write(client, tmp_path):
    before = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    for u in ("/api/runs", "/api/runs/r1/profiles", "/api/runs/r1/clips", "/api/runs/r1/bouts",
              "/api/runs/r1/reports", "/api/runs/r1/features/k_01", "/api/runs/r1/videos/k_01"):
        client.get(u)
    after = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert os.path.exists(tmp_path / "data" / "roster.csv")
