"""Karate features on a synthetic bout: AKA advances and throws one straight punch at ~1.04 s,
AO stands still. Pure logic, no model or video."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.features import analyze_bout, load_bout  # noqa: E402

FPS = 25.0


def _person(cx, punch=0.0, facing=1):
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    pts = {0: (cx, 180), 5: (cx - 8, 200), 6: (cx + 8, 200), 11: (cx - 6, 260), 12: (cx + 6, 260),
           13: (cx - 8, 310), 14: (cx + 8, 310), 15: (cx - 10, 360), 16: (cx + 10, 360),
           7: (cx - 5, 230), 9: (cx + facing * 10, 215)}
    for i, (x, y) in pts.items():
        kp[i] = [x, y, 0.9]
    kp[8] = [cx + facing * (10 + 20 * punch), 210, 0.9]
    kp[10] = [cx + facing * (15 + 50 * punch), 205 - 5 * punch, 0.9]
    return kp, [cx - 25, 170, cx + 25, 365]


def _write_bout(path: Path, punch_at=1.1, graphic=None):
    with path.open("w") as f:
        for t in range(100):
            ts = t / FPS
            aka_x = 200 + 60 * min(1, max(0, (ts - 0.5) / 0.5))
            p = max(0.0, 1 - abs(ts - punch_at) / 0.12)
            for aid, cx, pu, fa in (("aka", aka_x, p, 1), ("ao", 400, 0.0, -1)):
                kp, bb = _person(cx, pu, fa)
                f.write(json.dumps({
                    "frame": t, "timestamp": ts, "athlete_id": aid, "tracker_id": 1 if aid == "aka" else 2,
                    "bbox": bb, "confidence": 0.9, "keypoints": kp, "ground_point": [cx, 365],
                    "identity_known": True, "graphic_on": bool(graphic and graphic[0] <= ts < graphic[1]),
                }) + "\n")


def test_punch_exchange_and_range_control(tmp_path):
    p = tmp_path / "bout.jsonl"
    _write_bout(p)
    r = analyze_bout(load_bout(p))
    aka, ao = r["athletes"]["aka"], r["athletes"]["ao"]
    assert aka["counts"]["punches"] == 1 and ao["counts"]["strikes"] == 0
    assert abs(aka["events"][0]["t"] - 1.04) < 0.15
    assert len(r["exchanges"]) == 1 and r["exchanges"][0]["initiator"] == "aka"
    assert aka["metrics"]["initiative_share"] == 1.0
    assert aka["metrics"]["range_control"] == 1.0 and ao["metrics"]["range_control"] == 0.0
    assert aka["reliability"] > 0.9


def test_no_strikes_counted_while_score_graphic_is_up(tmp_path):
    p = tmp_path / "bout.jsonl"
    _write_bout(p, graphic=(0.0, 3.0))  # the punch at 1.1 s falls inside a long stoppage
    r = analyze_bout(load_bout(p))
    assert r["athletes"]["aka"]["counts"]["strikes"] == 0
    assert r["paused_s"] > 1.5
