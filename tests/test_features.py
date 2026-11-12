"""Karate features on a synthetic bout: AKA advances and throws one straight punch at ~1.04 s,
AO stands still. Pure logic, no model or video."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.features import analyze_bout, load_bout  # noqa: E402

FPS = 25.0


def _person(cx, punch=0.0, facing=1, knee_up=False):
    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    pts = {0: (cx, 180), 5: (cx - 8, 200), 6: (cx + 8, 200), 11: (cx - 6, 260), 12: (cx + 6, 260),
           13: (cx - 8, 250 if knee_up else 310), 14: (cx + 8, 310), 15: (cx - 10, 360), 16: (cx + 10, 360),
           7: (cx - 5, 230), 9: (cx + facing * 10, 215)}
    for i, (x, y) in pts.items():
        kp[i] = [x, y, 0.9]
    kp[8] = [cx + facing * (10 + 20 * punch), 210, 0.9]
    kp[10] = [cx + facing * (15 + 50 * punch), 205 - 5 * punch, 0.9]
    return kp, [cx - 25, 170, cx + 25, 365]


def _write_bout(path: Path, punch_at=1.1, graphic=None, ao_x=400, knee=None):
    with path.open("w") as f:
        for t in range(100):
            ts = t / FPS
            aka_x = 200 + 60 * min(1, max(0, (ts - 0.5) / 0.5))
            p = 0.0 if punch_at is None else max(0.0, 1 - abs(ts - punch_at) / 0.12)
            k = bool(knee and knee[0] <= ts < knee[1])
            for aid, cx, pu, fa, ku in (("aka", aka_x, p, 1, k), ("ao", ao_x, 0.0, -1, False)):
                kp, bb = _person(cx, pu, fa, ku)
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


def test_raised_knee_does_not_turn_a_punch_into_a_kick(tmp_path):
    # lunge punch: the knee comes up while the hand strikes (FRA's yuko, clip 02 18.8 s, looked like this)
    p = tmp_path / "bout.jsonl"
    _write_bout(p, knee=(0.95, 1.25))
    aka = analyze_bout(load_bout(p))["athletes"]["aka"]
    assert aka["counts"]["strikes"] == 1 and aka["counts"]["punches"] == 1 and aka["counts"]["kicks"] == 0
    assert aka["events"][0]["type"] == "punch" and aka["events"][0]["limb"].endswith("hand")
    assert aka["counts"]["knee_raises"] == 0  # the knee was part of the punch, not a feint


def test_reach_close_vs_far(tmp_path):
    far, near = tmp_path / "far.jsonl", tmp_path / "near.jsonl"
    _write_bout(far)               # fist stops ~1 torso short of the opponent's shoulder
    _write_bout(near, ao_x=340)    # opponent close enough that the fist gets to the shoulder
    ef = analyze_bout(load_bout(far))["athletes"]["aka"]["events"][0]
    en = analyze_bout(load_bout(near))["athletes"]["aka"]["events"][0]
    assert ef["reach"] > 0.8 and ef["reached"] is False
    assert en["reach"] < 0.5 and en["reached"] is True
    assert en["reach"] < ef["reach"]


def test_knee_raise_without_strike_is_a_possible_feint(tmp_path):
    p = tmp_path / "bout.jsonl"
    _write_bout(p, punch_at=None, knee=(2.0, 2.4))
    r = analyze_bout(load_bout(p))
    aka = r["athletes"]["aka"]
    assert aka["counts"]["strikes"] == 0 and aka["counts"]["knee_raises"] == 1
    assert abs(aka["knee_raises"][0]["t"] - 2.0) < 0.1
    plain = tmp_path / "plain.jsonl"
    _write_bout(plain, punch_at=None)
    # not counted as an action anywhere: same metrics as the same bout without the knee raise
    assert aka["metrics"] == analyze_bout(load_bout(plain))["athletes"]["aka"]["metrics"]
