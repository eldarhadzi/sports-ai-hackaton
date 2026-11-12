"""Karate match features from the pipeline's per-frame JSONL (one bout = one clip).

    python src/analysis/features.py data/<run>/<clip>.jsonl [--out data/<run>/features/<clip>.json]

Everything is computed from pose keypoints only (no extra model) and normalized by
body size (torso length), so it works in pixel space and on a zooming camera.
Units below: "torso" = one shoulder-to-hip length (~0.5 m on an adult).

Definitions (heuristics - tune DEFAULTS against a hand count):
  strike         a wrist (punch) or ankle (kick) speed peak relative to the athlete's own hips,
                 fast enough, with the arm extended / the foot raised, moving toward the
                 opponent, while the two athletes are within striking range. One strike per
                 athlete per `refractory_s`. The type comes from the limb that struck (hand =
                 punch, foot = kick); a raised knee never turns a hand strike into a kick.
  reach          per strike: closest 2D distance (torso units) from the striking wrist/ankle to
                 the opponent's head/shoulders/hips around the peak. `reached` = reach <= reach_thr.
                 A proxy, NOT contact: depth is invisible in 2D, and a far-side arm hidden behind
                 the body reads as "not reached" (FRA's yuko punch, clip 02 18.8 s: reach 1.04).
                 A low reach is a trustworthy "got there"; a high one is not proof of a miss.
  knee raise     knee chambered / ankle up for >= knee_raise_min_s, in range, with no strike of
                 the same athlete within +-knee_raise_quiet_s: a possible feint. Not a strike and
                 not part of any metric.
  advance        the athlete moves toward the opponent faster than `advance_speed` for at least
                 `advance_min_s`.
  exchange       a cluster of strikes (either athlete) with gaps <= `exchange_gap_s`.
                 initiator = whoever first advanced (within `lead_window_s` before) or struck.
  strike_rate    strikes per minute of active time (both athletes tracked).
  initiative_share  exchanges this athlete initiated / exchanges with a known initiator.
  range_control  share of "moving" frames where this athlete presses more than the opponent
                 (forward speed toward opponent minus the opponent's) - who dictates distance.
  tempo          actions (strikes + advances) per 10 s of active time.
  reliability    tracked-both coverage x identity-known share x keypoint quality.
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AKA, AO = "aka", "ao"
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

# name: (tip, root, middle, kind)
LIMBS = {
    "left_hand": (L_WR, L_SH, L_EL, "punch"),
    "right_hand": (R_WR, R_SH, R_EL, "punch"),
    "left_foot": (L_AN, L_HIP, L_KN, "kick"),
    "right_foot": (R_AN, R_HIP, R_KN, "kick"),
}

PAUSE_UNPAD_S = 0.5  # = config.yaml classifier.graphic_fade_pad_seconds

DEFAULTS = {
    "kp_conf": 0.3,            # keypoints below this confidence are treated as missing
    "pos_smooth_s": 0.15,      # smoothing of body (hip) position
    "limb_smooth_s": 0.07,     # smoothing of limb tip position
    "punch_speed": 5.0,        # torso/s, wrist speed relative to own hips
    "kick_speed": 5.0,         # torso/s, ankle speed relative to own hips
    "punch_extension": 0.85,   # shoulder-wrist distance / arm length
    "kick_height": -0.8,       # (hip_y - ankle_y)/torso; standing ~ -1.7, ankle at knee ~ -0.8
    "knee_kick_height": -0.35, # (hip_y - knee_y)/torso; standing ~ -0.8, chambered knee ~ -0.2
    "toward_cos": 0.3,         # cos(limb velocity, direction to opponent)
    "max_range": 6.0,          # torso units between the hips; beyond this nothing can land
    "min_range": 1.2,          # closer than this = clinch; arm motion there is grip-fighting
    "chamber_window_s": 0.3,   # the strike shape must develop within this time before the peak
    "punch_delta": 0.25,       # arm extension gained (fraction of arm length)
    "kick_delta": 0.6,         # ankle height gained (torso units)
    "peak_window_s": 0.12,     # speed peak must be the max within +-this
    "refractory_s": 0.35,      # min time between two strikes of the same athlete
    "advance_speed": 1.5,      # torso/s toward the opponent
    "advance_min_s": 0.2,
    "exchange_gap_s": 1.0,
    "lead_window_s": 0.8,
    "moving_thr": 0.5,         # torso/s, combined forward speed for a frame to count as "moving"
    "timeline_step_s": 0.5,
    "idle_window_s": 1.0,
    "idle_speed": 1.0,         # torso/s, both athletes' hip speeds summed
    "idle_dist": 4.0,          # torso units apart
    "id_jump_mask_s": 0.1,     # ignore limb speed around a tracker-id change (it's a jump, not a strike)
    "reach_thr": 0.5,          # torso units: limb tip this close to the opponent's head/shoulders/hips = reached
    "knee_raise_min_s": 0.12,  # knee chambered at least this long to count as a knee raise
    "knee_raise_quiet_s": 0.3, # ...and no strike by the same athlete within +-this (else it was part of that strike)
}

# Opponent target points for "reach": nose, shoulders, hips (COCO indices).
TARGETS = (0, 5, 6, 11, 12)


# ---------------------------------------------------------------- loading

def load_bout(jsonl_path: Path, total_frames: int | None = None, fps: float | None = None) -> dict:
    """Per-athlete arrays over the whole clip (NaN where the athlete isn't tracked)."""
    rows = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
    meta_path = Path(jsonl_path).with_name(Path(jsonl_path).stem + "_run_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        total_frames = total_frames or meta.get("frames_processed")
        fps = fps or meta.get("input_fps")
    if not rows:
        raise ValueError(f"{jsonl_path} has no records")
    T = total_frames or (max(r["frame"] for r in rows) + 1)
    if not fps:
        ts = sorted({(r["frame"], r["timestamp"]) for r in rows})
        (f0, t0), (f1, t1) = ts[0], ts[-1]
        fps = (f1 - f0) / (t1 - t0) if t1 > t0 else 30.0

    out = {"T": T, "fps": float(fps), "paused": np.zeros(T, dtype=bool)}
    for r in rows:
        if r.get("graphic_on") and r["frame"] < T:
            out["paused"][r["frame"]] = True
    # the pipeline widens graphic windows by a fade pad (to hide pictograms); undo it here, the
    # bout often restarts right as the graphic fades out
    pad = int(round(PAUSE_UNPAD_S * fps))
    for s, e in _runs(out["paused"]):
        out["paused"][s:e] = False
        if e - s > 2 * pad:
            out["paused"][s + pad:e - pad] = True
    for a in (AKA, AO):
        out[a] = {
            "kp": np.full((T, 17, 3), np.nan),
            "bbox": np.full((T, 4), np.nan),
            "tid": np.full(T, -1, dtype=int),
            "known": np.zeros(T, dtype=bool),
        }
    best_conf = {}
    for r in rows:
        a, t = r.get("athlete_id"), r["frame"]
        if a not in (AKA, AO) or t >= T:
            continue
        if best_conf.get((a, t), -1) >= r["confidence"]:
            continue
        best_conf[(a, t)] = r["confidence"]
        d = out[a]
        d["kp"][t] = np.asarray(r["keypoints"], dtype=float)
        d["bbox"][t] = r["bbox"]
        d["tid"][t] = -1 if r["tracker_id"] is None else r["tracker_id"]
        d["known"][t] = bool(r.get("identity_known"))
    return out


# ---------------------------------------------------------------- signal helpers

def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centered NaN-aware moving average along axis 0."""
    if w <= 1:
        return x.copy()
    flat = x.reshape(len(x), -1)
    df = pd.DataFrame(flat).rolling(w, center=True, min_periods=1).mean()
    res = df.to_numpy().copy()
    res[np.isnan(flat)] = np.nan  # don't invent values inside gaps
    return res.reshape(x.shape)


def _deriv(x: np.ndarray, fps: float) -> np.ndarray:
    d = np.full_like(x, np.nan)
    d[1:-1] = (x[2:] - x[:-2]) * fps / 2.0
    return d


def _mid(kp, i, j, conf):
    """Midpoint of keypoints i, j (falls back to whichever one is confident)."""
    a, b = kp[:, i, :2].copy(), kp[:, j, :2].copy()
    a[kp[:, i, 2] < conf] = np.nan
    b[kp[:, j, 2] < conf] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([a, b]), axis=0)


def _pt(kp, i, conf):
    p = kp[:, i, :2].copy()
    p[~(kp[:, i, 2] >= conf)] = np.nan
    return p


def _norm(v):
    return np.linalg.norm(v, axis=-1)


def _runs(mask: np.ndarray):
    """(start, end_exclusive) of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# ---------------------------------------------------------------- features

def body_signals(d: dict, fps: float, cfg: dict) -> dict:
    kp, c = d["kp"], cfg["kp_conf"]
    sh, hip = _mid(kp, L_SH, R_SH, c), _mid(kp, L_HIP, R_HIP, c)
    torso = _norm(sh - hip)
    bbox_h = d["bbox"][:, 3] - d["bbox"][:, 1]
    torso = np.where(np.isnan(torso) | (torso < 0.15 * bbox_h), bbox_h / 3.0, torso)
    scale = pd.Series(torso).rolling(int(2 * fps), center=True, min_periods=1).median().to_numpy().copy()
    scale[np.isnan(torso)] = np.nan
    w = max(1, int(round(cfg["pos_smooth_s"] * fps)))
    hip_s = _smooth(hip, w)
    return {"hip": hip_s, "scale": scale, "present": ~np.isnan(hip_s[:, 0]) & ~np.isnan(scale)}


def _reach(tip: np.ndarray, opp_kp: np.ndarray, conf: float, scale: np.ndarray, t: int, win: int) -> float | None:
    """Closest 2D distance (own torso units) from a limb tip to the opponent's head/shoulders/hips
    around frame t. It's a proxy: depth is invisible in 2D, so "reached" never means "made contact"."""
    sl = slice(max(0, t - win), t + win + 1)
    tg = np.stack([_pt(opp_kp[sl], i, conf) for i in TARGETS], axis=1)  # (w, 5, 2)
    d = _norm(tg - tip[sl][:, None, :]) / scale[sl][:, None]
    return None if np.isnan(d).all() else round(float(np.nanmin(d)), 2)


def detect_strikes(d, me, opp_d, fps, cfg, toward, dist, paused):
    """Strikes of one athlete. The type comes from the limb that struck (hand = punch, foot = kick).

    `opp_d` is the opponent's raw arrays (for reach). Returns (strikes, knee_raises)."""
    kp, c = d["kp"], cfg["kp_conf"]
    T = len(kp)
    # leg raised (knee chambered or ankle up). Used only for knee raises below; it does NOT turn a hand
    # strike into a kick (a lunge punch lifts the knee too, e.g. FRA's yuko, clip 02 at 18.8 s).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        knee_h = np.nanmax(np.stack([(me["hip"][:, 1] - _pt(kp, k, c)[:, 1]) / me["scale"] for k in (L_KN, R_KN)]), axis=0)
        ankle_h = np.nanmax(np.stack([(me["hip"][:, 1] - _pt(kp, k, c)[:, 1]) / me["scale"] for k in (L_AN, R_AN)]), axis=0)
    leg_up = (knee_h >= cfg["knee_kick_height"]) | (ankle_h >= cfg["kick_height"])
    win = max(1, int(round(cfg["peak_window_s"] * fps)))
    pre = max(1, int(round(cfg["chamber_window_s"] * fps)))
    lw = max(1, int(round(cfg["limb_smooth_s"] * fps)))
    # mask frames around a tracker-id change: the "limb motion" there is a jump between people
    jump = np.zeros(T, dtype=bool)
    tid = d["tid"]
    changes = np.where((tid[1:] != tid[:-1]) & (tid[1:] >= 0) & (tid[:-1] >= 0))[0] + 1
    m = max(1, int(round(cfg["id_jump_mask_s"] * fps)))
    for ch in changes:
        jump[max(0, ch - m): ch + m + 1] = True

    cands = []
    for limb, (tip_i, root_i, mid_i, kind) in LIMBS.items():
        tip, root, mid = _pt(kp, tip_i, c), _pt(kp, root_i, c), _pt(kp, mid_i, c)
        rel = _smooth((tip - me["hip"]) / me["scale"][:, None], lw)
        speed = _norm(_deriv(rel, fps))
        speed[jump] = np.nan
        vel = _deriv(_smooth(tip, lw), fps)
        cos = np.einsum("ij,ij->i", vel, toward) / np.maximum(_norm(vel), 1e-6)
        if kind == "punch":
            seg = np.nanmedian(_norm(root - mid) + _norm(mid - tip))
            shape = _norm(tip - root) / seg if seg and not math.isnan(seg) else np.full(T, np.nan)
            shape_ok, thr = cfg["punch_extension"], cfg["punch_speed"]
        else:
            shape = (me["hip"][:, 1] - tip[:, 1]) / me["scale"]
            shape_ok, thr = cfg["kick_height"], cfg["kick_speed"]
        s = np.nan_to_num(speed, nan=0.0)
        for t in range(win, T - win):
            if paused[t] or s[t] < thr or s[t] < s[t - win: t + win + 1].max():
                continue
            if not (cfg["min_range"] <= dist[t] <= cfg["max_range"]) or not (cos[t] >= cfg["toward_cos"]):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sh_max = np.nanmax(shape[t - win: t + win + 1])
                sh_min = np.nanmin(shape[max(0, t - pre): t + 1])
            # must reach the strike shape AND get there from a clearly retracted one (guards and
            # grip-fighting move fast but never go bent -> extended)
            delta = cfg["punch_delta"] if kind == "punch" else cfg["kick_delta"]
            if not (sh_max >= shape_ok) or not (sh_max - sh_min >= delta):
                continue
            reach = _reach(tip, opp_d["kp"], c, me["scale"], t, win)
            cands.append({"frame": t, "t": round(t / fps, 2), "type": kind, "limb": limb,
                          "speed": round(float(s[t]), 1), "reach": reach,
                          "reached": None if reach is None else bool(reach <= cfg["reach_thr"])})
    cands.sort(key=lambda e: e["frame"])
    strikes, last = [], -1e9
    ref = cfg["refractory_s"] * fps
    for e in cands:
        if e["frame"] - last >= ref:
            strikes.append(e)
            last = e["frame"]
        elif e["speed"] > strikes[-1]["speed"]:
            strikes[-1] = e  # same action seen on two limbs: keep the stronger read

    # knee raises: knee/ankle up for a moment, in range, live, with no strike of this athlete around it.
    # A possible feint or a chambered kick that never came - not a strike, not counted in any metric.
    quiet = cfg["knee_raise_quiet_s"] * fps
    knee_raises = []
    for s0, e0 in _runs(leg_up & ~paused):
        if e0 - s0 < cfg["knee_raise_min_s"] * fps:
            continue
        mid_f = (s0 + e0 - 1) // 2
        if not (cfg["min_range"] <= np.nan_to_num(dist[mid_f], nan=-1) <= cfg["max_range"]):
            continue
        if any(s0 - quiet <= e["frame"] <= e0 - 1 + quiet for e in strikes):
            continue
        knee_raises.append({"frame": int(s0), "t": round(s0 / fps, 2), "duration_s": round((e0 - s0) / fps, 2)})
    return strikes, knee_raises


def analyze_bout(bout: dict, cfg: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    T, fps = bout["T"], bout["fps"]
    sig = {a: body_signals(bout[a], fps, cfg) for a in (AKA, AO)}
    vec = sig[AO]["hip"] - sig[AKA]["hip"]
    gap = _norm(vec)
    unit = vec / np.maximum(gap, 1e-6)[:, None]
    mean_scale = (sig[AKA]["scale"] + sig[AO]["scale"]) / 2
    dist = gap / mean_scale  # torso units
    # idle = both barely moving while out of range for ~1 s (walk-back, waiting for "hajime")
    spd = sum(_norm(_deriv(sig[a]["hip"], fps)) / sig[a]["scale"] for a in (AKA, AO))
    calm = pd.Series(spd).rolling(int(cfg["idle_window_s"] * fps), center=True, min_periods=1).mean().to_numpy()
    idle = (calm < cfg["idle_speed"]) & (dist > cfg["idle_dist"])
    graphic = bout.get("paused", np.zeros(T, dtype=bool))
    paused = graphic | idle
    # "active" = both athletes tracked and the bout is live (no score graphic, not idle)
    both = sig[AKA]["present"] & sig[AO]["present"] & ~paused
    toward = {AKA: unit, AO: -unit}

    fwd = {}
    for a in (AKA, AO):
        v = _deriv(sig[a]["hip"], fps)
        fwd[a] = np.einsum("ij,ij->i", v, toward[a]) / sig[a]["scale"]

    res = {"T": T, "fps": fps, "duration_s": round(T / fps, 2), "active_s": round(both.sum() / fps, 2),
           "paused_s": round(float(graphic.sum() / fps), 2), "idle_s": round(float((idle & ~graphic).sum() / fps), 2),
           "athletes": {}, "exchanges": [], "config": cfg}
    detected = {a: detect_strikes(bout[a], sig[a], bout[AO if a == AKA else AKA], fps, cfg, toward[a], dist, paused)
                for a in (AKA, AO)}
    strikes = {a: detected[a][0] for a in (AKA, AO)}
    knee_raises = {a: detected[a][1] for a in (AKA, AO)}

    advances = {}
    for a in (AKA, AO):
        runs = _runs((np.nan_to_num(fwd[a], nan=0.0) >= cfg["advance_speed"]) & ~paused)
        advances[a] = [s for s, e in runs if (e - s) >= cfg["advance_min_s"] * fps]

    # exchanges
    allst = sorted([(e["frame"], a) for a in (AKA, AO) for e in strikes[a]])
    clusters = []
    for f, a in allst:
        if clusters and f - clusters[-1][-1][0] <= cfg["exchange_gap_s"] * fps:
            clusters[-1].append((f, a))
        else:
            clusters.append([(f, a)])
    lead = cfg["lead_window_s"] * fps
    for cl in clusters:
        f0 = cl[0][0]
        first = {}
        for a in (AKA, AO):
            starts = [s for s in advances[a] if f0 - lead <= s <= f0]
            hits = [f for f, b in cl if b == a]
            cand = (starts[:1] or []) + hits[:1]
            if cand:
                first[a] = min(cand)
        initiator = None
        if first:
            ordered = sorted(first.items(), key=lambda kv: kv[1])
            if len(ordered) == 1 or ordered[0][1] < ordered[1][1]:
                initiator = ordered[0][0]
        res["exchanges"].append({"t0": round(f0 / fps, 2), "t1": round(cl[-1][0] / fps, 2),
                                 "n_strikes": len(cl), "initiator": initiator})
    known_ex = [x for x in res["exchanges"] if x["initiator"]]

    # range control
    moving = both & ((np.abs(np.nan_to_num(fwd[AKA])) + np.abs(np.nan_to_num(fwd[AO]))) >= cfg["moving_thr"])
    press = np.nan_to_num(fwd[AKA] - fwd[AO])  # >0: AKA presses
    aka_rc = float((press[moving] > 0).mean()) if moving.any() else 0.5

    active_min = max(both.sum() / fps / 60.0, 1e-6)
    step = max(1, int(round(cfg["timeline_step_s"] * fps)))
    for a in (AKA, AO):
        opp = AO if a == AKA else AKA
        d = bout[a]
        present = sig[a]["present"]
        known_share = float(d["known"][both].mean()) if both.any() else 0.0
        limb_idx = [L_WR, R_WR, L_AN, R_AN, L_SH, R_SH, L_HIP, R_HIP]
        kpq = float(np.nanmean(d["kp"][present][:, limb_idx, 2])) if present.any() else 0.0
        coverage = float(both.sum() / max(1, (~paused).sum()))  # tracked share of live (non-paused) time
        reliability = coverage * known_share * min(1.0, kpq / 0.6)
        n_str = len(strikes[a])
        n_adv = len(advances[a])
        r = np.nan_to_num(fwd[a] - fwd[opp])
        strike_frames = {e["frame"] for e in strikes[a]}
        timeline = []
        for t0 in range(0, T, step):
            seg = slice(t0, min(T, t0 + step))
            m = both[seg]
            val = 0.5 + 0.5 * math.tanh(float(r[seg][m].mean()) / 1.5) if m.any() else None
            timeline.append({"t": round(t0 / fps, 2),
                             "initiative": None if val is None else round(val, 2),
                             "strike": any(f in strike_frames for f in range(seg.start, seg.stop))})
        res["athletes"][a] = {
            "reliability": round(reliability, 2),
            "coverage": round(coverage, 3), "identity_known": round(known_share, 3), "keypoint_quality": round(kpq, 3),
            "metrics": {
                "strike_rate": round(float(n_str / active_min), 1),
                "initiative_share": round(sum(x["initiator"] == a for x in known_ex) / len(known_ex), 2) if known_ex else None,
                "range_control": round(aka_rc if a == AKA else 1 - aka_rc, 2),
                "tempo": round(float((n_str + n_adv) / (active_min * 6.0)), 1),
            },
            "counts": {
                "strikes": n_str, "punches": sum(e["type"] == "punch" for e in strikes[a]),
                "kicks": sum(e["type"] == "kick" for e in strikes[a]), "advances": n_adv,
                "reached": sum(bool(e["reached"]) for e in strikes[a]),
                "reach_known": sum(e["reached"] is not None for e in strikes[a]),
                "knee_raises": len(knee_raises[a]),
                "exchanges_initiated": sum(x["initiator"] == a for x in known_ex),
                "exchanges_total": len(res["exchanges"]),
            },
            "events": strikes[a],
            "knee_raises": knee_raises[a],
            "timeline": timeline,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    jp = Path(args.jsonl)
    res = analyze_bout(load_bout(jp))
    res["clip"] = jp.stem
    out = Path(args.out) if args.out else jp.parent / "features" / f"{jp.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for a, v in res["athletes"].items():
        print(f"{a}: reliability={v['reliability']} metrics={v['metrics']} counts={v['counts']}")
    print(f"exchanges={len(res['exchanges'])} active={res['active_s']}s of {res['duration_s']}s -> {out}")


if __name__ == "__main__":
    main()
