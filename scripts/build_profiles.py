"""Per-athlete profiles (the team's JSON format) from per-clip features + a roster.

    python scripts/build_profiles.py --run v1 [--roster data/roster.csv]
    -> data/<run>/profiles.json   (list of athlete objects)

Roster CSV columns (one row per clip x slot):
    clip,slot,athlete_id,label,age_group,belt,bout_id,points_for,points_against
  slot            aka | ao (as the pipeline labels them: aka = red belt)
  bout_id         clips cut from the same bout share it, so they count as ONE bout
  points_for/against  scoreboard at the END of that clip (cumulative; max per bout is used)

Definitions:
  metrics         active-time-weighted over all clips (initiative_share: pooled exchanges)
  skill_pct       percentile (within cohort = same age_group) of the mean of the athlete's
                  per-metric cohort percentiles
  standing_pct    percentile within cohort of the bout points share (points_for / total).
                  NOTE: placeholder definition until the team confirms what "standing" means.
  gap             skill_pct - standing_pct (positive = performs better than the result shows)
  confidence      high: reliability >= 0.85 and >= 3 bouts and cohort >= 5;
                  medium: reliability >= 0.7; else low
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METRICS = ["strike_rate", "initiative_share", "range_control", "tempo"]


def pct_rank(value, values):
    """Percentile of value within values (ties count half), 0-100."""
    vals = [v for v in values if v is not None]
    if value is None or not vals:
        return None
    below = sum(v < value for v in vals)
    equal = sum(v == value for v in vals)
    return round(100.0 * (below + 0.5 * equal) / len(vals))


def build(run_dir: Path, roster_path: Path) -> list[dict]:
    roster = list(csv.DictReader(roster_path.open(encoding="utf-8")))
    per_athlete = defaultdict(list)
    info = {}
    for row in roster:
        fpath = run_dir / "features" / f"{row['clip']}.json"
        if not fpath.exists():
            print(f"skip {row['clip']}: no features ({fpath})")
            continue
        feat = json.loads(fpath.read_text())
        a = feat["athletes"][row["slot"]]
        per_athlete[row["athlete_id"]].append((row, feat, a))
        info[row["athlete_id"]] = row

    profiles = []
    for aid, items in per_athlete.items():
        row0 = info[aid]
        active = [f["active_s"] for _, f, _ in items]
        tot = sum(active) or 1e-6
        m = {}
        for k in ("strike_rate", "range_control", "tempo"):
            m[k] = round(sum(a["metrics"][k] * w for (_, _, a), w in zip(items, active)) / tot, 2)
        init = sum(a["counts"]["exchanges_initiated"] for _, _, a in items)
        known = sum(sum(1 for x in f["exchanges"] if x["initiator"]) for _, f, _ in items)
        m["initiative_share"] = round(init / known, 2) if known else None
        rel = round(sum(a["reliability"] * w for (_, _, a), w in zip(items, active)) / tot, 2)

        bouts = defaultdict(lambda: [0.0, 0.0])
        for r, _, _ in items:
            b = bouts[r.get("bout_id") or r["clip"]]
            b[0] = max(b[0], float(r.get("points_for") or 0))
            b[1] = max(b[1], float(r.get("points_against") or 0))
        pf, pa = sum(b[0] for b in bouts.values()), sum(b[1] for b in bouts.values())

        # timeline: clips concatenated in roster order, times offset so they don't overlap
        timeline, events, knee_raises, offset = [], [], [], 0.0
        clips = []
        for r, f, a in items:
            timeline += [{**p, "t": round(p["t"] + offset, 2), "clip": r["clip"]} for p in a["timeline"]]
            events += [{**e, "t": round(e["t"] + offset, 2), "clip": r["clip"], "clip_t": e["t"]} for e in a["events"]]
            knee_raises += [{**e, "t": round(e["t"] + offset, 2), "clip": r["clip"], "clip_t": e["t"]}
                            for e in a.get("knee_raises", [])]
            clips.append({"clip": r["clip"], "slot": r["slot"], "offset": round(offset, 2),
                          "duration_s": f["duration_s"], "active_s": f["active_s"],
                          "metrics": a["metrics"], "counts": a["counts"], "reliability": a["reliability"]})
            offset += f["duration_s"]

        profiles.append({
            "id": aid, "label": row0.get("label") or aid,
            "age_group": row0.get("age_group") or "unknown", "belt": row0.get("belt") or "unknown",
            "n_bouts": len(bouts), "n_clips": len(items), "active_s": round(sum(active), 1),
            "reliability": rel, "metrics": m,
            "counts": {k: sum(a["counts"].get(k, 0) for _, _, a in items)
                       for k in ("strikes", "punches", "kicks", "advances", "exchanges_initiated",
                                 "reached", "reach_known", "knee_raises")},
            "points_for": pf, "points_against": pa,
            "_points_share": pf / (pf + pa) if (pf + pa) else None,
            "timeline": timeline, "events": events, "knee_raises": knee_raises, "clips": clips,
        })

    # cohort percentiles
    cohorts = defaultdict(list)
    for p in profiles:
        cohorts[p["age_group"]].append(p)
    for group in cohorts.values():
        n = len(group)
        for p in group:
            p["metric_pct"] = {k: pct_rank(p["metrics"][k], [q["metrics"][k] for q in group]) for k in METRICS}
            ps = [v for v in p["metric_pct"].values() if v is not None]
            p["_skill_raw"] = sum(ps) / len(ps) if ps else None
        for p in group:
            p["cohort_n"] = n
            p["skill_pct"] = pct_rank(p["_skill_raw"], [q["_skill_raw"] for q in group])
            p["standing_pct"] = pct_rank(p["_points_share"], [q["_points_share"] for q in group])
            p["standing_basis"] = "bout points share (placeholder)"
            p["gap"] = (p["skill_pct"] - p["standing_pct"]) if None not in (p["skill_pct"], p["standing_pct"]) else None
            if p["reliability"] >= 0.85 and p["n_bouts"] >= 3 and n >= 5:
                p["confidence"] = "high"
            elif p["reliability"] >= 0.7:
                p["confidence"] = "medium"
            else:
                p["confidence"] = "low"
    for p in profiles:
        for k in ("_skill_raw", "_points_share"):
            p.pop(k, None)
    order = ["id", "label", "age_group", "belt", "n_bouts", "reliability", "metrics", "skill_pct", "standing_pct",
             "gap", "confidence", "cohort_n"]
    return [{**{k: p[k] for k in order}, **{k: v for k, v in p.items() if k not in order}} for p in profiles]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--roster", default="data/roster.csv")
    args = ap.parse_args()
    run_dir = Path("data") / args.run
    profiles = build(run_dir, Path(args.roster))
    out = run_dir / "profiles.json"
    out.write_text(json.dumps(profiles, indent=1))
    for p in profiles:
        print(f"{p['id']:>10} {p['label']:<12} bouts={p['n_bouts']} rel={p['reliability']} {p['metrics']} "
              f"skill={p['skill_pct']} standing={p['standing_pct']} gap={p['gap']} conf={p['confidence']} n={p['cohort_n']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
