"""LLM coach report per athlete, from the computed metrics only (no video/frames are sent).

    python scripts/coach_report.py --run v1 [--athlete FRA-1] [--model claude-opus-5-5]
    -> data/<run>/reports/<athlete_id>.json   (read by dashboard/app.py)

Needs Claude API credentials (ANTHROPIC_API_KEY or `ant auth login`); a Claude Pro chat
subscription does not include API access. Reports are cached files, so the demo never depends
on a live API call.
"""
import argparse
import json
from pathlib import Path

import anthropic
from pydantic import BaseModel

SYSTEM = """You are an experienced WKF kumite coach reviewing computer-vision match statistics.
Write for the athlete's coach: concrete, short, practical.

Rules:
- Ground every strength/weakness in a specific number from the data (value, cohort percentile or
  comparison with the opponent). Never invent events that are not in the data.
- Metrics are estimates from pose tracking. Respect `reliability`, `confidence` and `cohort_n`:
  with low confidence or a small cohort, phrase findings as hypotheses to check on video.
- Known limitation: the punch/kick split is unreliable at low video resolution; don't build
  conclusions on it.
- Focus items must be trainable (tactics, distance management, initiative, rhythm), with a drill each.

Metric definitions:
- strike_rate: punches+kicks per minute of live fighting
- initiative_share: share of exchanges this athlete started
- range_control: share of moving time this athlete pressed forward more than the opponent (dictated distance)
- tempo: strikes + forward attacks per 10 s of live fighting
- skill_pct: cohort percentile of the athlete's metric percentiles; standing_pct: cohort percentile of
  the result (bout points share); gap = skill_pct - standing_pct (positive = fights better than the score shows)
"""


class Report(BaseModel):
    headline: str
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    focus: list[str]
    drills: list[str]
    caveats: list[str]


def athlete_payload(p: dict, profiles: list[dict]) -> dict:
    cohort = [q for q in profiles if q["age_group"] == p["age_group"]]
    keys = ["id", "label", "age_group", "belt", "n_bouts", "reliability", "metrics", "metric_pct", "skill_pct",
            "standing_pct", "gap", "confidence", "cohort_n", "counts", "points_for", "points_against", "active_s"]
    opp_clips = {c["clip"] for c in p["clips"]}
    opponents = [{"label": q["label"], "metrics": q["metrics"], "counts": q["counts"]}
                 for q in profiles if q["id"] != p["id"] and opp_clips & {c["clip"] for c in q["clips"]}]
    return {
        "athlete": {k: p.get(k) for k in keys},
        "per_clip": [{k: c[k] for k in ("clip", "active_s", "metrics", "counts")} for c in p["clips"]],
        "opponents_same_bouts": opponents,
        "cohort_metric_values": {k: sorted(q["metrics"][k] for q in cohort if q["metrics"][k] is not None)
                                 for k in p["metrics"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--athlete", default=None)
    ap.add_argument("--model", default="claude-opus-5-5")
    args = ap.parse_args()
    run_dir = Path("data") / args.run
    profiles = json.loads((run_dir / "profiles.json").read_text(encoding="utf-8"))
    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    for p in profiles:
        if args.athlete and p["id"] != args.athlete:
            continue
        payload = athlete_payload(p, profiles)
        try:
            resp = client.messages.parse(
                model=args.model,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": "Coach report for this athlete:\n```json\n"
                           + json.dumps(payload, indent=1) + "\n```"}],
                output_format=Report,
            )
        except anthropic.AuthenticationError:
            raise SystemExit("No valid Claude API credentials (set ANTHROPIC_API_KEY or run `ant auth login`).")
        except anthropic.APIStatusError as e:
            print(f"{p['id']}: API error {e.status_code}: {e.message}")
            continue
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            print(f"{p['id']}: no report (stop_reason={resp.stop_reason})")
            continue
        rep = resp.parsed_output.model_dump() | {"model": args.model, "athlete_id": p["id"]}
        (out_dir / f"{p['id']}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"{p['id']}: {rep['headline']}")


if __name__ == "__main__":
    main()
