"""Prints the per-clip metrics report for OLD (round 2, pre this-round fixes)
vs NEW (this round: real imgsz wiring, belt/glove color lock, timestamp-based
trails) side by side, for all 6 clips. Run after both data/baseline_round2/
and data/ hold a full set of *_metrics.json.

    python scripts/compare_metrics.py
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLD_DIR = DATA_DIR / "baseline_round2"

CLIPS = [
    "kumite_01_siroki_kadar",
    "kumite_02_siroki_kadar",
    "kumite_03_siroki_kadar",
    "kumite_04_siroki_kadar_pauza",
    "kumite_05_siroki_kadar",
    "kumite_06_medium_kamera_prati",
]

FIELDS = [
    ("pct_both_assigned", "both%"),
    ("pct_one_assigned", "one%"),
    ("pct_zero_assigned", "zero%"),
    ("id_switch_count", "id_sw"),
    ("color_swap_count", "flips"),
    ("filter_violations", "filt_viol"),
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    header = f"{'clip':<32}" + "".join(f"{label:>10}" for _, label in FIELDS)
    print("OLD (round 2) vs NEW (this round) - NEW value shown, (OLD) in parens")
    print(header)
    print("-" * len(header))

    for clip in CLIPS:
        old = load(OLD_DIR / f"{clip}_metrics.json")
        new = load(DATA_DIR / f"{clip}_metrics.json")
        if new is None:
            print(f"{clip:<32}  NEW METRICS MISSING - run main.py first")
            continue
        row = f"{clip:<32}"
        for key, _ in FIELDS:
            new_v = new.get(key, "?")
            old_v = old.get(key, "?") if old else "?"
            row += f"{f'{new_v}({old_v})':>10}"
        print(row)

    print()
    print("both%/one%/zero% = % of frames with 2/1/0 slots assigned (both is the target).")
    print("id_sw = tracker-ID reassignment count. flips = AKA/AO color-swap corrections.")
    print("filt_viol should always be 0 (an assigned slot must pass the athlete filter that frame).")


if __name__ == "__main__":
    main()
