"""Per-clip diagnostic metrics for athlete selection/ID assignment - numbers
to compare runs by, not just eyeballed video review."""
import json
from pathlib import Path


def compute_clip_metrics(jsonl_path: Path, total_frames: int, id_switch_count: int, color_swap_count: int) -> dict:
    frames: dict[int, list[dict]] = {}
    for line in jsonl_path.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        frames.setdefault(r["frame"], []).append(r)

    n_both = n_one = n_zero = 0
    filter_violations = 0

    for frame_idx in range(total_frames):
        recs = frames.get(frame_idx, [])
        slots_present = {r["athlete_id"] for r in recs if r["athlete_id"] is not None}
        n = len(slots_present)
        if n >= 2:
            n_both += 1
        elif n == 1:
            n_one += 1
        else:
            n_zero += 1
        for r in recs:
            if r["athlete_id"] is not None and not r.get("passed_athlete_filter", True):
                filter_violations += 1

    def pct(n):
        return round(n / total_frames * 100, 1) if total_frames else 0.0

    return {
        "total_frames": total_frames,
        "pct_both_assigned": pct(n_both),
        "pct_one_assigned": pct(n_one),
        "pct_zero_assigned": pct(n_zero),
        "id_switch_count": id_switch_count,
        "color_swap_count": color_swap_count,
        # Should always be 0 - a detection failing the athlete filter must
        # never hold a slot. Non-zero here means that invariant broke.
        "filter_violations": filter_violations,
    }
