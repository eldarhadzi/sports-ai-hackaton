"""Score runs against ground truth.   .venv/bin/python scripts/gt_report.py v4 [v5 ...]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis.ground_truth import compute_gt_metrics, gt_path, load_frame_detections, load_gt  # noqa: E402

CLIPS = sorted(p.stem for p in (ROOT / "videos" / "input").glob("*.mp4"))

for run in sys.argv[1:] or ["v5"]:
    print(f"== {run}")
    print(f"{'clip':<32}{'labeled':>8}{'id_acc%':>9}{'swapped':>9}{'unknown':>9}{'nonathl':>9}{'missed':>8}")
    for c in CLIPS:
        gt, jl = load_gt(gt_path(ROOT / "data", c)), ROOT / "data" / run / f"{c}.jsonl"
        if gt is None or not jl.exists():
            print(f"{c:<32}  (no ground truth labels yet)" if gt is None else f"{c:<32}  (no {jl})")
            continue
        m = compute_gt_metrics(load_frame_detections(jl), gt)
        print(f"{c:<32}{m['labeled_frames']:>8}{str(m['identity_accuracy_pct']):>9}{m['identity_swapped']:>9}"
              f"{m['identity_unknown']:>9}{m['non_athlete_slot_frames']:>9}{m['missed_athletes']:>8}")
