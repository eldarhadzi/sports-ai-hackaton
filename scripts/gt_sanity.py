"""Do the human labels agree with the measured belt/glove colors?

For every labeled frame, takes the detection matched to GT-AKA and to GT-AO and
compares their pixel red/blue evidence (color_evidence in the run's jsonl, the
same measurement the pipeline uses). A frame is flagged when the box labeled
AKA is measurably LESS red than the box labeled AO. Flags are for a human to
re-check - they can also be the measurement being wrong (clinch, glare).

    .venv/bin/python scripts/gt_sanity.py v5
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis.ground_truth import _gt_box, gt_path, iou, load_frame_detections, load_gt  # noqa: E402

MARGIN = 0.3   # in score units (R-B), per frame; weak evidence is not flagged

for clip in sorted(p.stem for p in (ROOT / "videos" / "input").glob("*.mp4")):
    gt = load_gt(gt_path(ROOT / "data", clip))
    run = sys.argv[1] if len(sys.argv) > 1 else "v5"
    frames = load_frame_detections(ROOT / "data" / run / f"{clip}.jsonl")
    flagged, checked = [], 0
    for key, labels in sorted(gt["frames"].items(), key=lambda kv: int(kv[0])):
        boxes = {s: _gt_box(labels.get(s)) for s in ("aka", "ao")}
        if None in boxes.values():
            continue
        ev = {}
        for s, b in boxes.items():
            best = max(frames.get(int(key), []), key=lambda r: iou(r["bbox"], b), default=None)
            if best is None or iou(best["bbox"], b) < 0.5:
                break
            ev[s] = best["color_evidence"]
        if len(ev) < 2:
            continue
        sc = {s: ev[s][0] - ev[s][1] for s in ev}
        checked += 1
        same_box = iou(boxes["aka"], boxes["ao"]) > 0.9
        if same_box or sc["aka"] < sc["ao"] - MARGIN:
            flagged.append((int(key), "SAME BOX for AKA and AO" if same_box else
                            f"AKA box R-B={sc['aka']:+.2f} < AO box R-B={sc['ao']:+.2f}"))
    print(f"{clip}: checked {checked}, flagged {len(flagged)}")
    for f, why in flagged:
        print(f"   f{f}: {why}")
