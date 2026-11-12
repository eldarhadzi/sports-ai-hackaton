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
from analysis.ground_truth import gt_path, load_frame_detections, load_gt, sanity_flags  # noqa: E402

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "v5"
    for clip in sorted(p.stem for p in (ROOT / "videos" / "input").glob("*.mp4")):
        gt = load_gt(gt_path(ROOT / "data", clip))
        checked, flagged = sanity_flags(load_frame_detections(ROOT / "data" / run / f"{clip}.jsonl"), gt)
        print(f"{clip}: checked {checked}, flagged {len(flagged)}")
        for f, why in flagged:
            print(f"   f{f}: {why}")
