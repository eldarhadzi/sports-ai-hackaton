"""List every missed / nonathlete / swapped / unknown case against ground truth
and save a debug image for each.

    .venv/bin/python scripts/gt_cases.py v5            -> data/v5/gt_cases/*.png + gt_cases.json

Image legend: thick RED/BLUE solid = ground-truth AKA/AO box; YELLOW = a box the
pipeline gave a slot (label = slot + tracker id); thin GREY = every other
detection (tracker id, and the filter reject reason if any). Top bar = case.
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis.ground_truth import gt_cases, gt_path, load_frame_detections, load_gt  # noqa: E402

KINDS = ("missed", "nonathlete", "swapped", "unknown")


def draw(frame, recs, gt_labels, case):
    img = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for r in recs:
        x1, y1, x2, y2 = (int(v * 2) for v in r["bbox"])
        if r["athlete_id"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            txt = f"P:{r['athlete_id']}#{r['tracker_id']}" + ("" if r.get("identity_known", True) else "?")
            cv2.putText(img, txt, (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)
            txt = f"#{r['tracker_id']}" + (f" {r['reject_reason'].split(' ')[0]}" if r["reject_reason"] else "")
            cv2.putText(img, txt, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)
    for slot, color in (("aka", (0, 0, 255)), ("ao", (255, 0, 0))):
        lab = gt_labels.get(slot)
        if lab and lab != "not_visible":
            x1, y1, x2, y2 = (int(v * 2) for v in lab["bbox"])
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 3)
            cv2.putText(img, f"GT-{slot.upper()}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, case, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "v5"
    out = ROOT / "data" / run / "gt_cases"
    out.mkdir(parents=True, exist_ok=True)
    all_cases = {}
    for video in sorted((ROOT / "videos" / "input").glob("*.mp4")):
        clip = video.stem
        gt = load_gt(gt_path(ROOT / "data", clip))
        jl = ROOT / "data" / run / f"{clip}.jsonl"
        if gt is None or not jl.exists():
            continue
        frames = load_frame_detections(jl)
        cases = gt_cases(frames, gt)
        all_cases[clip] = cases
        cap = cv2.VideoCapture(str(video))
        for c in cases:
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame"])
            ok, fr = cap.read()
            if not ok:
                continue
            desc = f"{clip[:9]} f{c['frame']} {c['kind']} {c['slot']}"
            c["image"] = f"{clip}_f{c['frame']:05d}_{c['kind']}_{c['slot']}.png"
            cv2.imwrite(str(out / c["image"]), draw(fr, frames.get(c["frame"], []), gt["frames"][str(c["frame"])], desc))
    (out / "gt_cases.json").write_text(json.dumps(all_cases, indent=1))
    for clip, cases in all_cases.items():
        print(f"== {clip}")
        for c in cases:
            extra = {k: v for k, v in c.items() if k not in ("frame", "kind", "slot", "image", "bbox", "gt_bbox")}
            print(f"  f{c['frame']:<5} {c['kind']:<10} {c['slot']:<4} {extra}")


if __name__ == "__main__":
    main()
