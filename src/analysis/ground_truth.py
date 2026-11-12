"""Human ground-truth labels (data/ground_truth/<clip>.json) and metrics that
score a run against them.

Self-consistency metrics (id switches, filter violations) can read 0 while the
labels are plainly wrong (AKA/AO inverted, a referee holding a slot). These
metrics compare against what a person actually saw.

Label file:
  {"clip": ..., "fps": ..., "frames": {"<frame>": {"aka": <box>, "ao": <box>}}}
where <box> is {"bbox": [x1,y1,x2,y2]} or "not_visible". A frame is only
scored when BOTH slots are labeled (otherwise a pipeline box matching nothing
could just be the unlabeled athlete).
"""
import json
from pathlib import Path

IOU_MATCH = 0.5


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def sample_frames(fps: float, total_frames: int) -> list[int]:
    """One frame per second of video."""
    n_seconds = int(total_frames / fps) + 1
    return [f for f in (int(round(s * fps)) for s in range(n_seconds)) if f < total_frames]


def gt_path(data_dir: Path, clip: str) -> Path:
    return Path(data_dir) / "ground_truth" / f"{clip}.json"


def load_gt(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save_gt(path: Path, gt: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gt, indent=1))


def load_frame_detections(jsonl_path: Path) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}
    for line in Path(jsonl_path).read_text().splitlines():
        if line:
            r = json.loads(line)
            frames.setdefault(r["frame"], []).append(r)
    return frames


def match_holders(slot_recs: list[dict], gt_boxes: dict) -> dict:
    """GT slot -> the pipeline slot-holder record matched to it, or None.
    One-to-one by best IoU (>= IOU_MATCH), highest-IoU pair first. Taking the
    first box over the threshold is ambiguous in a clinch, where both GT boxes
    overlap both pipeline boxes."""
    pairs = sorted(
        ((iou(r["bbox"], b), slot, i) for slot, b in gt_boxes.items() if b is not None
         for i, r in enumerate(slot_recs)),
        reverse=True,
    )
    out, used = {slot: None for slot in gt_boxes}, set()
    for score, slot, i in pairs:
        if score < IOU_MATCH:
            break
        if out[slot] is None and i not in used:
            out[slot] = slot_recs[i]
            used.add(i)
    return out


def _gt_box(label):
    return None if label is None or label == "not_visible" else label["bbox"]


def compute_gt_metrics(frames: dict[int, list[dict]], gt: dict) -> dict:
    """frames: jsonl records grouped by frame. A pipeline slot box 'matches' a
    GT box at IoU >= IOU_MATCH."""
    labeled = 0
    gt_visible = correct = swapped = unknown_identity = wrong_other = 0
    non_athlete_slot_frames = 0
    missed_athletes = 0
    non_athlete_frames = []
    missed_frames = []
    swapped_frames = []

    for key, labels in gt["frames"].items():
        if "aka" not in labels or "ao" not in labels:
            continue
        labeled += 1
        f = int(key)
        gt_boxes = {s: _gt_box(labels[s]) for s in ("aka", "ao")}
        recs = frames.get(f, [])
        slot_recs = [r for r in recs if r["athlete_id"] is not None]

        # Pipeline slot holders that match no GT athlete at all.
        for r in slot_recs:
            if not any(b is not None and iou(r["bbox"], b) >= IOU_MATCH for b in gt_boxes.values()):
                non_athlete_slot_frames += 1
                non_athlete_frames.append(f)

        holders = match_holders(slot_recs, gt_boxes)
        for slot, box in gt_boxes.items():
            if box is None:
                continue
            gt_visible += 1
            holder = holders[slot]
            if holder is None:
                missed_athletes += 1
                missed_frames.append(f)
            elif not holder.get("identity_known", True):
                unknown_identity += 1
            elif holder["athlete_id"] == slot:
                correct += 1
            else:
                swapped += 1
                swapped_frames.append(f)

    return {
        "labeled_frames": labeled,
        "gt_athlete_instances": gt_visible,
        "identity_accuracy_pct": round(correct / gt_visible * 100, 1) if gt_visible else None,
        "identity_correct": correct,
        "identity_swapped": swapped,
        "identity_unknown": unknown_identity,
        "missed_athletes": missed_athletes,
        "non_athlete_slot_frames": non_athlete_slot_frames,
        "swapped_frames": sorted(set(swapped_frames)),
        "non_athlete_frames": sorted(set(non_athlete_frames)),
        "missed_frames": sorted(set(missed_frames)),
    }


def gt_cases(frames: dict[int, list[dict]], gt: dict) -> list[dict]:
    """Every scored frame's problems, one dict per case, with the evidence
    needed to say WHY: kind in {missed, nonathlete, swapped, unknown}.
    For 'missed' it reports the best-overlapping detection of ANY kind (so
    'no detection' vs 'box mismatch' vs 'rejected by filter' vs 'passed the
    filter but unassigned' are distinguishable)."""
    cases = []
    for key, labels in sorted(gt["frames"].items(), key=lambda kv: int(kv[0])):
        if "aka" not in labels or "ao" not in labels:
            continue
        f = int(key)
        gt_boxes = {s: _gt_box(labels[s]) for s in ("aka", "ao")}
        recs = frames.get(f, [])
        slot_recs = [r for r in recs if r["athlete_id"] is not None]
        both_iou = (
            iou(gt_boxes["aka"], gt_boxes["ao"]) if gt_boxes["aka"] and gt_boxes["ao"] else None
        )

        for r in slot_recs:
            if not any(b is not None and iou(r["bbox"], b) >= IOU_MATCH for b in gt_boxes.values()):
                best = max(((s, iou(r["bbox"], b)) for s, b in gt_boxes.items() if b), key=lambda x: x[1], default=(None, 0))
                cases.append({"frame": f, "kind": "nonathlete", "slot": r["athlete_id"], "tracker": r["tracker_id"],
                              "bbox": r["bbox"], "nearest_gt": best[0], "nearest_iou": round(best[1], 2),
                              "passed_filter": r["passed_athlete_filter"], "reject": r["reject_reason"],
                              "gt_pair_iou": both_iou})

        holders = match_holders(slot_recs, gt_boxes)
        for slot, box in gt_boxes.items():
            if box is None:
                continue
            holder = holders[slot]
            if holder is None:
                best = max(recs, key=lambda r: iou(r["bbox"], box), default=None)
                b_iou = iou(best["bbox"], box) if best else 0.0
                cases.append({
                    "frame": f, "kind": "missed", "slot": slot, "gt_bbox": box, "gt_pair_iou": both_iou,
                    "best_iou": round(b_iou, 2),
                    "best_tracker": best["tracker_id"] if best and b_iou > 0.1 else None,
                    "best_passed_filter": best["passed_athlete_filter"] if best and b_iou > 0.1 else None,
                    "best_reject": best["reject_reason"] if best and b_iou > 0.1 else None,
                    "best_slot": best["athlete_id"] if best and b_iou > 0.1 else None,
                    "gt_box_aspect": round((box[2] - box[0]) / max(box[3] - box[1], 1), 2),
                })
            elif not holder.get("identity_known", True):
                cases.append({"frame": f, "kind": "unknown", "slot": slot, "tracker": holder["tracker_id"],
                              "gt_pair_iou": both_iou})
            elif holder["athlete_id"] != slot:
                cases.append({"frame": f, "kind": "swapped", "slot": slot, "tracker": holder["tracker_id"],
                              "pipeline_slot": holder["athlete_id"], "gt_pair_iou": both_iou})
    return cases
