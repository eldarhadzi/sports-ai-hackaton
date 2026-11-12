import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analysis.ground_truth import compute_gt_metrics, iou, load_gt, sample_frames, save_gt  # noqa: E402
from label_ground_truth import box_at, numbered_boxes  # noqa: E402

A = [10, 10, 50, 100]
B = [200, 10, 240, 100]
OFF = [400, 10, 440, 100]


def rec(frame, slot, bbox, known=True):
    return {"frame": frame, "athlete_id": slot, "bbox": bbox, "identity_known": known}


def gt(**frames):
    return {"frames": {k[1:]: {s: ({"bbox": b} if b else "not_visible") for s, b in v.items()} for k, v in frames.items()}}


def test_sample_frames_one_per_second():
    assert sample_frames(25.0, 60) == [0, 25, 50]


def test_iou_and_hit_test():
    assert iou(A, A) == 1.0 and iou(A, B) == 0.0
    boxes = numbered_boxes([{"bbox": B}, {"bbox": A}])
    assert [b["n"] for b in boxes] == [1, 2] and boxes[0]["bbox"] == A
    assert box_at(boxes, 20, 20)["n"] == 1 and box_at(boxes, 300, 300) is None


def test_metrics_correct_swapped_missed_nonathlete(tmp_path):
    g = gt(f0={"aka": A, "ao": B}, f1={"aka": A, "ao": B}, f2={"aka": A, "ao": None}, f3={"aka": A, "ao": B})
    frames = {
        0: [rec(0, "aka", A), rec(0, "ao", B)],            # both correct
        1: [rec(1, "aka", B), rec(1, "ao", A)],            # swapped
        2: [rec(2, "aka", A), rec(2, "ao", OFF)],          # referee holds ao
        3: [rec(3, "aka", A)],                             # ao missed
    }
    m = compute_gt_metrics(frames, g)
    assert m["labeled_frames"] == 4 and m["gt_athlete_instances"] == 7
    assert m["identity_correct"] == 4 and m["identity_swapped"] == 2
    assert m["non_athlete_slot_frames"] == 1 and m["non_athlete_frames"] == [2]
    assert m["missed_athletes"] == 1 and m["missed_frames"] == [3]
    assert m["identity_accuracy_pct"] == round(4 / 7 * 100, 1)
    p = tmp_path / "x.json"
    save_gt(p, g)
    assert load_gt(p) == g


def test_unknown_identity_is_not_correct_and_half_labeled_skipped():
    g = {"frames": {"0": {"aka": {"bbox": A}, "ao": {"bbox": B}}, "1": {"aka": {"bbox": A}}}}
    m = compute_gt_metrics({0: [rec(0, "aka", A, known=False), rec(0, "ao", B)]}, g)
    assert m["labeled_frames"] == 1 and m["identity_unknown"] == 1 and m["identity_correct"] == 1


def test_clinch_matching_is_one_to_one_best_iou():
    from analysis.ground_truth import match_holders
    gt_a, gt_o = [100, 0, 200, 200], [120, 0, 220, 200]            # overlapping GT boxes (clinch)
    recs = [rec(0, "aka", gt_a), rec(0, "ao", gt_o)]
    h = match_holders(recs, {"aka": gt_a, "ao": gt_o})
    assert h["aka"]["athlete_id"] == "aka" and h["ao"]["athlete_id"] == "ao"
