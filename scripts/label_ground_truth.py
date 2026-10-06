"""Ground-truth labeling tool: 1 frame per second, all detected boxes numbered.

    .venv/bin/python scripts/label_ground_truth.py --clip kumite_06_medium_kamera_prati
    .venv/bin/python scripts/label_ground_truth.py --clip all          # every clip in videos/input

Boxes come from an earlier run's data/<run>/<clip>.jsonl (default v4). Labels are
saved to data/ground_truth/<clip>.json after every frame; re-running resumes
where you stopped (frames already labeled are skipped; use --relabel to revisit).

Per frame you must label BOTH athletes (or mark them not visible):
  mouse   left-click a box   -> that box is AKA (red belt/gloves)
          right-click a box  -> that box is AO  (blue belt/gloves)
  keys    type box number then  a = AKA   o = AO
          r = AKA not visible   b = AO not visible
          ENTER/SPACE = save + next (needs both set)    s = skip frame (unlabeled)
          BACKSPACE/h = previous frame      q = save + quit
Use pause/zoom in your head: the frame is shown at 2x. Judge by belt/glove color,
not by position or by what the pipeline thinks.
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis.ground_truth import (  # noqa: E402
    gt_path, load_frame_detections, load_gt, sample_frames, save_gt,
)

SCALE = 2
WIN = "ground truth labeling"


def numbered_boxes(dets: list[dict]) -> list[dict]:
    """Stable numbering: left-to-right by box center, 1-based."""
    ordered = sorted(dets, key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)
    return [{"n": i + 1, "bbox": d["bbox"]} for i, d in enumerate(ordered)]


def box_at(boxes: list[dict], x: float, y: float) -> dict | None:
    """Smallest box containing the point (original-frame coords)."""
    hits = [b for b in boxes if b["bbox"][0] <= x <= b["bbox"][2] and b["bbox"][1] <= y <= b["bbox"][3]]
    return min(hits, key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])) if hits else None


def label_value(box: dict | None):
    return "not_visible" if box is None else {"bbox": [round(v, 1) for v in box["bbox"]]}


def render(frame, boxes, sel, frame_idx, t, status):
    img = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for b in boxes:
        x1, y1, x2, y2 = (int(v * SCALE) for v in b["bbox"])
        chosen = next((s for s, v in sel.items() if v not in (None, "not_visible") and v["bbox"] == [round(c, 1) for c in b["bbox"]]), None)
        color = {"aka": (0, 0, 255), "ao": (255, 0, 0)}.get(chosen, (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if chosen else 1)
        tag = f"{b['n']}" + (f" {chosen.upper()}" if chosen else "")
        cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + 12 * len(tag) + 6, max(22, y1)), color, -1)
        cv2.putText(img, tag, (x1 + 3, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    def show(v):
        return "?" if v is None else ("not visible" if v == "not_visible" else "set")
    bar = f"f{frame_idx} t={t:.0f}s  AKA:{show(sel.get('aka'))}  AO:{show(sel.get('ao'))}  {status}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, bar, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def label_clip(clip: str, run: str, relabel: bool):
    video = ROOT / "videos" / "input" / f"{clip}.mp4"
    jsonl = ROOT / "data" / run / f"{clip}.jsonl"
    if not jsonl.exists():
        sys.exit(f"missing detections {jsonl} - run the pipeline into data/{run} first")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dets = load_frame_detections(jsonl)
    path = gt_path(ROOT / "data", clip)
    gt = load_gt(path) or {"clip": clip, "fps": fps, "frames": {}}
    todo = sample_frames(fps, total)
    i = 0
    state = {"sel": {}, "buf": "", "status": "", "boxes": []}

    def on_mouse(event, x, y, flags, _):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        b = box_at(state["boxes"], x / SCALE, y / SCALE)
        if b is None:
            return
        state["sel"]["aka" if event == cv2.EVENT_LBUTTONDOWN else "ao"] = label_value(b)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while 0 <= i < len(todo):
        f = todo[i]
        if str(f) in gt["frames"] and not relabel and not state.get("back"):
            i += 1
            continue
        state["back"] = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            i += 1
            continue
        state["boxes"] = numbered_boxes(dets.get(f, []))
        state["sel"] = dict(gt["frames"].get(str(f), {}))
        state["buf"] = ""
        while True:
            status = f"[{i + 1}/{len(todo)}] typed:{state['buf']}  {state['status']}"
            cv2.imshow(WIN, render(frame, state["boxes"], state["sel"], f, f / fps, status))
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            ch = chr(k)
            state["status"] = ""
            if ch.isdigit():
                state["buf"] = (state["buf"] + ch)[-2:]
            elif ch in "ao":
                n = int(state["buf"]) if state["buf"] else -1
                b = next((b for b in state["boxes"] if b["n"] == n), None)
                if b is None:
                    state["status"] = "no such box number"
                else:
                    state["sel"]["aka" if ch == "a" else "ao"] = label_value(b)
                state["buf"] = ""
            elif ch == "r":
                state["sel"]["aka"] = "not_visible"
            elif ch == "b":
                state["sel"]["ao"] = "not_visible"
            elif k in (13, 32, 10):
                if "aka" in state["sel"] and "ao" in state["sel"]:
                    gt["frames"][str(f)] = state["sel"]
                    save_gt(path, gt)
                    i += 1
                    break
                state["status"] = "label BOTH AKA and AO (or r / b for not visible), or s to skip"
            elif ch == "s":
                i += 1
                break
            elif k in (8, 127) or ch == "h":
                i = max(0, i - 1)
                state["back"] = True
                break
            elif ch == "q":
                save_gt(path, gt)
                cv2.destroyAllWindows()
                return False
    save_gt(path, gt)
    print(f"{clip}: {len(gt['frames'])} frames labeled -> {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="clip stem, or 'all'")
    ap.add_argument("--run", default="v4", help="data/<run>/ holding the detections jsonl")
    ap.add_argument("--relabel", action="store_true")
    args = ap.parse_args()
    clips = sorted(p.stem for p in (ROOT / "videos" / "input").glob("*.mp4")) if args.clip == "all" else [args.clip]
    for c in clips:
        if not label_clip(c, args.run, args.relabel):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
