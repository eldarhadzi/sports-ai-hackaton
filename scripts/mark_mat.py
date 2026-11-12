"""Mark the mat the bout is on: click its 4 corners on one frame.

    .venv/bin/python scripts/mark_mat.py --clip club_02                 # starts at frame 0
    .venv/bin/python scripts/mark_mat.py --clip club_02 --frame 1802
    .venv/bin/python scripts/mark_mat.py --clip club_02 --frame 1802 --corners "255,410,885,410,985,540,175,540"

Saved to data/mats/<clip>.json (corners + the frame they were clicked on). main.py
then lets only people whose ground point is inside that mat (plus a margin) hold a
slot, following the camera if it moves. Clips without a file are unaffected.

Click the OUTER corners of the competition area (the red border included), in any
order. A corner hidden behind a box or a person: click where it would be.
  left-click  add a corner (up to 4)          u / right-click  remove the last one
  n / p       1 s later / earlier (pick a frame where the corners are visible; resets the clicks)
  ENTER       save (needs 4)                  q  quit without saving
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from mat_region import mat_path, order_corners, save_mat  # noqa: E402
from label_ground_truth import fit_scale  # noqa: E402

WIN = "mark mat corners"


def draw(frame, points, scale, frame_idx, fps):
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    pts = order_corners(points) if len(points) == 4 else points
    for x, y in pts:
        cv2.circle(img, (int(x * scale), int(y * scale)), 5, (0, 255, 255), -1)
    if len(points) == 4:
        cv2.polylines(img, [np.int32([[x * scale, y * scale] for x, y in pts])], True, (0, 255, 255), 2)
    bar = f"f{frame_idx} t={frame_idx / fps:.0f}s  corners {len(points)}/4  (ENTER save, u undo, n/p next/prev second, q quit)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, bar, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--corners", default=None, help="x1,y1,...,x4,y4 in frame pixels; skips the window")
    ap.add_argument("--scale", default="fit", help="'fit' (default) or a number")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(ROOT / "videos" / "input" / f"{args.clip}.mp4"))
    if not cap.isOpened():
        sys.exit(f"cannot open videos/input/{args.clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = mat_path(ROOT / "data", args.clip)

    if args.corners:
        v = [float(s) for s in args.corners.split(",")]
        if len(v) != 8:
            sys.exit("--corners needs 8 numbers")
        data = save_mat(out, args.clip, args.frame, [(v[i], v[i + 1]) for i in range(0, 8, 2)], (w, h))
        print(f"saved {out}: frame {data['frame']} corners {data['corners']}")
        return

    scale = fit_scale(w, h) if args.scale == "fit" else float(args.scale)
    frame_idx, points = min(max(args.frame, 0), total - 1), []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            sys.exit(f"cannot read frame {frame_idx}")
        while True:
            cv2.imshow(WIN, draw(frame, points, scale, frame_idx, fps))
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            ch = chr(k)
            if ch in "np":
                frame_idx = min(max(frame_idx + int(round(fps)) * (1 if ch == "n" else -1), 0), total - 1)
                points.clear()
                break
            if ch == "u" and points:
                points.pop()
            elif k in (13, 10) and len(points) == 4:
                data = save_mat(out, args.clip, frame_idx, points, (w, h))
                print(f"saved {out}: frame {data['frame']} corners {[tuple(round(c) for c in p) for p in data['corners']]}")
                cv2.destroyAllWindows()
                return
            elif ch == "q":
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    main()
