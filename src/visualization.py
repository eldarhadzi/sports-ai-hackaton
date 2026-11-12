"""Drawing: boxes, pose skeletons, movement trails; and per-athlete heatmaps."""
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# COCO-17 skeleton edges (keypoint index pairs).
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# BGR colors matching kumite belt convention: athlete_1 = aka (red), athlete_2 = ao (blue).
ATHLETE_COLORS = {
    "athlete_1": (0, 0, 255),
    "athlete_2": (255, 0, 0),
    None: (160, 160, 160),
}

KEYPOINT_CONF_THRESHOLD = 0.3


def draw_frame(frame, detections, selector, trails, viz_config) -> np.ndarray:
    annotated = frame.copy()

    for det in detections:
        athlete_id = selector.tracker_to_athlete(det.tracker_id)
        color = ATHLETE_COLORS.get(athlete_id, ATHLETE_COLORS[None])

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = athlete_id if athlete_id else f"id {det.tracker_id}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if viz_config.get("draw_skeleton", True):
            _draw_skeleton(annotated, det.keypoints, color)

    if viz_config.get("draw_trail", True):
        for athlete_id, points in trails.items():
            color = ATHLETE_COLORS[athlete_id]
            pts = [(int(x), int(y)) for x, y in points]
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], color, 2)

    return annotated


def _draw_skeleton(frame, keypoints, color):
    for i, j in SKELETON_EDGES:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci >= KEYPOINT_CONF_THRESHOLD and cj >= KEYPOINT_CONF_THRESHOLD:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2)
    for x, y, c in keypoints:
        if c >= KEYPOINT_CONF_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def render_heatmap(
    positions: list[tuple[float, float, float]],
    out_path: Path,
    calibration=None,
    frame_for_overlay: np.ndarray | None = None,
    tatami_size: tuple[float, float] | None = None,
):
    """positions: list of (timestamp, x, y). x/y are meters if calibration is
    not None, else pixels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not positions:
        logger.warning("No positions recorded, skipping heatmap: %s", out_path)
        return

    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]

    fig, ax = plt.subplots(figsize=(6, 6))

    if calibration is not None and tatami_size is not None:
        w, h = tatami_size
        heat, _, _ = np.histogram2d(xs, ys, bins=40, range=[[0, w], [0, h]])
        ax.imshow(heat.T, origin="lower", extent=[0, w, 0, h], cmap="hot", aspect="equal")
        ax.set_xlabel("meters (approx)")
        ax.set_ylabel("meters (approx)")
        ax.set_title("Tatami heatmap (approx meters)")
    else:
        if frame_for_overlay is not None:
            rgb = cv2.cvtColor(frame_for_overlay, cv2.COLOR_BGR2RGB)
            h_img, w_img = frame_for_overlay.shape[:2]
            ax.imshow(rgb, extent=[0, w_img, h_img, 0])
        heat, _, _ = np.histogram2d(xs, ys, bins=40)
        ax.imshow(
            heat.T, origin="upper", extent=[min(xs), max(xs), max(ys), min(ys)],
            cmap="hot", alpha=0.6, aspect="auto",
        )
        ax.set_xlabel("pixels")
        ax.set_ylabel("pixels")
        ax.set_title("Heatmap (image space, no calibration)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap to %s", out_path)
