"""Drawing: boxes, pose skeletons, movement trails; and per-athlete heatmaps."""
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from athletes import AKA, AO

logger = logging.getLogger(__name__)

# COCO-17 skeleton edges (keypoint index pairs).
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# BGR colors matching kumite belt convention: AKA = red, AO = blue.
ATHLETE_COLORS = {
    AKA: (0, 0, 255),
    AO: (255, 0, 0),
    None: (160, 160, 160),
}
SLOT_LABELS = {AKA: "AKA", AO: "AO"}

KEYPOINT_CONF_THRESHOLD = 0.3


def slot_style(selector, slot):
    """(label, BGR color). A slot whose holder has no color evidence is shown
    as "?" in neutral white instead of a position-based AKA/AO guess."""
    if slot is not None and not selector.identity_known.get(slot, False):
        return "?", (255, 255, 255)
    return SLOT_LABELS.get(slot), ATHLETE_COLORS.get(slot, ATHLETE_COLORS[None])


def draw_frame(frame, detections, selector, trails, visible_slots, viz_config, debug: bool = False) -> np.ndarray:
    """trails: {slot: [(timestamp, point-or-None), ...]}. visible_slots: the
    set of slots with a live detection THIS frame - a slot currently missing
    draws no trail at all, rather than a stale tail with nothing at its head.
    Only the two athlete slots are drawn; grey boxes for everyone else (and
    their skeletons) appear only when debug=True (--debug)."""
    annotated = frame.copy()

    for det in detections:
        slot = selector.tracker_to_slot(det.tracker_id)
        if slot is None and not debug:
            continue
        slot_label, color = slot_style(selector, slot)

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = slot_label if slot else f"id {det.tracker_id}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if viz_config.get("draw_skeleton", True):
            _draw_skeleton(annotated, det.keypoints, color)

    if viz_config.get("draw_trail", True):
        for slot, entries in trails.items():
            if slot not in visible_slots:
                continue
            _, color = slot_style(selector, slot)
            pts = [(int(p[0]), int(p[1])) if p is not None else None for _, p in entries]
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(annotated, pts[i - 1], pts[i], color, 2)

    return annotated


def save_color_debug_strip(frame, bbox, samples, voted_color, out_path: Path, patch_display_size: int = 72):
    """Saves one debug image showing every patch classify_belt_color() sampled
    this frame (belt, left wrist, right wrist), upscaled and labeled with its
    role and read color, so a wrong read (e.g. picking up the tatami instead
    of the belt) is visible directly rather than inferred from logs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = []
    for sample in samples:
        tile = np.full((patch_display_size, patch_display_size, 3), 40, dtype=np.uint8)
        rect = sample["rect"]
        if rect is not None:
            x1, y1, x2, y2 = rect
            patch = frame[y1:y2, x1:x2]
            if patch.size > 0:
                tile = cv2.resize(patch, (patch_display_size, patch_display_size), interpolation=cv2.INTER_NEAREST)
        label_bar = np.full((20, patch_display_size, 3), 0, dtype=np.uint8)
        cv2.putText(label_bar, sample["label"][:10], (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        color_bar = np.full((16, patch_display_size, 3), 0, dtype=np.uint8)
        text_color = {"red": (0, 0, 255), "blue": (255, 0, 0)}.get(sample["color"], (150, 150, 150))
        cv2.putText(color_bar, f"{sample['color'][:7]} {sample.get('red', 0):.1f}/{sample.get('blue', 0):.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
        tiles.append(np.vstack([label_bar, tile, color_bar]))

    strip = np.hstack(tiles) if tiles else np.zeros((1, 1, 3), dtype=np.uint8)
    header = np.full((20, strip.shape[1], 3), 0, dtype=np.uint8)
    header_color = {"red": (0, 0, 255), "blue": (255, 0, 0)}.get(voted_color, (150, 150, 150))
    cv2.putText(header, f"voted: {voted_color}", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, header_color, 1)
    strip = np.vstack([header, strip])

    cv2.imwrite(str(out_path), strip)


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
