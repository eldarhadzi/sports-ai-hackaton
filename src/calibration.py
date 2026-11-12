"""Manual tatami calibration: click the 4 corners of the competition area in an
OpenCV window, save a homography per video, and reuse it on later runs.

No automatic tatami detection in V1 (by design).
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Corners must be clicked in this order (as seen in the frame):
# top-left, top-right, bottom-right, bottom-left.
CORNER_ORDER_HINT = "top-left, top-right, bottom-right, bottom-left"


class TatamiCalibration:
    def __init__(self, homography: np.ndarray):
        self.homography = homography

    def to_tatami(self, image_point: tuple[float, float]) -> tuple[float, float] | None:
        x, y = image_point
        pt = self.homography @ np.array([x, y, 1.0])
        if abs(pt[2]) < 1e-9:
            return None
        return (float(pt[0] / pt[2]), float(pt[1] / pt[2]))


def calibrate_from_click(
    frame: np.ndarray, out_path: Path, tatami_width_m: float, tatami_height_m: float
) -> TatamiCalibration:
    points: list[tuple[float, float]] = []
    window = f"Click 4 tatami corners ({CORNER_ORDER_HINT}), then press any key"
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_click)
    while len(points) < 4:
        if cv2.waitKey(50) != -1 and len(points) >= 4:
            break
    cv2.waitKey(0)
    cv2.destroyWindow(window)

    if len(points) != 4:
        raise RuntimeError(f"Calibration needs exactly 4 corner clicks, got {len(points)}")

    return save_calibration(points, out_path, tatami_width_m, tatami_height_m)


def save_calibration(
    image_points: list[tuple[float, float]],
    out_path: Path,
    tatami_width_m: float,
    tatami_height_m: float,
) -> TatamiCalibration:
    world_points = [
        [0.0, 0.0],
        [tatami_width_m, 0.0],
        [tatami_width_m, tatami_height_m],
        [0.0, tatami_height_m],
    ]
    homography, _ = cv2.findHomography(
        np.array(image_points, dtype=np.float32),
        np.array(world_points, dtype=np.float32),
    )
    if homography is None:
        raise RuntimeError("Could not compute homography from the given corners")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "image_points": image_points,
        "world_points": world_points,
        "tatami_width_m": tatami_width_m,
        "tatami_height_m": tatami_height_m,
        "homography": homography.tolist(),
    }, indent=2))
    logger.info("Saved tatami calibration to %s", out_path)
    return TatamiCalibration(homography)


def load_calibration(path: Path) -> TatamiCalibration | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return TatamiCalibration(np.array(data["homography"]))
