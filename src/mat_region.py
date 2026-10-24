"""Active mat region: only people standing on the mat the bout is on can hold a slot.

Not a metric calibration (no meters, no stats): the 4 clicked corners only define
a region. They are clicked once on a reference frame (scripts/mark_mat.py, saved
to data/mats/<clip>.json). A handheld camera drifts by up to ~15% of the frame
width over a bout, so every frame is registered to that reference frame
(ORB features + RANSAC homography) and the polygon moves with the camera.

The inside test maps a ground point into normalized mat coordinates (the clicked
quad -> unit square), so the margin means the same fraction of the mat at the near
and the far edge: margin 0.1 on an 8 m mat ~ 0.8 m outside the edge.
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
# Frame-to-reference registration: features at half resolution, and a frame whose
# registration has fewer RANSAC inliers than this keeps the last good transform.
REG_SCALE = 0.5
REG_FEATURES = 1500
REG_MIN_INLIERS = 40


def mat_path(data_dir: Path, clip: str) -> Path:
    return Path(data_dir) / "mats" / f"{clip}.json"


def order_corners(points) -> list[tuple[float, float]]:
    """Any 4 clicks -> top-left, top-right, bottom-right, bottom-left (as seen)."""
    pts = sorted(((float(x), float(y)) for x, y in points), key=lambda p: p[1])
    top, bottom = sorted(pts[:2]), sorted(pts[2:])
    return [top[0], top[1], bottom[1], bottom[0]]


def save_mat(path: Path, clip: str, frame: int, corners, frame_size: tuple[int, int]) -> dict:
    data = {"clip": clip, "frame": int(frame), "corners": order_corners(corners), "frame_size": list(frame_size)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def load_mat(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


class FrameRegistrar:
    """Homography from each frame to the reference frame."""

    def __init__(self, ref_frame: np.ndarray):
        self.orb = cv2.ORB_create(REG_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp, self.ref_des = self._features(ref_frame)
        self.last_good = np.eye(3)
        self.failed_frames = 0

    def _features(self, frame):
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=REG_SCALE, fy=REG_SCALE), cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def to_ref(self, frame: np.ndarray) -> np.ndarray:
        kp, des = self._features(frame)
        H = None
        if des is not None and self.ref_des is not None and len(kp) >= 8:
            matches = sorted(self.matcher.match(des, self.ref_des), key=lambda m: m.distance)[:500]
            if len(matches) >= REG_MIN_INLIERS:
                src = np.float32([kp[m.queryIdx].pt for m in matches]) / REG_SCALE
                dst = np.float32([self.ref_kp[m.trainIdx].pt for m in matches]) / REG_SCALE
                H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if H is None or int(inliers.sum()) < REG_MIN_INLIERS:
                    H = None
        if H is None:
            self.failed_frames += 1
            return self.last_good
        self.last_good = H
        return H


class MatRegion:
    def __init__(self, corners, margin: float):
        self.corners = np.float32(corners)
        self.margin = margin
        self.ref_to_unit = cv2.getPerspectiveTransform(self.corners, UNIT_SQUARE)
        self.frame_to_ref: list[np.ndarray] = []

    def add_frame(self, frame_to_ref: np.ndarray):
        self.frame_to_ref.append(frame_to_ref)

    def mat_coords(self, frame_idx: int, point) -> tuple[float, float] | None:
        H = self.ref_to_unit @ self.frame_to_ref[frame_idx]
        p = H @ np.array([point[0], point[1], 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def inside(self, frame_idx: int, point) -> bool:
        uv = self.mat_coords(frame_idx, point)
        m = self.margin
        return uv is not None and -m <= uv[0] <= 1 + m and -m <= uv[1] <= 1 + m

    def on_mat(self, frame_idx: int, point, bbox, frame_h: int) -> bool:
        """inside(), except for a person cut off by the bottom of the frame while the
        whole mat (margin included) is above that edge: their feet are below the
        frame, so they stand in front of the mat - pose keypoints guessed for the
        hidden legs can otherwise land on the mat (a passing spectator, a coach)."""
        if bbox[3] >= frame_h - 2 and self.margin_polygon(frame_idx)[:, 1].max() < frame_h - 2:
            return False
        return point is not None and self.inside(frame_idx, point)

    def margin_polygon(self, frame_idx: int) -> np.ndarray:
        m = self.margin
        unit = np.float32([[-m, -m], [1 + m, -m], [1 + m, 1 + m], [-m, 1 + m]]).reshape(-1, 1, 2)
        H = np.linalg.inv(self.ref_to_unit @ self.frame_to_ref[frame_idx])
        return cv2.perspectiveTransform(unit, H).reshape(-1, 2)

    def polygon(self, frame_idx: int) -> np.ndarray:
        """The clicked corners as they appear in frame `frame_idx` (for drawing)."""
        return cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), np.linalg.inv(self.frame_to_ref[frame_idx])).reshape(-1, 2)
