"""Thin wrapper around Ultralytics YOLO pose + built-in tracking.

One model, one pass per frame: bbox + confidence + 17 COCO keypoints + track id.
Kept behind this interface so the underlying model/library can be swapped later
(e.g. for a permissively-licensed alternative to AGPL Ultralytics YOLO).
"""
from dataclasses import dataclass
import logging

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


@dataclass
class Detection:
    tracker_id: int | None
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    keypoints: list[tuple[float, float, float]]  # 17 x (x, y, conf)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PoseTracker:
    """Wraps model.track(..., persist=True) so callers never touch Ultralytics directly."""

    def __init__(self, model_name: str, device: str, tracker_cfg: str = "bytetrack.yaml", conf: float = 0.25):
        self.device = resolve_device(device)
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.model = YOLO(model_name)
        self._mps_fallback_done = False
        logger.info(
            "PoseTracker: model=%s device=%s tracker=%s conf=%.2f", model_name, self.device, tracker_cfg, conf,
        )

    def track(self, frame) -> list[Detection]:
        try:
            results = self.model.track(
                frame, persist=True, device=self.device, tracker=self.tracker_cfg,
                conf=self.conf, verbose=False,
            )
        except Exception as exc:
            if self.device == "mps" and not self._mps_fallback_done:
                logger.warning(
                    "MPS pose tracking failed (%s); falling back to CPU for the rest of this run.", exc
                )
                self.device = "cpu"
                self._mps_fallback_done = True
                results = self.model.track(
                    frame, persist=True, device=self.device, tracker=self.tracker_cfg,
                    conf=self.conf, verbose=False,
                )
            else:
                raise
        return self._parse(results)

    @staticmethod
    def _parse(results) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or r.keypoints is None:
            return detections

        boxes = r.boxes
        kpts = r.keypoints
        ids = boxes.id

        for i in range(len(boxes)):
            tracker_id = int(ids[i].item()) if ids is not None else None
            xyxy = tuple(boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i].item())
            kp_xy = kpts.xy[i].tolist()
            kp_conf = kpts.conf[i].tolist() if kpts.conf is not None else [1.0] * len(kp_xy)
            keypoints = [(x, y, c) for (x, y), c in zip(kp_xy, kp_conf)]
            detections.append(Detection(tracker_id, xyxy, conf, keypoints))

        return detections
