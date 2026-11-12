"""Choosing the two athletes among everyone visible (referee, judges, coaches,
audience), and keeping a stable Athlete 1 / Athlete 2 identity across tracker-ID
switches (which happen after occlusions/crossings).
"""
from dataclasses import dataclass, field
import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ATHLETE_1 = "athlete_1"
ATHLETE_2 = "athlete_2"


@dataclass
class TrackStats:
    tracker_id: int
    frames_seen: int = 0
    total_area: float = 0.0
    last_ground_point: tuple[float, float] | None = None
    last_seen_frame: int = -1

    @property
    def avg_area(self) -> float:
        return self.total_area / self.frames_seen if self.frames_seen else 0.0


def _dist(a, b) -> float:
    return math.dist(a, b)


class AthleteSelector:
    def __init__(
        self,
        min_track_len: int,
        min_bbox_area_frac: float,
        reassoc_max_dist_frac: float,
        frame_width: int,
        frame_height: int,
    ):
        self.min_track_len = min_track_len
        self.min_bbox_area = min_bbox_area_frac * frame_width * frame_height
        self.reassoc_max_dist = reassoc_max_dist_frac * math.hypot(frame_width, frame_height)

        self.tracks: dict[int, TrackStats] = {}
        self.athlete_to_tracker: dict[str, int | None] = {ATHLETE_1: None, ATHLETE_2: None}
        self.athlete_last_point: dict[str, tuple[float, float]] = {}

        self.id_switch_count = 0
        self.reassociation_log: list[dict] = []

    def update_track_stats(self, frame_idx: int, detections, ground_points):
        for det, gp in zip(detections, ground_points):
            if det.tracker_id is None:
                continue
            area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
            stats = self.tracks.setdefault(det.tracker_id, TrackStats(det.tracker_id))
            stats.frames_seen += 1
            stats.total_area += area
            stats.last_seen_frame = frame_idx
            if gp is not None:
                stats.last_ground_point = gp

    def _candidate_tracks(self, frame_idx: int, tatami_mask_fn) -> list[TrackStats]:
        # Must be visible THIS frame - otherwise a track that disappeared
        # (occlusion, left frame) but was large/persistent while it lasted
        # would keep winning selection forever over tracks that are actually
        # present now.
        candidates = [
            t for t in self.tracks.values()
            if t.last_seen_frame == frame_idx
            and t.frames_seen >= self.min_track_len
            and t.avg_area >= self.min_bbox_area
        ]
        if tatami_mask_fn is not None:
            inside = [
                t for t in candidates
                if t.last_ground_point is not None and tatami_mask_fn(t.last_ground_point)
            ]
            if len(inside) >= 2:
                if len(inside) > 2:
                    logger.info(
                        "%d candidates inside tatami (expected 2) - referee is likely among them",
                        len(inside),
                    )
                candidates = inside
        return candidates

    def select_two(self, frame_idx: int, tatami_mask_fn=None):
        candidates = self._candidate_tracks(frame_idx, tatami_mask_fn)
        candidates.sort(key=lambda t: t.avg_area, reverse=True)
        self._assign(frame_idx, candidates[:2])

    def _assign(self, frame_idx: int, top_two: list[TrackStats]):
        available_ids = {t.tracker_id for t in top_two}

        still_assigned = set()
        for athlete_id in (ATHLETE_1, ATHLETE_2):
            if self.athlete_to_tracker[athlete_id] in available_ids:
                still_assigned.add(self.athlete_to_tracker[athlete_id])

        unassigned_athletes = [
            a for a in (ATHLETE_1, ATHLETE_2) if self.athlete_to_tracker[a] not in available_ids
        ]
        unassigned_tracks = [t for t in top_two if t.tracker_id not in still_assigned]

        for athlete_id in unassigned_athletes:
            if not unassigned_tracks:
                break

            last_point = self.athlete_last_point.get(athlete_id)
            if last_point is not None:
                unassigned_tracks.sort(
                    key=lambda t: _dist(t.last_ground_point, last_point)
                    if t.last_ground_point is not None else float("inf")
                )
            best = unassigned_tracks.pop(0)

            old_tracker_id = self.athlete_to_tracker[athlete_id]
            self.athlete_to_tracker[athlete_id] = best.tracker_id

            if old_tracker_id is not None and old_tracker_id != best.tracker_id:
                self.id_switch_count += 1
                dist = (
                    _dist(best.last_ground_point, last_point)
                    if last_point is not None and best.last_ground_point is not None else None
                )
                if dist is not None and dist > self.reassoc_max_dist:
                    logger.warning(
                        "frame %d: %s re-associated tracker %s -> %s at distance %.1f "
                        "(> max %.1f) - low confidence match",
                        frame_idx, athlete_id, old_tracker_id, best.tracker_id, dist, self.reassoc_max_dist,
                    )
                else:
                    logger.info(
                        "frame %d: %s re-associated tracker %s -> %s",
                        frame_idx, athlete_id, old_tracker_id, best.tracker_id,
                    )

            self.reassociation_log.append({
                "frame": frame_idx,
                "athlete_id": athlete_id,
                "old_tracker_id": old_tracker_id,
                "new_tracker_id": best.tracker_id,
            })

    def tracker_to_athlete(self, tracker_id: int | None) -> str | None:
        if tracker_id is None:
            return None
        for athlete_id, tid in self.athlete_to_tracker.items():
            if tid == tracker_id:
                return athlete_id
        return None

    def note_ground_point(self, athlete_id: str, ground_point):
        if ground_point is not None:
            self.athlete_last_point[athlete_id] = ground_point


# --- Belt color: investigated, NOT wired into selection for V1 -------------
#
# Kumite athletes wear a red (aka) or blue (ao) belt/gear, which in principle
# gives a cheap, occlusion-robust signal for re-association after crossings -
# more reliable than "nearest last known position" alone. This function is a
# simple proof of concept: it samples average hue in the lower-third of the
# bbox (roughly where the belt/gear sits) and classifies red vs. blue vs.
# unknown. It is deliberately NOT used in AthleteSelector yet:
#
#   - it hasn't been validated against a real clip (lighting, tatami color,
#     mat branding, and belt visibility during grappling all affect it);
#   - a bad color read could make re-association WORSE, not better, if trusted
#     blindly.
#
# Feasibility verdict goes in the README once tested on real footage.
def estimate_belt_hue(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    band_top = y1 + int((y2 - y1) * 0.55)
    band_bottom = y1 + int((y2 - y1) * 0.85)
    band = frame[band_top:band_bottom, x1:x2]
    if band.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = hsv.reshape(-1, 3).mean(axis=0)

    if mean_s < 60 or mean_v < 40:
        return "unknown"
    if mean_h <= 10 or mean_h >= 170:
        return "red"
    if 100 <= mean_h <= 130:
        return "blue"
    return "unknown"
