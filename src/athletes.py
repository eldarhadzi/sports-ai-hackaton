"""Choosing the two athletes among everyone visible (referee, judges, coaches,
audience) and keeping a stable AKA (red) / AO (blue) identity across
tracker-ID switches (which happen after occlusions/crossings).

Selection is no longer just "largest, most persistent track" - real footage
showed that alone regularly picks seated officials or the referee instead of
the athletes. A detection must also pass `passes_athlete_filter()` (not
seated, not dark-suited, not a broadcast-graphic overlay) every frame to be a
candidate, and re-association prefers a belt/glove-color match (AKA=red,
AO=blue) over pure distance.
"""
from collections import deque
from dataclasses import dataclass, field
import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

AKA = "aka"  # red belt
AO = "ao"    # blue belt
SLOT_COLORS = {AKA: "red", AO: "blue"}

# COCO-17 keypoint indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14

KEYPOINT_CONF_MIN = 0.3
COLOR_VOTE_WINDOW = 15
COLOR_VOTE_MIN_COUNT = 4
COLOR_VOTE_MIN_MAJORITY = 0.6


@dataclass
class TrackStats:
    tracker_id: int
    frames_seen: int = 0
    total_area: float = 0.0
    last_ground_point: tuple[float, float] | None = None
    last_seen_frame: int = -1
    last_passed_filter: bool = False
    last_reject_reason: str | None = None
    color_votes: deque = field(default_factory=lambda: deque(maxlen=COLOR_VOTE_WINDOW))

    @property
    def avg_area(self) -> float:
        return self.total_area / self.frames_seen if self.frames_seen else 0.0

    @property
    def color_verdict(self) -> str:
        votes = [c for c in self.color_votes if c != "unknown"]
        if len(votes) < COLOR_VOTE_MIN_COUNT:
            return "unknown"
        red = votes.count("red")
        blue = votes.count("blue")
        top_color, top_count = ("red", red) if red >= blue else ("blue", blue)
        if top_count / len(votes) >= COLOR_VOTE_MIN_MAJORITY:
            return top_color
        return "unknown"


def _dist(a, b) -> float:
    return math.dist(a, b)


def _kp_ok(keypoints, idx: int) -> bool:
    return keypoints[idx][2] >= KEYPOINT_CONF_MIN


def is_seated(keypoints) -> bool | None:
    """True if this person looks seated (hips ~level with knees, not well
    above them like a standing person). None if there isn't enough confident
    pose to tell - callers should NOT reject on None."""
    shoulder_ys = [keypoints[i][1] for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if _kp_ok(keypoints, i)]
    hip_ys = [keypoints[i][1] for i in (LEFT_HIP, RIGHT_HIP) if _kp_ok(keypoints, i)]
    knee_ys = [keypoints[i][1] for i in (LEFT_KNEE, RIGHT_KNEE) if _kp_ok(keypoints, i)]
    if not shoulder_ys or not hip_ys or not knee_ys:
        return None

    shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
    hip_y = sum(hip_ys) / len(hip_ys)
    knee_y = sum(knee_ys) / len(knee_ys)

    torso = hip_y - shoulder_y
    thigh = knee_y - hip_y
    if torso <= 1:
        return None
    return (thigh / torso) < 0.35


def torso_brightness(frame: np.ndarray, keypoints) -> float | None:
    """Mean V-channel brightness of the region between shoulders and hips -
    where a white gi or a dark suit jacket would show. None if not enough
    confident keypoints."""
    shoulder_pts = [keypoints[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if _kp_ok(keypoints, i)]
    hip_pts = [keypoints[i] for i in (LEFT_HIP, RIGHT_HIP) if _kp_ok(keypoints, i)]
    if not shoulder_pts or not hip_pts:
        return None

    xs = [p[0] for p in shoulder_pts + hip_pts]
    ys = [p[1] for p in shoulder_pts + hip_pts]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    pad = max(4, int((x2 - x1) * 0.15))
    x1, y1 = max(x1 - pad, 0), max(y1, 0)
    x2 = min(x2 + pad, frame.shape[1])
    y2 = min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    v_channel = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(v_channel.mean())


def bbox_ignore_overlap_frac(bbox: tuple[float, float, float, float], zone: tuple[float, float, float, float]) -> float:
    bx1, by1, bx2, by2 = bbox
    zx1, zy1, zx2, zy2 = zone
    ix1, iy1 = max(bx1, zx1), max(by1, zy1)
    ix2, iy2 = min(bx2, zx2), min(by2, zy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    bbox_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return inter / bbox_area


def in_ignore_zone(bbox, ignore_zones, overlap_frac_threshold: float = 0.5) -> bool:
    return any(
        bbox_ignore_overlap_frac(bbox, zone) >= overlap_frac_threshold
        for zone in ignore_zones
    )


def passes_athlete_filter(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    keypoints,
    ignore_zones,
    dark_brightness_threshold: float,
) -> tuple[bool, str | None]:
    """Returns (passes, reject_reason). reject_reason is None when passing."""
    if in_ignore_zone(bbox, ignore_zones):
        return False, "ignore_zone (broadcast overlay)"

    seated = is_seated(keypoints)
    if seated:
        return False, "seated"

    brightness = torso_brightness(frame, keypoints)
    if brightness is not None and brightness < dark_brightness_threshold:
        return False, f"dark_clothing (brightness={brightness:.0f})"

    return True, None


def classify_belt_color(frame: np.ndarray, keypoints) -> str:
    """Samples small patches around hips (belt) and wrists (gloves), classifies
    each red/blue/unknown by hue, and majority-votes. Returns "unknown" if no
    patch yields a confident read."""
    votes = []
    for idx in (LEFT_HIP, RIGHT_HIP, LEFT_WRIST, RIGHT_WRIST):
        if not _kp_ok(keypoints, idx):
            continue
        x, y = keypoints[idx][0], keypoints[idx][1]
        color = _sample_patch_color(frame, x, y)
        if color != "unknown":
            votes.append(color)

    if not votes:
        return "unknown"
    red = votes.count("red")
    blue = votes.count("blue")
    return "red" if red >= blue else "blue"


def _sample_patch_color(frame: np.ndarray, x: float, y: float, radius: int = 8) -> str:
    h, w = frame.shape[:2]
    x1, y1 = max(int(x - radius), 0), max(int(y - radius), 0)
    x2, y2 = min(int(x + radius), w), min(int(y + radius), h)
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_h, mean_s, mean_v = hsv.reshape(-1, 3).mean(axis=0)

    if mean_s < 60 or mean_v < 40:
        return "unknown"
    if mean_h <= 10 or mean_h >= 170:
        return "red"
    if 100 <= mean_h <= 130:
        return "blue"
    return "unknown"


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
        # Current, live mapping - cleared to None the instant a slot's
        # tracker stops being a valid candidate, so a stale/invalid tracker
        # is never drawn or exported as an athlete even for one extra frame.
        self.slot_to_tracker: dict[str, int | None] = {AKA: None, AO: None}
        # Last tracker id/position this slot was ever confidently linked to -
        # persists across a multi-frame unassigned gap (unlike the above), so
        # re-association after a gap still knows where to look and still
        # counts as one id switch rather than silently forgetting.
        self.slot_last_tracker_id: dict[str, int | None] = {AKA: None, AO: None}
        self.slot_last_point: dict[str, tuple[float, float]] = {}

        self.id_switch_count = 0
        self.color_swap_count = 0
        self.reassociation_log: list[dict] = []
        # Per-frame bookkeeping for the metrics report (see analysis/metrics.py).
        self.frame_log: list[dict] = []

    def update_track_stats(self, frame_idx: int, detections, ground_points, filter_results, colors):
        for det, gp, (passes, reject_reason), color in zip(detections, ground_points, filter_results, colors):
            if det.tracker_id is None:
                continue
            area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
            stats = self.tracks.setdefault(det.tracker_id, TrackStats(det.tracker_id))
            stats.frames_seen += 1
            stats.total_area += area
            stats.last_seen_frame = frame_idx
            stats.last_passed_filter = passes
            stats.last_reject_reason = reject_reason
            stats.color_votes.append(color)
            if gp is not None:
                stats.last_ground_point = gp

    def _candidate_tracks(self, frame_idx: int, tatami_mask_fn) -> list[TrackStats]:
        # Must be visible AND pass the athlete filter THIS frame - a track
        # that disappeared, sat down, or is a suited official right now
        # should not keep winning selection just because it was big/
        # persistent earlier.
        candidates = [
            t for t in self.tracks.values()
            if t.last_seen_frame == frame_idx
            and t.last_passed_filter
            and t.frames_seen >= self.min_track_len
            and t.avg_area >= self.min_bbox_area
        ]
        if tatami_mask_fn is not None:
            candidates = [
                t for t in candidates
                if t.last_ground_point is not None and tatami_mask_fn(t.last_ground_point)
            ]
            if len(candidates) > 2:
                logger.info(
                    "frame %d: %d candidates inside tatami (expected 2) - referee is likely among them",
                    frame_idx, len(candidates),
                )
        return candidates

    def select_two(self, frame_idx: int, tatami_mask_fn=None):
        candidates = self._candidate_tracks(frame_idx, tatami_mask_fn)
        candidates.sort(key=lambda t: t.avg_area, reverse=True)
        self._assign(frame_idx, candidates[:2])
        self._maybe_correct_color_swap(frame_idx)
        self.frame_log.append({
            "frame": frame_idx,
            "n_candidates": len(candidates),
            "aka_tracker": self.slot_to_tracker[AKA],
            "ao_tracker": self.slot_to_tracker[AO],
        })

    def _closest(self, tracks: list[TrackStats], slot: str) -> TrackStats:
        last_point = self.slot_last_point.get(slot)
        if last_point is None:
            return tracks[0]
        return min(
            tracks,
            key=lambda t: _dist(t.last_ground_point, last_point) if t.last_ground_point is not None else float("inf"),
        )

    def _assign(self, frame_idx: int, top_two: list[TrackStats]):
        available_ids = {t.tracker_id for t in top_two}

        still_assigned = set()
        for slot in (AKA, AO):
            if self.slot_to_tracker[slot] in available_ids:
                still_assigned.add(self.slot_to_tracker[slot])

        unassigned_slots = [s for s in (AKA, AO) if self.slot_to_tracker[s] not in available_ids]
        unassigned_tracks = [t for t in top_two if t.tracker_id not in still_assigned]

        for slot in unassigned_slots:
            # Clear the LIVE mapping immediately: an old tracker that's no
            # longer a valid candidate (failed the filter, left the mat,
            # lost the size/persistence race) must stop being drawn/exported
            # as this slot right away, even if no replacement is found this
            # frame. slot_last_tracker_id/slot_last_point are NOT cleared -
            # they persist across the gap so a reconnect several frames
            # later still knows where to look and still counts as one switch.
            self.slot_to_tracker[slot] = None
            last_known_tracker_id = self.slot_last_tracker_id[slot]
            last_point = self.slot_last_point.get(slot)

            if not unassigned_tracks:
                continue

            wanted_color = SLOT_COLORS[slot]
            color_matches = [t for t in unassigned_tracks if t.color_verdict == wanted_color]

            best = None
            reason = None
            if color_matches:
                best = self._closest(color_matches, slot)
                reason = f"color match ({wanted_color})"
            else:
                candidate = self._closest(unassigned_tracks, slot)
                dist = (
                    _dist(candidate.last_ground_point, last_point)
                    if last_point is not None and candidate.last_ground_point is not None else None
                )
                if dist is not None and dist > self.reassoc_max_dist:
                    logger.warning(
                        "frame %d: %s has no confident match (color unknown, nearest candidate "
                        "tracker %s at distance %.1f > max %.1f) - leaving unassigned this frame",
                        frame_idx, slot, candidate.tracker_id, dist, self.reassoc_max_dist,
                    )
                    continue
                best = candidate
                reason = "distance fallback (color unknown)" if last_point is not None else "bootstrap (no prior position)"

            unassigned_tracks.remove(best)
            self.slot_to_tracker[slot] = best.tracker_id
            self.slot_last_tracker_id[slot] = best.tracker_id

            if last_known_tracker_id is not None and last_known_tracker_id != best.tracker_id:
                self.id_switch_count += 1

            logger.info(
                "frame %d: %s <- tracker %s (%s)%s",
                frame_idx, slot, best.tracker_id, reason,
                f", was tracker {last_known_tracker_id}" if last_known_tracker_id is not None else " [initial]",
            )
            self.reassociation_log.append({
                "frame": frame_idx, "slot": slot, "old_tracker_id": last_known_tracker_id,
                "new_tracker_id": best.tracker_id, "reason": reason,
            })

    def _maybe_correct_color_swap(self, frame_idx: int):
        aka_track = self.tracks.get(self.slot_to_tracker[AKA])
        ao_track = self.tracks.get(self.slot_to_tracker[AO])
        if aka_track is None or ao_track is None:
            return
        if aka_track.color_verdict == "blue" and ao_track.color_verdict == "red":
            logger.info(
                "frame %d: AKA/AO swapped - color evidence says they're reversed "
                "(tracker %s reads blue, tracker %s reads red)",
                frame_idx, aka_track.tracker_id, ao_track.tracker_id,
            )
            self.slot_to_tracker[AKA], self.slot_to_tracker[AO] = (
                self.slot_to_tracker[AO], self.slot_to_tracker[AKA],
            )
            self.slot_last_tracker_id[AKA], self.slot_last_tracker_id[AO] = (
                self.slot_last_tracker_id[AO], self.slot_last_tracker_id[AKA],
            )
            if AKA in self.slot_last_point or AO in self.slot_last_point:
                aka_pt = self.slot_last_point.get(AKA)
                ao_pt = self.slot_last_point.get(AO)
                if ao_pt is not None:
                    self.slot_last_point[AKA] = ao_pt
                else:
                    self.slot_last_point.pop(AKA, None)
                if aka_pt is not None:
                    self.slot_last_point[AO] = aka_pt
                else:
                    self.slot_last_point.pop(AO, None)
            self.color_swap_count += 1
            self.reassociation_log.append({
                "frame": frame_idx, "slot": "swap", "old_tracker_id": aka_track.tracker_id,
                "new_tracker_id": ao_track.tracker_id, "reason": "color correction",
            })

    def tracker_to_slot(self, tracker_id: int | None) -> str | None:
        if tracker_id is None:
            return None
        for slot, tid in self.slot_to_tracker.items():
            if tid == tracker_id:
                return slot
        return None

    def note_ground_point(self, slot: str, ground_point):
        if ground_point is not None:
            self.slot_last_point[slot] = ground_point
