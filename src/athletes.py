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

# Color sampling needs a much higher bar than the seated/brightness filter:
# a wrong hip/wrist position (even at KEYPOINT_CONF_MIN=0.3) can land the
# patch on the tatami (also red/blue) instead of the belt/glove.
KEYPOINT_CONF_MIN_COLOR = 0.55
COLOR_PATCH_RADIUS = 6

# Per-PIXEL color classes inside a patch (not the patch mean: at 360p a belt
# is ~5px thick, so a 12px patch is mostly white gi and its mean is washed out
# to S~30-60 even for a vividly red belt).
RED_SAT_MIN, RED_VAL_MIN = 100, 70
BLUE_SAT_MIN, BLUE_VAL_MIN = 100, 50
# Dark navy / purple (a blue belt at 360p): weak blue evidence, not "unknown".
WEAK_BLUE_HUE = (100, 155)
WEAK_BLUE_SAT_MIN = 60
WEAK_BLUE_VAL = (25, 120)
WEAK_BLUE_WEIGHT = 0.5
# A patch is only labeled red/blue (for debug strips / jsonl) above this
# pixel fraction; the evidence itself is the raw fraction, never thresholded.
PATCH_LABEL_MIN_FRAC = 0.15

# Track-level official (dark suit) verdict.
OFFICIAL_MIN_DARK_FRAMES = 3
OFFICIAL_DARK_FRAC = 0.3
OFFICIAL_WINDOW = 15   # frames each side (pass 1) / 2x trailing (online)

# Pair-level identity decision. Evidence is cumulative per track over the
# whole clip: red_ev / blue_ev are sums of per-patch pixel fractions.
MIN_EVIDENCE = 3.0
# Both athletes have evidence: AKA is whoever is redder relative to the other
# (score = (R-B)/(R+B) in [-1,1]); no absolute threshold, only this gap.
PAIR_MARGIN = 0.25
# No reference opponent (or only one has evidence): its own score must be at
# least this far from 0 to say which belt it is.
SINGLE_MIN = 0.5


@dataclass
class TrackStats:
    tracker_id: int
    frames_seen: int = 0
    total_area: float = 0.0
    last_ground_point: tuple[float, float] | None = None
    last_seen_frame: int = -1
    last_passed_filter: bool = False
    last_reject_reason: str | None = None
    red_ev: float = 0.0
    blue_ev: float = 0.0
    # (frame, failed-dark-check) per frame seen. Feeds a WINDOWED track-level
    # verdict: the per-frame check silently passes whenever the torso
    # keypoints are too weak to measure, so a suited referee slips through on
    # exactly those frames. Windowed (not whole-lifetime) because tracker IDs
    # get reused across different people (a track can be the referee for 30s
    # and an athlete after).
    dark_flags: list = field(default_factory=list)

    @property
    def avg_area(self) -> float:
        return self.total_area / self.frames_seen if self.frames_seen else 0.0


def evidence_score(red: float, blue: float) -> float | None:
    """(R-B)/(R+B) in [-1 (all blue), +1 (all red)]; None if too little evidence."""
    total = red + blue
    if total < MIN_EVIDENCE:
        return None
    return (red - blue) / total


def _dist(a, b) -> float:
    return math.dist(a, b)


def _kp_ok(keypoints, idx: int, threshold: float = KEYPOINT_CONF_MIN) -> bool:
    return keypoints[idx][2] >= threshold


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


def scale_rect(rect, ref_resolution, frame_size):
    """Rect given at ref_resolution (w, h) -> pixel rect at the actual frame size."""
    sx, sy = frame_size[0] / ref_resolution[0], frame_size[1] / ref_resolution[1]
    return (rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy)


def graphic_on_screen(frame: np.ndarray, only_when: dict, ref_resolution) -> bool:
    """Broadcast graphics (YUKO/IPPON panel) put a dark text bar at a fixed
    spot; on a normal frame that spot is bright tatami. True when at least
    `dark_frac_min` of the pixels in `rect` have V < `dark_val_below`."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in scale_rect(only_when["rect"], ref_resolution, (w, h)))
    region = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if region.size == 0:
        return False
    v = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float((v < only_when["dark_val_below"]).mean()) >= only_when["dark_frac_min"]


def active_ignore_zones(frame: np.ndarray, zone_cfgs: list[dict]) -> list[tuple[float, float, float, float]]:
    """Pixel-space ignore rects that apply to THIS frame. Each cfg:
      rect: [x1,y1,x2,y2] at ref_resolution [w,h] (scaled to the real frame size)
      only_when (optional): graphic-detection spec; without it the zone is
        always on. With it, the zone applies only while the graphic is up, so
        a real athlete standing there on a clean frame is never dropped."""
    h, w = frame.shape[:2]
    active = []
    for cfg in zone_cfgs:
        ref = cfg.get("ref_resolution", [w, h])
        only_when = cfg.get("only_when")
        if only_when is not None and not graphic_on_screen(frame, only_when, ref):
            continue
        active.append(scale_rect(cfg["rect"], ref, (w, h)))
    return active


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


def _belt_point(keypoints, bbox) -> tuple[float, float] | None:
    """Midpoint between the two hip keypoints, nudged up toward where the
    obi/belt actually sits rather than the hip joints themselves - moving up
    (not down toward the legs/floor) also keeps the patch away from the
    tatami, which is the same red/blue as the belts."""
    if not (_kp_ok(keypoints, LEFT_HIP, KEYPOINT_CONF_MIN_COLOR) and _kp_ok(keypoints, RIGHT_HIP, KEYPOINT_CONF_MIN_COLOR)):
        return None
    lx, ly, _ = keypoints[LEFT_HIP]
    rx, ry, _ = keypoints[RIGHT_HIP]
    mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
    bbox_h = max(bbox[3] - bbox[1], 1.0)
    my -= 0.06 * bbox_h
    return mx, my


def classify_belt_color(
    frame: np.ndarray, bbox: tuple[float, float, float, float], keypoints,
) -> tuple[str, list[dict], tuple[float, float]]:
    """Samples a small patch at the belt (hip midpoint, shifted up) and at
    each wrist (glove), strictly clamped inside the detection's own bbox so a
    patch can never land on the tatami floor or another person.

    Returns (color, samples, evidence):
      evidence = (red, blue) summed over the patches - the per-pixel red and
        (strong + 0.5*weak dark-navy/purple) blue fractions. This, not a
        per-frame verdict, is what the selector accumulates over the clip.
      color    = red/blue/unknown for this frame (debug/jsonl only).
      samples  = one dict per attempted patch for the debug strips: label,
        pixel rect, red/blue fractions, color (or "no_keypoint")."""
    samples: list[dict] = []
    red_total = blue_total = 0.0

    belt_pt = _belt_point(keypoints, bbox)
    points = [("belt", belt_pt)] + [
        (label, (keypoints[idx][0], keypoints[idx][1]) if _kp_ok(keypoints, idx, KEYPOINT_CONF_MIN_COLOR) else None)
        for label, idx in (("left_wrist", LEFT_WRIST), ("right_wrist", RIGHT_WRIST))
    ]
    for label, pt in points:
        if pt is None:
            samples.append({"label": label, "rect": None, "color": "no_keypoint", "red": 0.0, "blue": 0.0})
            continue
        red, blue, rect = _sample_patch_evidence(frame, bbox, *pt)
        color = "unknown"
        if rect is not None:
            if red >= PATCH_LABEL_MIN_FRAC and red > blue:
                color = "red"
            elif blue >= PATCH_LABEL_MIN_FRAC and blue > red:
                color = "blue"
        samples.append({"label": label, "rect": rect, "color": color, "red": red, "blue": blue})
        red_total += red
        blue_total += blue

    if red_total == blue_total == 0.0:
        return "unknown", samples, (0.0, 0.0)
    return ("red" if red_total > blue_total else "blue"), samples, (red_total, blue_total)


def patch_pixel_evidence(patch_bgr: np.ndarray) -> tuple[float, float]:
    """(red_frac, blue_frac) of pixels in a BGR patch. blue_frac counts
    saturated blue fully and dark navy/purple at WEAK_BLUE_WEIGHT."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, sat, val = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    red = ((h <= 10) | (h >= 168)) & (sat >= RED_SAT_MIN) & (val >= RED_VAL_MIN)
    strong_blue = (h >= 100) & (h <= 130) & (sat >= BLUE_SAT_MIN) & (val >= BLUE_VAL_MIN)
    weak_blue = (
        (h >= WEAK_BLUE_HUE[0]) & (h <= WEAK_BLUE_HUE[1]) & (sat >= WEAK_BLUE_SAT_MIN)
        & (val >= WEAK_BLUE_VAL[0]) & (val <= WEAK_BLUE_VAL[1]) & ~strong_blue
    )
    n = len(hsv)
    return float(red.sum()) / n, float(strong_blue.sum() + WEAK_BLUE_WEIGHT * weak_blue.sum()) / n


def _sample_patch_evidence(
    frame: np.ndarray, bbox: tuple[float, float, float, float], x: float, y: float,
    radius: int = COLOR_PATCH_RADIUS,
) -> tuple[float, float, tuple[int, int, int, int] | None]:
    bx1, by1, bx2, by2 = bbox
    x1 = max(int(x - radius), int(bx1))
    y1 = max(int(y - radius), int(by1))
    x2 = min(int(x + radius), int(bx2))
    y2 = min(int(y + radius), int(by2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, None
    red, blue = patch_pixel_evidence(frame[y1:y2, x1:x2])
    return red, blue, (x1, y1, x2, y2)


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

        # Whether the CURRENT holder of each slot has color evidence backing
        # its AKA/AO label. False -> render "?" instead of guessing from position.
        self.identity_known: dict[str, bool] = {AKA: False, AO: False}
        # Optional {tracker_id: (red, blue)} computed over the WHOLE clip in a
        # first pass; when set it replaces the online running totals.
        self.final_evidence: dict[int, tuple[float, float]] | None = None

        # Optional {tracker_id: [(frame, dark_flag), ...]} from pass 1: lets
        # the official verdict use a window centered on the frame (future
        # frames too) instead of only the trailing one.
        self.final_dark_flags: dict[int, list] | None = None

        self.id_switch_count = 0
        self.color_swap_count = 0
        self.reassociation_log: list[dict] = []
        # Per-frame bookkeeping for the metrics report (see analysis/metrics.py).
        self.frame_log: list[dict] = []

    def update_track_stats(self, frame_idx: int, detections, ground_points, filter_results, evidence):
        """evidence: per detection (red, blue) from classify_belt_color()."""
        for det, gp, (passes, reject_reason), (red, blue) in zip(detections, ground_points, filter_results, evidence):
            if det.tracker_id is None:
                continue
            area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
            stats = self.tracks.setdefault(det.tracker_id, TrackStats(det.tracker_id))
            stats.frames_seen += 1
            stats.total_area += area
            stats.last_seen_frame = frame_idx
            stats.last_passed_filter = passes
            stats.last_reject_reason = reject_reason
            stats.dark_flags.append((frame_idx, bool(reject_reason and reject_reason.startswith("dark_clothing"))))
            stats.red_ev += red
            stats.blue_ev += blue
            if gp is not None:
                stats.last_ground_point = gp

    def is_official(self, track: TrackStats, frame_idx: int) -> bool:
        """Windowed track-level dark-clothing rejection (referee/coach): the
        track failed the dark check in >= OFFICIAL_DARK_FRAC of the frames
        around `frame_idx` (centered with pass-1 flags, trailing online), so
        it is an official on frames where its torso couldn't be measured too.
        Evaluated in _candidate_tracks, i.e. on every (re)assignment."""
        if self.final_dark_flags is not None:
            flags = self.final_dark_flags.get(track.tracker_id, [])
            lo, hi = frame_idx - OFFICIAL_WINDOW, frame_idx + OFFICIAL_WINDOW
        else:
            flags = track.dark_flags
            lo, hi = frame_idx - 2 * OFFICIAL_WINDOW, frame_idx
        window = [d for f, d in flags if lo <= f <= hi]
        dark = sum(window)
        return dark >= OFFICIAL_MIN_DARK_FRAMES and dark / len(window) >= OFFICIAL_DARK_FRAC

    def score(self, tracker_id: int | None) -> float | None:
        if tracker_id is None:
            return None
        if self.final_evidence is not None:
            red, blue = self.final_evidence.get(tracker_id, (0.0, 0.0))
        else:
            t = self.tracks.get(tracker_id)
            red, blue = (t.red_ev, t.blue_ev) if t else (0.0, 0.0)
        return evidence_score(red, blue)

    def verdict(self, tracker_id: int | None) -> str:
        """Absolute read of one track, used only to prefer a color-matching
        candidate when re-associating; the AKA/AO decision is made jointly
        in _resolve_identity()."""
        sc = self.score(tracker_id)
        if sc is None or abs(sc) < SINGLE_MIN:
            return "unknown"
        return "red" if sc > 0 else "blue"

    def _candidate_tracks(self, frame_idx: int, tatami_mask_fn) -> list[TrackStats]:
        # Must be visible AND pass the athlete filter THIS frame - a track
        # that disappeared, sat down, or is a suited official right now
        # should not keep winning selection just because it was big/
        # persistent earlier.
        candidates = [
            t for t in self.tracks.values()
            if t.last_seen_frame == frame_idx
            and t.last_passed_filter
            and not self.is_official(t, frame_idx)
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
        self._resolve_identity(frame_idx)
        self.frame_log.append({
            "frame": frame_idx,
            "n_candidates": len(candidates),
            "aka_tracker": self.slot_to_tracker[AKA],
            "ao_tracker": self.slot_to_tracker[AO],
            "aka_known": self.identity_known[AKA],
            "ao_known": self.identity_known[AO],
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
            color_matches = [t for t in unassigned_tracks if self.verdict(t.tracker_id) == wanted_color]

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

    def _resolve_identity(self, frame_idx: int):
        """Decide AKA/AO jointly for the pair from cumulative red/blue evidence.

        The two athletes are compared RELATIVELY: whoever is redder (by
        PAIR_MARGIN) is AKA, neither needs to clear an absolute bar. When only
        one slot is currently held, the empty slot's most recent holder is the
        reference, so a lone athlete is still judged against their opponent
        rather than in isolation (a pale gi bleeds into "slightly red"). With
        no reference at all, the lone athlete's own score must clear
        SINGLE_MIN. No usable evidence: the slot keeps its internal
        continuity label but identity_known stays False ("?")."""
        holders = {s: self.slot_to_tracker[s] for s in (AKA, AO)}
        if holders[AKA] is None and holders[AO] is None:
            self.identity_known = {AKA: False, AO: False}
            return
        ref = {s: holders[s] if holders[s] is not None else self.slot_last_tracker_id[s] for s in (AKA, AO)}
        if ref[AKA] is not None and ref[AKA] == ref[AO]:
            ref = {s: holders[s] for s in (AKA, AO)}
        a, o = self.score(ref[AKA]), self.score(ref[AO])
        self.identity_known = {AKA: False, AO: False}

        if a is not None and o is not None:
            if abs(a - o) < PAIR_MARGIN:
                return
            aka_is_redder = a > o
        else:
            held = AKA if holders[AKA] is not None else AO
            s_x = a if held == AKA else o
            if holders[AKA] is not None and holders[AO] is not None:
                held = AKA if a is not None else AO
                s_x = a if a is not None else o
            if s_x is None or abs(s_x) < SINGLE_MIN:
                return
            x_is_red = s_x > 0
            aka_is_redder = x_is_red if held == AKA else not x_is_red

        if not aka_is_redder:
            self._swap_slots(frame_idx, {AKA: a, AO: o})
        for slot in (AKA, AO):
            self.identity_known[slot] = self.slot_to_tracker[slot] is not None

    def _swap_slots(self, frame_idx: int, scores):
        old_aka, old_ao = self.slot_to_tracker[AKA], self.slot_to_tracker[AO]
        logger.info(
            "frame %d: AKA/AO swapped by color evidence (aka-slot tracker %s score=%s, ao-slot tracker %s score=%s)",
            frame_idx, old_aka, scores[AKA], old_ao, scores[AO],
        )
        self.slot_to_tracker[AKA], self.slot_to_tracker[AO] = old_ao, old_aka
        self.slot_last_tracker_id[AKA], self.slot_last_tracker_id[AO] = (
            self.slot_last_tracker_id[AO], self.slot_last_tracker_id[AKA],
        )
        aka_pt, ao_pt = self.slot_last_point.get(AKA), self.slot_last_point.get(AO)
        for slot, pt in ((AKA, ao_pt), (AO, aka_pt)):
            if pt is not None:
                self.slot_last_point[slot] = pt
            else:
                self.slot_last_point.pop(slot, None)
        self.color_swap_count += 1
        self.reassociation_log.append({
            "frame": frame_idx, "slot": "swap", "old_tracker_id": old_aka,
            "new_tracker_id": old_ao, "reason": "color evidence",
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
