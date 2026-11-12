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
from bisect import bisect_left, bisect_right
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

# Track-level official verdict from the LOWER body (round v8): a referee in a white
# shirt passes the dark-torso check, but wears dark trousers and no gloves, while an
# athlete wears white gi trousers and red/blue gloves. Measured on the labeled frames
# (median V along hip-knee-ankle): athletes p5 = 139 (club_02) / 164 (broadcast),
# club_02 referee p95 = 63. Glove color (strong red/blue only - the weak navy band
# reads dark trousers as blue) is a veto, never a reason to reject: athletes show it
# in only ~50% of frames.
LEG_SEGMENTS = ((LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, 15), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, 16))
LEG_MIN_SAMPLES = 2
LEG_MIN_MEASURED = 3        # measured frames in the window before a verdict
LEG_DARK_FRAC = 0.5         # of measured frames in the window with dark legs
GLOVE_MIN_FRAC = 0.15       # strong red/blue pixel fraction at a wrist = a glove seen
GLOVE_VETO_FRAC = 0.2       # gloves seen in >= this fraction of the window -> not an official

# Evidence-based CONTINUITY (not identity - that is identity.py's job): when a
# slot's track vanishes and new tracks appear, a candidate whose color evidence
# in the next CONT_WINDOW frames resembles the lost slot's last CONT_WINDOW
# frames is the same athlete. Scores are (R-B)/(R+B) in [-1, 1].
CONT_WINDOW = 30
MIN_WINDOW_EVIDENCE = 2.0
CONT_CONFIRMED = 0.4    # |score diff| below this => same athlete, distance gate waived


@dataclass
class TrackStats:
    tracker_id: int
    frames_seen: int = 0
    total_area: float = 0.0
    last_ground_point: tuple[float, float] | None = None
    last_seen_frame: int = -1
    last_passed_filter: bool = False
    last_reject_reason: str | None = None
    # (frame, red, blue) per frame seen - windowed color evidence.
    ev: list = field(default_factory=list)
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
    if total < MIN_WINDOW_EVIDENCE:
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


def leg_brightness(frame: np.ndarray, keypoints, bbox) -> float | None:
    """Median V of small patches along hip-knee and knee-ankle (30/50/70% of each
    segment) - white gi trousers vs dark suit trousers. None if fewer than
    LEG_MIN_SAMPLES segments points are confident."""
    h, w = frame.shape[:2]
    r = max(2, int((bbox[3] - bbox[1]) * 0.02))
    values = []
    for a, b in LEG_SEGMENTS:
        if not (_kp_ok(keypoints, a) and _kp_ok(keypoints, b)):
            continue
        for t in (0.3, 0.5, 0.7):
            x = int(keypoints[a][0] + t * (keypoints[b][0] - keypoints[a][0]))
            y = int(keypoints[a][1] + t * (keypoints[b][1] - keypoints[a][1]))
            patch = frame[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
            if patch.size:
                values.append(float(np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2])))
    return float(np.median(values)) if len(values) >= LEG_MIN_SAMPLES else None


def glove_evidence(frame: np.ndarray, keypoints, bbox) -> float | None:
    """Largest STRONG red-or-blue pixel fraction of the two wrist patches (no weak
    navy). None if no wrist keypoint is confident."""
    best = None
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        if not _kp_ok(keypoints, idx, KEYPOINT_CONF_MIN_COLOR):
            continue
        _, _, rect = _sample_patch_evidence(frame, bbox, keypoints[idx][0], keypoints[idx][1])
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        red = (((hue <= 10) | (hue >= 168)) & (sat >= RED_SAT_MIN) & (val >= RED_VAL_MIN)).mean()
        blue = ((hue >= 100) & (hue <= 130) & (sat >= BLUE_SAT_MIN) & (val >= BLUE_VAL_MIN)).mean()
        best = max(best or 0.0, float(red), float(blue))
    return best


def frame_window(rows: list[tuple], lo: int, hi: int) -> list[tuple]:
    """Rows (frame, ...) of one track, appended in frame order, with lo <= frame <= hi.
    Binary search: a whole-clip scan per call is quadratic on long tracks."""
    return rows[bisect_left(rows, lo, key=lambda r: r[0]):bisect_right(rows, hi, key=lambda r: r[0])]


def official_by_legs(rows: list[tuple[int, float | None, float | None]], lo: int, hi: int, leg_dark: float) -> bool:
    """rows: (frame, leg brightness, glove evidence) of one track. Official over
    frames [lo, hi] if most measured frames show dark legs and gloves are (almost)
    never seen."""
    window = [(v, g) for _, v, g in frame_window(rows, lo, hi)]
    legs = [v for v, _ in window if v is not None]
    if len(legs) < LEG_MIN_MEASURED:
        return False
    if sum(v < leg_dark for v in legs) / len(legs) < LEG_DARK_FRAC:
        return False
    gloves = [g for _, g in window if g is not None]
    return not gloves or sum(g >= GLOVE_MIN_FRAC for g in gloves) / len(gloves) < GLOVE_VETO_FRAC


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
    """Is a broadcast graphic up? Looks at a small `rect` where the graphic draws
    something a normal frame never has there, and needs `frac_min` of its pixels
    to match. Two matchers:
      dark_val_below: V below this (the dark text bar of the YUKO/IPPON panel);
      hsv_min/hsv_max: inside this inclusive HSV range (the yellow card swatch of
        the Chukoku panel).
    (`dark_frac_min` is the old spelling of `frac_min`.)"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in scale_rect(only_when["rect"], ref_resolution, (w, h)))
    region = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if region.size == 0:
        return False
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    if "hsv_min" in only_when:
        match = cv2.inRange(hsv, np.array(only_when["hsv_min"], np.uint8), np.array(only_when["hsv_max"], np.uint8)) > 0
    else:
        match = hsv[:, :, 2] < only_when["dark_val_below"]
    return float(match.mean()) >= only_when.get("frac_min", only_when.get("dark_frac_min"))


def zone_flags(frame: np.ndarray, zone_cfgs: list[dict]) -> list[bool]:
    """Per zone: is its graphic up on THIS frame? Zones without `only_when` are
    always on."""
    h, w = frame.shape[:2]
    return [
        cfg.get("only_when") is None or graphic_on_screen(frame, cfg["only_when"], cfg.get("ref_resolution", [w, h]))
        for cfg in zone_cfgs
    ]


def dilate_flags(flags: list[bool], pad: int) -> list[bool]:
    """True within `pad` frames of any True. Graphics fade in/out: the pictogram
    figures are on screen a few frames before the marker pixels used to detect the
    graphic show up (and after they are gone). Offline we can look both ways."""
    out = [False] * len(flags)
    for i, v in enumerate(flags):
        if v:
            for j in range(max(0, i - pad), min(len(flags), i + pad + 1)):
                out[j] = True
    return out


def zone_rects(frame_size: tuple[int, int], zone_cfgs: list[dict], on: list[bool]) -> list[tuple[float, float, float, float]]:
    """Pixel-space ignore rects of the zones switched on."""
    return [scale_rect(c["rect"], c.get("ref_resolution", list(frame_size)), frame_size)
            for c, v in zip(zone_cfgs, on) if v]


def active_ignore_zones(frame: np.ndarray, zone_cfgs: list[dict]) -> list[tuple[float, float, float, float]]:
    """Pixel-space ignore rects that apply to THIS frame. Each cfg:
      rect: [x1,y1,x2,y2] at ref_resolution [w,h] (scaled to the real frame size)
      only_when (optional): graphic-detection spec; without it the zone is
        always on. With it, the zone applies only while the graphic is up, so
        a real athlete standing there on a clean frame is never dropped."""
    h, w = frame.shape[:2]
    return zone_rects((w, h), zone_cfgs, zone_flags(frame, zone_cfgs))


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

        # Slots AKA/AO here are internal continuity slots A/B only; which one
        # is really AKA is decided offline in identity.py.
        # Optional {tracker_id: [(frame, red, blue), ...]} from pass 1: lets the
        # continuity check look at a candidate's FUTURE evidence too.
        self.final_ev: dict[int, list] | None = None

        # Optional {tracker_id: [(frame, dark_flag), ...]} from pass 1: lets
        # the official verdict use a window centered on the frame (future
        # frames too) instead of only the trailing one.
        self.final_dark_flags: dict[int, list] | None = None

        # Optional {tracker_id: (frames, total_area)} over each track's WHOLE
        # lifetime from pass 1. Offline, a track that will turn out to be long
        # enough is eligible from its first frame, not only once it has been seen
        # min_track_len times (back-fill: no ~10-frame blind spot at track start).
        self.final_track_stats: dict[int, tuple[int, float]] | None = None

        # Optional {tracker_id: [(frame, leg brightness, glove evidence), ...]} from
        # pass 1 plus the leg threshold and the window (frames each side): the
        # lower-body official verdict (official_by_legs). None = off.
        self.final_legs: dict[int, list] | None = None
        self.leg_dark: float = 100.0
        self.leg_window: int = 15

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
            stats.ev.append((frame_idx, red, blue))
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
        window = [d for _, d in frame_window(flags, lo, hi)]
        dark = sum(window)
        if dark >= OFFICIAL_MIN_DARK_FRAMES and dark / len(window) >= OFFICIAL_DARK_FRAC:
            return True
        return self.official_by_legs(track.tracker_id, frame_idx)

    def official_by_legs(self, tracker_id: int, frame_idx: int) -> bool:
        if self.final_legs is None:
            return False
        return official_by_legs(self.final_legs.get(tracker_id, []), frame_idx - self.leg_window,
                                frame_idx + self.leg_window, self.leg_dark)

    def window_score(self, tracker_id: int | None, lo: int, hi: int) -> float | None:
        """(R-B)/(R+B) of a track's evidence over frames [lo, hi]; None if thin."""
        if tracker_id is None:
            return None
        if self.final_ev is not None:
            rows = self.final_ev.get(tracker_id, [])
        else:
            t = self.tracks.get(tracker_id)
            rows = t.ev if t else []
        red = sum(r for f, r, _ in rows if lo <= f <= hi)
        blue = sum(b for f, _, b in rows if lo <= f <= hi)
        return evidence_score(red, blue)

    def track_len_area(self, t: TrackStats) -> tuple[int, float]:
        """(frames, avg area) of a track: whole lifetime when known (offline)."""
        if self.final_track_stats is not None and t.tracker_id in self.final_track_stats:
            n, total = self.final_track_stats[t.tracker_id]
            return n, total / max(n, 1)
        return t.frames_seen, t.avg_area

    def _candidate_tracks(self, frame_idx: int, tatami_mask_fn, allowed_ids: set | None = None) -> list[TrackStats]:
        # Must be visible AND pass the athlete filter THIS frame - a track
        # that disappeared, sat down, or is a suited official right now
        # should not keep winning selection just because it was big/
        # persistent earlier.
        candidates = [
            t for t in self.tracks.values()
            if t.last_seen_frame == frame_idx
            and t.last_passed_filter
            and not self.is_official(t, frame_idx)
            and self.track_len_area(t)[0] >= self.min_track_len
            and self.track_len_area(t)[1] >= self.min_bbox_area
            and (allowed_ids is None or t.tracker_id in allowed_ids)
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

    def select_two(self, frame_idx: int, tatami_mask_fn=None, allowed_ids: set | None = None):
        """allowed_ids: if given, only these tracks may be candidates this frame
        (e.g. the ones standing on the active mat region)."""
        candidates = self._candidate_tracks(frame_idx, tatami_mask_fn, allowed_ids)
        # The current slot holders keep their place in the top two: a newcomer
        # (e.g. a pictogram figure of a broadcast graphic, or a bigger official)
        # must not push out an athlete who is still a valid candidate. Only the
        # remaining places are filled by size.
        holding = {self.slot_to_tracker[AKA], self.slot_to_tracker[AO]} - {None}
        candidates.sort(key=lambda t: (t.tracker_id not in holding, -self.track_len_area(t)[1]))
        self._assign(frame_idx, candidates[:2])
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

            # Color continuity: compare the lost slot's last evidence with each
            # candidate's upcoming evidence (see CONT_* above).
            ref = None
            old_t = self.tracks.get(last_known_tracker_id)
            if old_t is not None:
                ref = self.window_score(last_known_tracker_id, old_t.last_seen_frame - CONT_WINDOW, old_t.last_seen_frame)
            cand_scores = {
                t.tracker_id: self.window_score(t.tracker_id, frame_idx, frame_idx + CONT_WINDOW)
                if self.final_ev is not None else self.window_score(t.tracker_id, frame_idx - CONT_WINDOW, frame_idx)
                for t in unassigned_tracks
            }
            def diff(t):
                sc = cand_scores[t.tracker_id]
                return None if ref is None or sc is None else abs(sc - ref)
            # Color never EXCLUDES a candidate: a slot's reference track can itself
            # have drifted onto the other athlete (clinches / ID jumps), and an
            # exclusion then leaves the slot empty. Color only CONFIRMS a match.
            compatible = list(unassigned_tracks)
            confirmed = [t for t in compatible if diff(t) is not None and diff(t) <= CONT_CONFIRMED]

            best = None
            reason = None
            if confirmed:
                best = self._closest(confirmed, slot)
                reason = f"color continuity (score diff {diff(best):.2f})"
            else:
                candidate = self._closest(compatible, slot)
                dist = (
                    _dist(candidate.last_ground_point, last_point)
                    if last_point is not None and candidate.last_ground_point is not None else None
                )
                if dist is not None and dist > self.reassoc_max_dist:
                    logger.warning(
                        "frame %d: %s has no confident match (nearest candidate "
                        "tracker %s at distance %.1f > max %.1f) - leaving unassigned this frame",
                        frame_idx, slot, candidate.tracker_id, dist, self.reassoc_max_dist,
                    )
                    continue
                best = candidate
                reason = "distance fallback" if last_point is not None else "bootstrap (no prior position)"

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
