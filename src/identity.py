"""Offline AKA/AO identity from the WHOLE clip's color evidence.

Selection (athletes.py) only builds two continuity slots, A and B, from
position/gap/color-continuity - it never decides which is AKA. Tracker IDs get
silently re-used across people (e.g. after a clinch tracker 3 can jump from the
red athlete to the blue one), so evidence cannot be pooled per track over its
lifetime. Instead, every frame contributes a pair signal

    x(t) = (A's red-blue) - (B's red-blue)       > 0: A is redder -> A = AKA

and a two-state Viterbi picks the orientation sequence o(t) in {+1: A is AKA,
-1: A is AO} that maximizes sum o(t)*x(t) minus a cost per switch. That uses
past AND future evidence, fills frames with no evidence of their own from their
neighbours, enforces pair exclusivity (one athlete red => the other is AO), and
places a switch where the evidence actually flips (the clinch), not before.
"""
import math

VALUE_CLIP = 1.0
# A frame where only ONE athlete is held (or only one has color evidence) is
# judged against nothing, so small leaning-red/blue readings (pale gi bleeding
# into the patch) are ignored: only the part of |r-b| above this counts.
SINGLE_DEADZONE = 0.4
# Detections with less total evidence than this count as "no evidence".
NO_EVIDENCE = 0.05
# Evidence units needed to justify flipping the orientation (~16 frames of
# consistent 0.5/frame contradiction).
SWITCH_COST = 16.0
# Side-order prior. Two athletes rarely swap sides without strong reason, and
# an ID jump after a clinch shows up as the A/B left-right order flipping. When
# the order flips between two frames, flipping the labels WITH it keeps "AKA is
# the left one" true (free); keeping the labels means AKA crossed to the other
# side, and flipping them without any order change means an ID swap in place.
# Both of those cost SWITCH_COST (one cost: two different ones can be gamed by
# flipping early at the cheaper price and flipping back free at the order flip).
# A/B order only counts as observed when their box centers are at least this
# many mean-box-widths apart (otherwise they are merged/clinched).
MIN_SEPARATION = 0.4

# A constant-orientation segment is only labeled (not "?") if the evidence in
# it supports its orientation by at least this much.
MIN_SEGMENT_EVIDENCE = 2.0


def det_value(ev: tuple[float, float] | None) -> float | None:
    if ev is None or ev[0] + ev[1] < NO_EVIDENCE:
        return None
    return max(-VALUE_CLIP, min(VALUE_CLIP, ev[0] - ev[1]))


def pair_signal(ev_a, ev_b) -> float:
    """ev_*: (red, blue) of the detection holding slot A / B this frame, or None."""
    va, vb = det_value(ev_a), det_value(ev_b)
    if va is not None and vb is not None:
        return va - vb
    if va is None and vb is None:
        return 0.0
    v, sign = (va, 1.0) if va is not None else (vb, -1.0)
    return sign * math.copysign(max(abs(v) - SINGLE_DEADZONE, 0.0), v)


def side_order(centers: list[tuple[float, float, float] | None]) -> list[int]:
    """centers[t] = (x_A, x_B, mean_box_width) when both slots are held, else
    None. Returns q[t] = +1 if A is left of B, -1 if right, carried forward
    through frames where the order is unobserved (one slot missing, or the two
    boxes merged in a clinch)."""
    q, last = [], 1
    for c in centers:
        if c is not None and abs(c[0] - c[1]) >= MIN_SEPARATION * c[2]:
            last = 1 if c[0] < c[1] else -1
        q.append(last)
    return q


def viterbi_orientation(
    x: list[float], q: list[int] | None = None,
    switch_cost: float = SWITCH_COST,
) -> list[int]:
    """argmax over o in {+1,-1}^T of sum o(t)*x(t) - transition costs, where a
    transition at t costs: nothing if o and q are both unchanged, or both
    flipped (AKA stayed on the same side); switch_cost if exactly one of them
    flipped (AKA changed sides / the labels swapped in place)."""
    n = len(x)
    if n == 0:
        return []
    q = q or [1] * n
    score = {1: x[0], -1: -x[0]}
    back = []
    for t in range(1, n):
        q_flipped = q[t] != q[t - 1]
        new, ptr = {}, {}
        for s in (1, -1):
            stay = score[s] - (switch_cost if q_flipped else 0.0)
            flip = score[-s] - (0.0 if q_flipped else switch_cost)
            new[s], ptr[s] = (stay, s) if stay >= flip else (flip, -s)
            new[s] += s * x[t]
        back.append(ptr)
        score = new
    s = 1 if score[1] >= score[-1] else -1
    path = [s]
    for ptr in reversed(back):
        s = ptr[s]
        path.append(s)
    return path[::-1]


def label_orientation(
    x: list[float], q: list[int] | None = None, switch_cost: float = SWITCH_COST,
):
    """Returns (orientation, known, n_switches). orientation[t] = +1 if slot A
    is AKA at frame t else -1. known[t] = the constant-orientation segment
    containing t has >= MIN_SEGMENT_EVIDENCE evidence behind it."""
    o = viterbi_orientation(x, q, switch_cost)
    known = [False] * len(o)
    switches = 0
    start = 0
    for t in range(1, len(o) + 1):
        if t == len(o) or o[t] != o[start]:
            support = sum(o[i] * x[i] for i in range(start, t))
            for i in range(start, t):
                known[i] = support >= MIN_SEGMENT_EVIDENCE
            if t < len(o):
                switches += 1
            start = t
    return o, known, switches
