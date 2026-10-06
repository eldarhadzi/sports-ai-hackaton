import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from athletes import AKA, AO, AthleteSelector, is_seated, patch_pixel_evidence


@dataclass
class FakeDetection:
    tracker_id: int
    bbox: tuple


def make_selector(frame_width=1000, frame_height=1000):
    # Tiny thresholds so short synthetic tracks qualify.
    return AthleteSelector(
        min_track_len=2, min_bbox_area_frac=0.001,
        reassoc_max_dist_frac=0.5, frame_width=frame_width, frame_height=frame_height,
    )


PASS = (True, None)
FAIL_SEATED = (False, "seated")


def feed(selector, frame_idx, dets, gps, filters, colors):
    """Mimics one frame of main.py's loop: update stats, select, then record
    each now-assigned slot's ground point (so later distance-based
    re-association has something real to compare against)."""
    evidence = [{"red": (1.0, 0.0), "blue": (0.0, 1.0)}.get(c, c if isinstance(c, tuple) else (0.0, 0.0)) for c in colors]
    selector.update_track_stats(frame_idx, dets, gps, filters, evidence)
    selector.select_two(frame_idx)
    for det, gp in zip(dets, gps):
        slot = selector.tracker_to_slot(det.tracker_id)
        if slot is not None:
            selector.note_ground_point(slot, gp)


def test_picks_two_largest_persistent_tracks():
    selector = make_selector()
    for frame in range(3):
        dets = [
            FakeDetection(1, (0, 0, 200, 400)),
            FakeDetection(2, (500, 0, 700, 400)),
            FakeDetection(3, (900, 900, 920, 920)),
        ]
        gps = [(100, 400), (600, 400), (910, 920)]
        feed(selector, frame, dets, gps, [PASS, PASS, PASS], ["unknown"] * 3)

    assert {selector.slot_to_tracker[AKA], selector.slot_to_tracker[AO]} == {1, 2}


def test_seated_person_never_gets_a_slot():
    selector = make_selector()
    for frame in range(5):
        dets = [
            FakeDetection(1, (0, 0, 200, 400)),
            FakeDetection(2, (500, 0, 700, 400)),  # seated coach, same size as athlete 1
        ]
        gps = [(100, 400), (600, 400)]
        feed(selector, frame, dets, gps, [PASS, FAIL_SEATED], ["unknown", "unknown"])

    assert selector.slot_to_tracker[AO] is None
    assert selector.tracker_to_slot(2) is None


def test_color_match_overrides_distance_gate():
    selector = make_selector()
    for frame in range(5):
        dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
        gps = [(100, 400), (600, 400)]
        feed(selector, frame, dets, gps, [PASS, PASS], ["red", "blue"])

    assert selector.slot_to_tracker[AKA] == 1
    assert selector.slot_last_point[AKA] == (100, 400)

    # Tracker 1 disappears; a new tracker 5 appears FAR away (further than
    # reassoc_max_dist from AKA's last known spot) but confidently reads red.
    # Without color, this would be rejected by the distance gate and AKA
    # would sit unassigned; with color it should be picked up right away.
    dets = [FakeDetection(2, (500, 0, 700, 400)), FakeDetection(5, (900, 900, 999, 999))]
    gps = [(600, 400), (950, 950)]
    for frame in range(5, 9):
        feed(selector, frame, dets, gps, [PASS, PASS], ["blue", "red"])

    assert selector.slot_to_tracker[AKA] == 5


def test_color_swap_correction():
    selector = make_selector()
    # AKA and AO get assigned backwards at bootstrap (no color info yet).
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(3):
        feed(selector, frame, dets, gps, [PASS, PASS], ["unknown", "unknown"])
    assert selector.slot_to_tracker[AKA] == 1

    # Now color evidence accumulates showing tracker 1 is actually blue and
    # tracker 2 is actually red - selector should swap them.
    for frame in range(3, 10):
        feed(selector, frame, dets, gps, [PASS, PASS], ["blue", "red"])

    assert selector.slot_to_tracker[AKA] == 2
    assert selector.slot_to_tracker[AO] == 1
    assert selector.color_swap_count == 1


def test_is_seated_heuristic():
    def kp(x, y, c=1.0):
        return (x, y, c)

    # shoulder y=50, hip y=100, knee y=160 -> thigh(60) > 0.35*torso(50): standing
    standing = [kp(0, 0)] * 5 + [kp(50, 50), kp(60, 50)] + [kp(0, 0)] * 4 + [kp(50, 100), kp(60, 100)] + [kp(50, 160), kp(60, 160)] + [kp(0, 0), kp(0, 0)]
    assert is_seated(standing) is False

    # shoulder y=50, hip y=100, knee y=105 -> thigh(5) << 0.35*torso(50): seated
    seated = [kp(0, 0)] * 5 + [kp(50, 50), kp(60, 50)] + [kp(0, 0)] * 4 + [kp(50, 100), kp(60, 100)] + [kp(50, 105), kp(60, 105)] + [kp(0, 0), kp(0, 0)]
    assert is_seated(seated) is True


def test_reassociation_falls_back_to_distance_when_color_unknown():
    selector = make_selector()
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(3):
        feed(selector, frame, dets, gps, [PASS, PASS], ["unknown", "unknown"])
    assert selector.slot_to_tracker[AKA] == 1

    # Tracker 1 disappears; tracker 3 appears close to its last spot, no color read.
    dets2 = [FakeDetection(2, (500, 0, 700, 400)), FakeDetection(3, (110, 10, 310, 410))]
    gps2 = [(600, 400), (110, 410)]
    feed(selector, 3, dets2, gps2, [PASS, PASS], ["unknown", "unknown"])
    feed(selector, 4, dets2, gps2, [PASS, PASS], ["unknown", "unknown"])

    assert selector.slot_to_tracker[AKA] == 3
    assert selector.id_switch_count == 1


def test_far_unknown_color_candidate_is_left_unassigned():
    selector = make_selector()
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(3):
        feed(selector, frame, dets, gps, [PASS, PASS], ["unknown", "unknown"])
    assert selector.slot_to_tracker[AKA] == 1

    # Tracker 1 disappears; tracker 9 appears far away with unknown color -
    # should NOT be force-assigned.
    dets2 = [FakeDetection(2, (500, 0, 700, 400)), FakeDetection(9, (900, 900, 999, 999))]
    gps2 = [(600, 400), (950, 950)]
    feed(selector, 3, dets2, gps2, [PASS, PASS], ["unknown", "unknown"])
    feed(selector, 4, dets2, gps2, [PASS, PASS], ["unknown", "unknown"])

    assert selector.slot_to_tracker[AKA] is None
    assert selector.tracker_to_slot(9) is None


def test_pair_decision_is_relative_not_absolute():
    # Both athletes have only WEAK, mostly-blue evidence (dark belts at 360p),
    # but tracker 2 is clearly less blue than tracker 1 -> tracker 2 is AKA.
    selector = make_selector()
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(10):
        feed(selector, frame, dets, gps, [PASS, PASS], [(0.05, 0.9), (0.4, 0.5)])
    assert selector.slot_to_tracker[AKA] == 2
    assert selector.identity_known == {AKA: True, AO: True}


def test_no_evidence_means_identity_unknown_not_a_position_guess():
    selector = make_selector()
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(5):
        feed(selector, frame, dets, gps, [PASS, PASS], ["unknown", "unknown"])
    assert selector.identity_known == {AKA: False, AO: False}


def test_single_athlete_with_evidence_fixes_both_by_elimination():
    selector = make_selector()
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(8):
        feed(selector, frame, dets, gps, [PASS, PASS], ["blue", "unknown"])
    assert selector.slot_to_tracker[AO] == 1 and selector.slot_to_tracker[AKA] == 2
    assert selector.identity_known == {AKA: True, AO: True}


def test_dark_navy_patch_counts_as_weak_blue_and_red_belt_as_red():
    navy = np.full((12, 12, 3), (70, 30, 35), dtype=np.uint8)      # BGR dark navy/purple
    red = np.full((12, 12, 3), (30, 30, 200), dtype=np.uint8)
    white = np.full((12, 12, 3), 235, dtype=np.uint8)
    r, b = patch_pixel_evidence(navy)
    assert r == 0 and b > 0
    r, b = patch_pixel_evidence(red)
    assert r == 1.0 and b == 0
    assert patch_pixel_evidence(white) == (0.0, 0.0)
