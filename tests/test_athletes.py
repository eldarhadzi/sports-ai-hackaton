import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from athletes import AthleteSelector


@dataclass
class FakeDetection:
    tracker_id: int
    bbox: tuple


def make_selector():
    # 1000x1000 frame, tiny thresholds so short synthetic tracks qualify.
    return AthleteSelector(
        min_track_len=2, min_bbox_area_frac=0.001,
        reassoc_max_dist_frac=0.5, frame_width=1000, frame_height=1000,
    )


def test_picks_two_largest_persistent_tracks():
    selector = make_selector()
    # tracker 1 and 2 are large and persistent; tracker 3 is small (referee/audience).
    for frame in range(3):
        dets = [
            FakeDetection(1, (0, 0, 200, 400)),
            FakeDetection(2, (500, 0, 700, 400)),
            FakeDetection(3, (900, 900, 920, 920)),
        ]
        gps = [(100, 400), (600, 400), (910, 920)]
        selector.update_track_stats(frame, dets, gps)
        selector.select_two(frame)

    assert {selector.athlete_to_tracker["athlete_1"], selector.athlete_to_tracker["athlete_2"]} == {1, 2}


def test_reassociation_after_tracker_id_switch():
    selector = make_selector()
    dets = [FakeDetection(1, (0, 0, 200, 400)), FakeDetection(2, (500, 0, 700, 400))]
    gps = [(100, 400), (600, 400)]
    for frame in range(3):
        selector.update_track_stats(frame, dets, gps)
        selector.select_two(frame)
        selector.note_ground_point("athlete_1", (100, 400))
        selector.note_ground_point("athlete_2", (600, 400))

    assert selector.id_switch_count == 0

    # Tracker 1 disappears (occlusion), a new tracker 3 appears near its last
    # position. min_track_len=2, so tracker 3 needs 2 frames before it's a
    # candidate - athlete_1 has a brief (honest) gap in the meantime.
    new_dets = [FakeDetection(2, (500, 0, 700, 400)), FakeDetection(3, (110, 10, 310, 410))]
    new_gps = [(600, 400), (110, 410)]

    # During this gap athlete_1's mapping still points at tracker 1, but that's
    # harmless: tracker 1 has no detection this frame, so nothing gets drawn
    # or exported under it.
    selector.update_track_stats(3, new_dets, new_gps)
    selector.select_two(3)
    assert selector.tracker_to_athlete(3) is None

    selector.update_track_stats(4, new_dets, new_gps)
    selector.select_two(4)
    assert selector.athlete_to_tracker["athlete_1"] == 3
    assert selector.id_switch_count == 1
