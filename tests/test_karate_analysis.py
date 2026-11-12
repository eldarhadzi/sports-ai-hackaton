import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.karate import compute_movement_stats


def make_record(frame, athlete_id, x, y, fps=30.0, tracker_id=1):
    return {
        "frame": frame,
        "timestamp": frame / fps,
        "athlete_id": athlete_id,
        "tracker_id": tracker_id,
        "ground_point": (x, y),
        "tatami_point": None,
    }


def test_straight_line_distance_and_speed():
    records = [make_record(i, "athlete_1", i * 100.0, 0.0) for i in range(5)]
    stats = compute_movement_stats(
        records, "athlete_1", total_frames=5,
        direction_change_angle_deg=30, min_move_speed=1.0, use_tatami=False,
    )
    assert stats.frames_visible == 5
    assert stats.total_distance == 400.0
    assert stats.peak_speed > 0
    assert stats.direction_changes == 0
    assert stats.space == "pixels"


def test_right_angle_turn_counts_as_direction_change():
    records = [
        make_record(0, "athlete_1", 0.0, 0.0),
        make_record(1, "athlete_1", 100.0, 0.0),
        make_record(2, "athlete_1", 100.0, 100.0),
    ]
    stats = compute_movement_stats(
        records, "athlete_1", total_frames=3,
        direction_change_angle_deg=30, min_move_speed=1.0, use_tatami=False,
    )
    assert stats.direction_changes == 1


def test_tracker_id_switch_does_not_inflate_speed():
    records = [
        make_record(0, "athlete_1", 0.0, 0.0, tracker_id=1),
        make_record(1, "athlete_1", 10.0, 0.0, tracker_id=1),
        # Re-association jump to a new tracker id far away - shouldn't count as motion.
        make_record(2, "athlete_1", 5000.0, 5000.0, tracker_id=2),
        make_record(3, "athlete_1", 5010.0, 5000.0, tracker_id=2),
    ]
    stats = compute_movement_stats(
        records, "athlete_1", total_frames=4,
        direction_change_angle_deg=30, min_move_speed=1.0, use_tatami=False,
    )
    assert stats.frames_visible == 4
    assert stats.total_distance == 20.0
    assert stats.peak_speed < 1000


def test_missing_athlete_gives_empty_stats():
    records = [make_record(0, "athlete_1", 0.0, 0.0)]
    stats = compute_movement_stats(
        records, "athlete_2", total_frames=1,
        direction_change_angle_deg=30, min_move_speed=1.0, use_tatami=False,
    )
    assert stats.frames_visible == 0
    assert stats.time_visible_frac == 0.0
