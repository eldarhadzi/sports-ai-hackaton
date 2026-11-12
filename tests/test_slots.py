import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slots import assign_slots, athlete_score  # noqa: E402


def cand(tid, x, score, color=None):
    return {"tid": tid, "gp": (x, 100.0), "score": score, "color": color}


def test_picks_the_two_best_candidates_every_frame():
    frames = [[cand(1, 100, 0.9), cand(2, 300, 0.8), cand(9, 200, 0.1)] for _ in range(20)]
    out = assign_slots(frames, diag=1000)
    assert all(set(p) == {1, 2} for p in out)


def test_holder_that_stops_being_a_candidate_loses_the_slot_that_frame():
    frames = [[cand(1, 100, 0.9), cand(2, 300, 0.8)] for _ in range(10)]
    frames[5] = [cand(2, 300, 0.8)]  # track 1 failed the filter at frame 5
    out = assign_slots(frames, diag=1000)
    assert 1 not in out[5] and 2 in out[5]
    assert set(out[6]) == {1, 2}


def test_early_wrong_holder_is_replaced_globally():
    # Referee (9) is the only other candidate at first; athlete 2 appears at frame 3
    # and stays: the sticky selector kept 9, the global one hands the slot to 2.
    frames = [[cand(1, 100, 0.9), cand(9, 200, 0.1)] for _ in range(3)]
    frames += [[cand(1, 100, 0.9), cand(9, 200, 0.1), cand(2, 300, 0.85)] for _ in range(40)]
    out = assign_slots(frames, diag=1000)
    assert set(out[-1]) == {1, 2}
    assert set(out[3]) == {1, 2}


def test_no_jitter_between_equal_candidates():
    frames = [[cand(1, 100, 0.8), cand(2, 300, 0.8), cand(3, 500, 0.79 + 0.02 * (t % 2))] for t in range(30)]
    out = assign_slots(frames, diag=1000)
    assert len(set(out)) == 1


def test_pair_bonus_prefers_red_with_blue():
    frames = [[cand(1, 100, 0.7, color=0.8), cand(2, 300, 0.75, color=0.6), cand(3, 500, 0.7, color=-0.7)] for _ in range(10)]
    out = assign_slots(frames, diag=1000)
    assert all(3 in p for p in out)


def test_tracker_handoff_keeps_the_slot_side():
    # Track 2 ends, track 7 starts where 2 was: B should continue on 7, A stays on 1.
    frames = [[cand(1, 100, 0.9), cand(2, 300, 0.9)] for _ in range(10)]
    frames += [[cand(1, 100, 0.9), cand(7, 305, 0.9)] for _ in range(10)]
    out = assign_slots(frames, diag=1000)
    a0 = out[0].index(1)
    assert out[-1][a0] == 1 and out[-1][1 - a0] == 7


def test_athlete_score_orders_white_gi_over_dark_trousers():
    gi = athlete_score(170, 0.3, 1.0, leg_dark=60, leg_white=150)
    ref = athlete_score(45, 0.0, 1.0, leg_dark=60, leg_white=150)
    assert gi > ref
    assert athlete_score(None, None, 0.5, 60, 150) == 0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5
