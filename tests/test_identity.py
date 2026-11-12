import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import label_orientation, pair_signal, side_order  # noqa: E402

RED, BLUE, NONE = (0.8, 0.0), (0.0, 0.8), (0.0, 0.0)


def test_pair_signal_is_relative_and_exclusive():
    assert pair_signal(RED, BLUE) > 1.0
    assert pair_signal(BLUE, RED) < -1.0
    # both weakly red: only the DIFFERENCE counts
    assert abs(pair_signal((0.5, 0.2), (0.45, 0.2))) < 0.1
    # one athlete has no evidence: the other's strong reading still decides
    assert pair_signal(RED, NONE) > 0 and pair_signal(NONE, BLUE) > 0
    # lone athlete with only a weak lean: ignored
    assert pair_signal((0.3, 0.1), None) == 0.0


def test_evidence_after_the_gap_labels_the_frames_before_it():
    x = [0.0] * 40 + [1.0] * 60                    # no evidence for the first 40 frames
    o, known, sw = label_orientation(x)
    assert o == [1] * 100 and all(known) and sw == 0


def test_no_evidence_anywhere_is_unknown():
    o, known, sw = label_orientation([0.0] * 50)
    assert not any(known)


def test_identity_switch_is_placed_where_evidence_flips_and_noise_is_ignored():
    x = [1.0] * 100 + [-1.0] * 100
    x[30], x[31], x[32] = -1.0, -1.0, -1.0         # short contradicting noise
    o, known, sw = label_orientation(x)
    assert sw == 1 and o[:100] == [1] * 100 and o[100:] == [-1] * 100 and all(known)


def test_side_prior_follows_an_id_jump_that_flips_the_left_right_order():
    # A is AKA and on the left for 100 frames; then the ID jumps (A is now on
    # the right) while color evidence is weakly and misleadingly unchanged.
    x = [1.0] * 100 + [0.05] * 90
    q = [1] * 100 + [-1] * 90
    o_free, _, _ = label_orientation(x)
    o_side, known, sw = label_orientation(x, q)
    assert o_free[-1] == 1                       # color alone keeps A = AKA (wrong here)
    assert o_side[:100] == [1] * 100 and o_side[100:] == [-1] * 90 and sw == 1


def test_strong_color_evidence_beats_the_side_prior_when_athletes_really_cross():
    x = [1.0] * 100 + [1.0] * 90                 # A stays clearly red although sides swap
    q = [1] * 100 + [-1] * 90
    o, _, sw = label_orientation(x, q)
    assert o == [1] * 190 and sw == 0


def test_side_order_is_carried_through_merged_frames():
    c = [(10, 50, 20), None, (30, 32, 20), (60, 20, 20)]
    assert side_order(c) == [1, 1, 1, -1]
