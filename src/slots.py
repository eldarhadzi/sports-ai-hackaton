"""Offline global slot assignment (round v8, step 3).

The old selector (athletes.AthleteSelector.select_two) is greedy and sticky: it
fills the two slots frame by frame and keeps a holder for as long as it stays a
valid candidate, so a referee or coach that once got a slot keeps it. Offline we
can instead choose, for the WHOLE clip at once, which two candidates hold the
continuity slots A and B in every frame:

    maximize  sum_t [ fill reward of A(t), B(t) + pair bonus(A(t), B(t)) ]
              - sum_t transition cost(state(t-1) -> state(t))

with a Viterbi over states (a, b), a/b in {candidate tracks of frame t} + {empty}.
Only candidates of frame t can hold a slot at t, so a holder that fails the
athlete filter (or leaves the mat, or is judged an official) loses it that frame.

Fill reward  = FILL_BASE + candidate athlete score (0..1, see athlete_score()).
Pair bonus   = one holder's windowed color reads red and the other's blue.
Transitions, per slot:
    same track                         0
    empty <-> a track                  GAP_COST
    track X -> track Y, X still a candidate at t (a real steal)   STEAL_COST
    track X -> track Y, X gone at t (tracker hand-off)            HANDOFF_COST + distance term
Which slot is AKA is still decided afterwards by identity.py from the color
evidence of whoever holds A and B.
"""
import math

import numpy as np

FILL_BASE = 1.0
PAIR_BONUS = 0.5
PAIR_MIN_SCORE = 0.2      # |(R-B)/(R+B)| each holder needs for the pair bonus
GAP_COST = 0.5
STEAL_COST = 4.0
HANDOFF_COST = 0.5
HANDOFF_DIST_COST = 4.0   # per frame-diagonal of distance between old and new ground point
MAX_CANDIDATES = 5        # per frame, by athlete score (keeps the state space ~ 31)


def athlete_score(leg_brightness: float | None, glove_frac: float | None, rel_size: float,
                  leg_dark: float, leg_white: float) -> float:
    """0..1. Lower-body brightness (white gi trousers vs dark trousers) is the main
    term; glove color (strong red/blue at the wrists) and relative box size add to
    it. Unknown terms count as neutral (0.5)."""
    leg = 0.5 if leg_brightness is None else min(1.0, max(0.0, (leg_brightness - leg_dark) / (leg_white - leg_dark)))
    glove = 0.5 if glove_frac is None else min(1.0, glove_frac / 0.5)
    return 0.5 * leg + 0.3 * glove + 0.2 * min(1.0, max(0.0, rel_size))


def assign_slots(frames: list[list[dict]], diag: float) -> list[tuple[int | None, int | None]]:
    """frames[t] = candidates of frame t, each {"tid", "gp" (x, y) or None,
    "score" (0..1), "color" ((R-B)/(R+B) or None)}. Returns per frame the
    (slot A tracker id, slot B tracker id), None = empty."""
    T = len(frames)
    if T == 0:
        return []
    cands = [sorted(f, key=lambda c: -c["score"])[:MAX_CANDIDATES] for f in frames]
    states = [_states(len(c)) for c in cands]

    score = np.array([_emission(cands[0], s) for s in states[0]])
    back = []
    for t in range(1, T):
        prev_c, cur_c = cands[t - 1], cands[t]
        cur_ids = {c["tid"] for c in cur_c}
        slot_cost = _slot_transition(prev_c, cur_c, cur_ids, diag)  # [n_prev+1, n_cur+1]
        ps, cs = states[t - 1], states[t]
        pa = np.array([s[0] for s in ps]); pb = np.array([s[1] for s in ps])
        ca = np.array([s[0] for s in cs]); cb = np.array([s[1] for s in cs])
        trans = slot_cost[pa[:, None], ca[None, :]] + slot_cost[pb[:, None], cb[None, :]]
        total = score[:, None] - trans
        best_prev = total.argmax(axis=0)
        score = total[best_prev, np.arange(len(cs))] + np.array([_emission(cur_c, s) for s in cs])
        back.append(best_prev)

    k = int(score.argmax())
    path = [k]
    for bp in reversed(back):
        k = int(bp[k])
        path.append(k)
    path.reverse()

    out = []
    for t, k in enumerate(path):
        a, b = states[t][k]
        n = len(cands[t])
        out.append((cands[t][a]["tid"] if a < n else None, cands[t][b]["tid"] if b < n else None))
    return out


def _states(n: int) -> list[tuple[int, int]]:
    """Ordered pairs over n candidates + empty (index n); a track holds at most one slot."""
    return [(a, b) for a in range(n + 1) for b in range(n + 1) if a != b or a == n]


def _emission(cands: list[dict], state: tuple[int, int]) -> float:
    n = len(cands)
    held = [cands[i] for i in state if i < n]
    reward = sum(FILL_BASE + c["score"] for c in held)
    if len(held) == 2:
        ca, cb = held[0]["color"], held[1]["color"]
        if ca is not None and cb is not None and abs(ca) >= PAIR_MIN_SCORE and abs(cb) >= PAIR_MIN_SCORE and ca * cb < 0:
            reward += PAIR_BONUS
    return reward


def _slot_transition(prev_c: list[dict], cur_c: list[dict], cur_ids: set, diag: float) -> np.ndarray:
    """Cost of one slot going from prev index i (n_prev = empty) to cur index j (n_cur = empty)."""
    np_, nc = len(prev_c), len(cur_c)
    cost = np.zeros((np_ + 1, nc + 1))
    for i in range(np_ + 1):
        for j in range(nc + 1):
            if i == np_ and j == nc:
                continue
            if i == np_ or j == nc:
                cost[i, j] = GAP_COST
                continue
            p, c = prev_c[i], cur_c[j]
            if p["tid"] == c["tid"]:
                continue
            if p["tid"] in cur_ids:
                cost[i, j] = STEAL_COST
            else:
                d = math.dist(p["gp"], c["gp"]) / diag if p["gp"] is not None and c["gp"] is not None else 0.5
                cost[i, j] = HANDOFF_COST + HANDOFF_DIST_COST * d
    return cost
