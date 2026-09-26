"""Sport-specific movement analysis layer (karate kumite).

Works on the exported per-frame records (JSONL rows already in memory). This
is the layer that becomes sport-specific later (karate, taekwondo, football);
detection/tracking/pose stay shared.
"""
from dataclasses import dataclass, field
import math


@dataclass
class MovementStats:
    athlete_id: str
    space: str  # "pixels" or "meters_approx"
    total_frames: int
    total_distance: float = 0.0
    avg_speed: float = 0.0
    peak_speed: float = 0.0
    direction_changes: int = 0
    frames_visible: int = 0
    positions: list[tuple[float, float, float]] = field(default_factory=list)  # (timestamp, x, y)

    @property
    def time_visible_frac(self) -> float:
        return self.frames_visible / self.total_frames if self.total_frames else 0.0


def compute_movement_stats(
    records: list[dict],
    athlete_id: str,
    total_frames: int,
    direction_change_angle_deg: float,
    min_move_speed: float,
    use_tatami: bool,
) -> MovementStats:
    space = "meters_approx" if use_tatami else "pixels"
    stats = MovementStats(athlete_id=athlete_id, space=space, total_frames=total_frames)

    own = sorted(
        (r for r in records if r["athlete_id"] == athlete_id),
        key=lambda r: r["frame"],
    )

    prev_point = None
    prev_t = None
    prev_tracker_id = None
    prev_heading = None
    speeds = []

    for r in own:
        point = r.get("tatami_point") if use_tatami else r.get("ground_point")
        if point is None:
            continue

        stats.frames_visible += 1
        stats.positions.append((r["timestamp"], point[0], point[1]))

        # A tracker-ID change means the position link to the previous frame is
        # not trustworthy (occlusion/re-association gap) - counting that jump
        # as real movement produces spurious "teleport" speed spikes. Still
        # record the position (above), just don't treat this step as motion.
        tracker_changed = prev_tracker_id is not None and r["tracker_id"] != prev_tracker_id

        if prev_point is not None and not tracker_changed:
            dt = r["timestamp"] - prev_t
            if dt > 0:
                dist = math.dist(point, prev_point)
                speed = dist / dt
                stats.total_distance += dist
                speeds.append(speed)
                stats.peak_speed = max(stats.peak_speed, speed)

                if speed >= min_move_speed:
                    heading = math.atan2(point[1] - prev_point[1], point[0] - prev_point[0])
                    if prev_heading is not None:
                        diff = math.degrees(abs(heading - prev_heading))
                        diff = min(diff, 360 - diff)
                        if diff >= direction_change_angle_deg:
                            stats.direction_changes += 1
                    prev_heading = heading

        prev_point = point
        prev_t = r["timestamp"]
        prev_tracker_id = r["tracker_id"]

    stats.avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    return stats
