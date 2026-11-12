"""Entry point.

    python src/main.py --input videos/input/match.mp4 --output videos/output/match_analyzed.mp4
"""
import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import cv2
import yaml

from pose_tracker import PoseTracker, LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX
from athletes import AKA, AO, AthleteSelector, classify_belt_color, passes_athlete_filter
from calibration import calibrate_from_click, load_calibration, save_calibration
from analysis.karate import compute_movement_stats
from analysis.metrics import compute_clip_metrics
from visualization import draw_frame, render_heatmap, save_color_debug_strip
from export import JsonlWriter, write_run_meta

MAX_COLOR_DEBUG_FRAMES = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def moving_average_point(history: list[tuple[float, float]]):
    if not history:
        return None
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def ground_point_from_detection(det, ankle_conf_threshold: float):
    left = det.keypoints[LEFT_ANKLE_IDX]
    right = det.keypoints[RIGHT_ANKLE_IDX]
    left_ok = left[2] >= ankle_conf_threshold
    right_ok = right[2] >= ankle_conf_threshold

    if left_ok and right_ok:
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    if left_ok:
        return (left[0], left[1])
    if right_ok:
        return (right[0], right[1])

    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, y2)


def append_trail_point(trail: list, timestamp: float, point, break_dist: float):
    """trail is a list of (timestamp, point-or-None). A None point marks an
    explicit break (occlusion/gap or too-large a jump) so the renderer never
    draws a connector across it. Called once per frame per slot, even when
    the athlete isn't currently detected (point=None), so a real gap in time
    is always recorded - not just a gap in position."""
    if point is not None:
        last_real = next((p for _, p in reversed(trail) if p is not None), None)
        if last_real is not None:
            dist = ((point[0] - last_real[0]) ** 2 + (point[1] - last_real[1]) ** 2) ** 0.5
            if dist > break_dist and (not trail or trail[-1][1] is not None):
                trail.append((timestamp, None))
    trail.append((timestamp, point))


def trim_trail(trail: list, now: float, trail_seconds: float):
    cutoff = now - trail_seconds
    while trail and trail[0][0] < cutoff:
        trail.pop(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run interactive tatami corner calibration on the first frame before processing.",
    )
    parser.add_argument(
        "--corners", default=None,
        help=(
            "Skip clicking: 'x1,y1,x2,y2,x3,y3,x4,y4' tatami corners in pixel "
            "coordinates, order top-left,top-right,bottom-right,bottom-left."
        ),
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip heatmap rendering (e.g. for a moving/panning camera where positions aren't meaningful).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override export.data_dir (jsonl/metrics/run_meta/debug images). Calibration is read from <data-dir>/calibration.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save belt/glove patch debug strips to <data-dir>/debug_color/.",
    )
    return parser.parse_args()


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(
            ["git", *cmd], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.rstrip("\n")
    except Exception:
        return "unknown"


def build_provenance(config: dict, args, device: str) -> dict:
    """What code/config actually produced this run. 'active_features' is read
    from live config/args, not hardcoded, so it can't claim something that
    isn't on."""
    dirty_files = [l[3:] for l in _git("status", "--porcelain").splitlines() if l]
    diff_hash = _git("diff", "HEAD", "--", "src", "config.yaml")
    commit = _git("rev-parse", "--short", "HEAD")
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_files),
        "git_dirty_files": dirty_files,
        "src_config_diff_sha1": __import__("hashlib").sha1(diff_hash.encode()).hexdigest()[:12],
        "model": config["model"]["name"],
        "imgsz": config["model"].get("imgsz", 640),
        "conf": config["model"].get("conf", 0.25),
        "device": device,
        "active_features": {
            "hsv_conversion": "cv2.COLOR_BGR2HSV (unchanged since HEAD; no separate fix exists)",
            "ignore_zones": config["classifier"].get("ignore_zones", []),
            "trail_breaking_px": config["visualization"]["trail_break_dist_px"],
            "debug_rendering": bool(args.debug),
        },
    }


def parse_corners(spec: str) -> list[tuple[float, float]]:
    values = [float(v) for v in spec.split(",")]
    if len(values) != 8:
        raise ValueError("--corners needs exactly 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return [(values[i], values[i + 1]) for i in range(0, 8, 2)]


def make_tatami_mask_fn(calibration, tatami_width_m, tatami_height_m):
    margin_m = 0.5

    def inside(point):
        tp = calibration.to_tatami(point)
        if tp is None:
            return False
        return -margin_m <= tp[0] <= tatami_width_m + margin_m and -margin_m <= tp[1] <= tatami_height_m + margin_m

    return inside


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir or config["export"]["data_dir"])

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Input: %s (%dx%d @ %.2f fps, ~%d frames)", input_path, width, height, fps, frame_count_hint)

    calib_cfg = config["calibration"]
    calib_path = data_dir / "calibration" / f"{input_path.stem}.json"
    calibration = load_calibration(calib_path)

    if args.corners and calibration is None:
        calibration = save_calibration(
            parse_corners(args.corners), calib_path,
            calib_cfg["tatami_width_m"], calib_cfg["tatami_height_m"],
        )
    elif args.calibrate or (calibration is None and calib_cfg.get("interactive_if_missing", False)):
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read first frame for calibration")
        calibration = calibrate_from_click(
            first_frame, calib_path, calib_cfg["tatami_width_m"], calib_cfg["tatami_height_m"],
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = PoseTracker(
        config["model"]["name"], config.get("device", "auto"),
        config.get("tracker", "bytetrack.yaml"), config["model"].get("conf", 0.25),
        config["model"].get("imgsz", 640),
    )

    provenance = build_provenance(config, args, tracker.device)
    logger.info("RUN PROVENANCE: %s", json.dumps(provenance, indent=2))

    selector = AthleteSelector(
        min_track_len=config["tracking"]["min_track_len"],
        min_bbox_area_frac=config["tracking"]["min_bbox_area_frac"],
        reassoc_max_dist_frac=config["athlete_selection"]["reassoc_max_dist_frac"],
        frame_width=width, frame_height=height,
    )

    classifier_cfg = config["classifier"]
    ignore_zones = [tuple(z) for z in classifier_cfg.get("ignore_zones", [])]
    dark_brightness_threshold = classifier_cfg["dark_brightness_threshold"]

    ankle_conf_threshold = config["ground_point"]["ankle_conf_threshold"]
    smoothing_window = config["ground_point"]["smoothing_window"]
    trail_seconds = config["visualization"]["trail_seconds"]
    trail_break_dist = config["visualization"]["trail_break_dist_px"]

    ground_point_history = {AKA: [], AO: []}
    trails = {AKA: [], AO: []}  # each: list of (timestamp, point-or-None)
    color_debug_dir = data_dir / "debug_color"
    color_debug_counts = {AKA: 0, AO: 0}

    tatami_mask_fn = (
        make_tatami_mask_fn(calibration, calib_cfg["tatami_width_m"], calib_cfg["tatami_height_m"])
        if calibration is not None else None
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    jsonl_path = data_dir / f"{input_path.stem}.jsonl"
    jsonl = JsonlWriter(jsonl_path)

    records = []
    frame_idx = 0
    last_frame = None
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame

        detections = tracker.track(frame)
        raw_ground_points = [ground_point_from_detection(d, ankle_conf_threshold) for d in detections]
        filter_results = [
            passes_athlete_filter(frame, d.bbox, d.keypoints, ignore_zones, dark_brightness_threshold)
            for d in detections
        ]
        color_results = [classify_belt_color(frame, d.bbox, d.keypoints) for d in detections]
        colors = [c for c, _ in color_results]

        selector.update_track_stats(frame_idx, detections, raw_ground_points, filter_results, colors)
        selector.select_two(frame_idx, tatami_mask_fn=tatami_mask_fn)

        timestamp = frame_idx / fps
        visible_slots = set()
        slot_points = {AKA: None, AO: None}

        for det, raw_gp, (passed, reject_reason), (color, color_samples) in zip(
            detections, raw_ground_points, filter_results, color_results,
        ):
            slot = selector.tracker_to_slot(det.tracker_id)
            smoothed_gp = raw_gp
            tatami_point = None

            if slot is not None and raw_gp is not None:
                visible_slots.add(slot)
                history = ground_point_history[slot]
                history.append(raw_gp)
                del history[: max(0, len(history) - smoothing_window)]
                smoothed_gp = moving_average_point(history)
                slot_points[slot] = smoothed_gp
                selector.note_ground_point(slot, smoothed_gp)

                if calibration is not None:
                    tatami_point = calibration.to_tatami(smoothed_gp)

                if args.debug and color_debug_counts[slot] < MAX_COLOR_DEBUG_FRAMES and any(
                    s["color"] != "no_keypoint" for s in color_samples
                ):
                    save_color_debug_strip(
                        frame, det.bbox, color_samples, color,
                        color_debug_dir / f"{input_path.stem}_{slot}_f{frame_idx:05d}.png",
                    )
                    color_debug_counts[slot] += 1

            record = {
                "frame": frame_idx,
                "timestamp": timestamp,
                "athlete_id": slot,
                "tracker_id": det.tracker_id,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "keypoints": det.keypoints,
                "ground_point": smoothed_gp,
                "tatami_point": tatami_point,
                "belt_color": color,
                "passed_athlete_filter": passed,
                "reject_reason": reject_reason,
            }
            jsonl.write(record)
            if slot is not None:
                records.append(record)

        for slot in (AKA, AO):
            append_trail_point(trails[slot], timestamp, slot_points[slot], trail_break_dist)
            trim_trail(trails[slot], timestamp, trail_seconds)

        annotated = draw_frame(frame, detections, selector, trails, visible_slots, config["visualization"])
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info("Processed %d frames", frame_idx)

    cap.release()
    writer.release()
    jsonl.close()

    elapsed = time.time() - start_time
    effective_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    write_run_meta(data_dir / f"{input_path.stem}_run_meta.json", {
        "provenance": provenance,
        "model": config["model"]["name"],
        "tracker": config.get("tracker", "bytetrack.yaml"),
        "device": tracker.device,
        "config_snapshot": config,
        "input_fps": fps,
        "input_resolution": [width, height],
        "frames_processed": frame_idx,
        "processing_time_sec": elapsed,
        "effective_fps": effective_fps,
        "id_switch_count": selector.id_switch_count,
        "color_swap_count": selector.color_swap_count,
        "reassociation_log": selector.reassociation_log,
        "calibrated": calibration is not None,
    })

    logger.info(
        "Done: %d frames in %.1fs (%.1f fps effective). ID switches: %d, color swaps: %d",
        frame_idx, elapsed, effective_fps, selector.id_switch_count, selector.color_swap_count,
    )

    metrics = compute_clip_metrics(jsonl_path, frame_idx, selector.id_switch_count, selector.color_swap_count)
    (data_dir / f"{input_path.stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Metrics: both=%.1f%% one=%.1f%% zero=%.1f%% id_switches=%d color_swaps=%d filter_violations=%d",
        metrics["pct_both_assigned"], metrics["pct_one_assigned"], metrics["pct_zero_assigned"],
        metrics["id_switch_count"], metrics["color_swap_count"], metrics["filter_violations"],
    )

    stats_summary = {}
    for slot in (AKA, AO):
        min_move_speed = (
            config["movement"]["min_move_speed_mps"] if calibration is not None
            else config["movement"]["min_move_speed_px"]
        )
        stats = compute_movement_stats(
            records, slot, frame_idx,
            config["movement"]["direction_change_angle_deg"],
            min_move_speed,
            use_tatami=calibration is not None,
        )
        stats_summary[slot] = {
            "space": stats.space,
            "total_distance": stats.total_distance,
            "avg_speed": stats.avg_speed,
            "peak_speed": stats.peak_speed,
            "direction_changes": stats.direction_changes,
            "time_visible_frac": stats.time_visible_frac,
        }
        logger.info(
            "%s [%s]: distance=%.1f avg_speed=%.2f peak_speed=%.2f direction_changes=%d visible=%.0f%%",
            slot, stats.space, stats.total_distance, stats.avg_speed,
            stats.peak_speed, stats.direction_changes, stats.time_visible_frac * 100,
        )
        if not args.no_heatmap:
            render_heatmap(
                stats.positions,
                data_dir / f"{input_path.stem}_{slot}_heatmap.png",
                calibration=calibration,
                frame_for_overlay=last_frame if calibration is None else None,
                tatami_size=(calib_cfg["tatami_width_m"], calib_cfg["tatami_height_m"]) if calibration else None,
            )

    (data_dir / f"{input_path.stem}_stats.json").write_text(json.dumps(stats_summary, indent=2))


if __name__ == "__main__":
    main()
