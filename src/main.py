"""Entry point.

    python src/main.py --input videos/input/match.mp4 --output videos/output/match_analyzed.mp4
"""
import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import yaml

from pose_tracker import PoseTracker, LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX
from athletes import AthleteSelector
from calibration import calibrate_from_click, load_calibration, save_calibration
from analysis.karate import compute_movement_stats
from visualization import draw_frame, render_heatmap
from export import JsonlWriter, write_run_meta

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
    return parser.parse_args()


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
    data_dir = Path(config["export"]["data_dir"])

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

    tracker = PoseTracker(config["model"]["name"], config.get("device", "auto"), config.get("tracker", "bytetrack.yaml"))

    selector = AthleteSelector(
        min_track_len=config["tracking"]["min_track_len"],
        min_bbox_area_frac=config["tracking"]["min_bbox_area_frac"],
        reassoc_max_dist_frac=config["athlete_selection"]["reassoc_max_dist_frac"],
        frame_width=width, frame_height=height,
    )

    ankle_conf_threshold = config["ground_point"]["ankle_conf_threshold"]
    smoothing_window = config["ground_point"]["smoothing_window"]
    trail_length = config["visualization"]["trail_length"]

    ground_point_history = {"athlete_1": [], "athlete_2": []}
    trails = {"athlete_1": [], "athlete_2": []}

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

        selector.update_track_stats(frame_idx, detections, raw_ground_points)
        selector.select_two(frame_idx, tatami_mask_fn=tatami_mask_fn)

        timestamp = frame_idx / fps

        for det, raw_gp in zip(detections, raw_ground_points):
            athlete_id = selector.tracker_to_athlete(det.tracker_id)
            smoothed_gp = raw_gp
            tatami_point = None

            if athlete_id is not None and raw_gp is not None:
                history = ground_point_history[athlete_id]
                history.append(raw_gp)
                del history[: max(0, len(history) - smoothing_window)]
                smoothed_gp = moving_average_point(history)
                selector.note_ground_point(athlete_id, smoothed_gp)

                if calibration is not None:
                    tatami_point = calibration.to_tatami(smoothed_gp)

                trail = trails[athlete_id]
                trail.append(smoothed_gp)
                del trail[: max(0, len(trail) - trail_length)]

            record = {
                "frame": frame_idx,
                "timestamp": timestamp,
                "athlete_id": athlete_id,
                "tracker_id": det.tracker_id,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "keypoints": det.keypoints,
                "ground_point": smoothed_gp,
                "tatami_point": tatami_point,
            }
            jsonl.write(record)
            if athlete_id is not None:
                records.append(record)

        annotated = draw_frame(frame, detections, selector, trails, config["visualization"])
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
        "reassociation_log": selector.reassociation_log,
        "calibrated": calibration is not None,
    })

    logger.info(
        "Done: %d frames in %.1fs (%.1f fps effective). ID switches: %d",
        frame_idx, elapsed, effective_fps, selector.id_switch_count,
    )

    stats_summary = {}
    for athlete_id in ("athlete_1", "athlete_2"):
        min_move_speed = (
            config["movement"]["min_move_speed_mps"] if calibration is not None
            else config["movement"]["min_move_speed_px"]
        )
        stats = compute_movement_stats(
            records, athlete_id, frame_idx,
            config["movement"]["direction_change_angle_deg"],
            min_move_speed,
            use_tatami=calibration is not None,
        )
        stats_summary[athlete_id] = {
            "space": stats.space,
            "total_distance": stats.total_distance,
            "avg_speed": stats.avg_speed,
            "peak_speed": stats.peak_speed,
            "direction_changes": stats.direction_changes,
            "time_visible_frac": stats.time_visible_frac,
        }
        logger.info(
            "%s [%s]: distance=%.1f avg_speed=%.2f peak_speed=%.2f direction_changes=%d visible=%.0f%%",
            athlete_id, stats.space, stats.total_distance, stats.avg_speed,
            stats.peak_speed, stats.direction_changes, stats.time_visible_frac * 100,
        )
        render_heatmap(
            stats.positions,
            data_dir / f"{input_path.stem}_{athlete_id}_heatmap.png",
            calibration=calibration,
            frame_for_overlay=last_frame if calibration is None else None,
            tatami_size=(calib_cfg["tatami_width_m"], calib_cfg["tatami_height_m"]) if calibration else None,
        )

    (data_dir / f"{input_path.stem}_stats.json").write_text(json.dumps(stats_summary, indent=2))


if __name__ == "__main__":
    main()
