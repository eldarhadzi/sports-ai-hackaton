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

from pose_tracker import PoseTracker, LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, det_cache_path, load_det_cache, save_det_cache
from athletes import (
    AKA, AO, AthleteSelector, TrackStats, classify_belt_color, dilate_flags, passes_athlete_filter, zone_flags, zone_rects,
)
from calibration import calibrate_from_click, load_calibration, save_calibration
from mat_region import FrameRegistrar, MatRegion, load_mat, mat_path
from analysis.karate import compute_movement_stats
from analysis.metrics import compute_clip_metrics
from identity import label_orientation, pair_signal, side_order
from visualization import draw_frame, render_heatmap, save_color_debug_strip, slot_style
from analysis.ground_truth import compute_gt_metrics, gt_path, load_frame_detections, load_gt
from export import JsonlWriter, write_run_meta

# Debug strips are spread over the whole clip: one every N frames per slot.
COLOR_DEBUG_EVERY = 20
MAX_COLOR_DEBUG_FRAMES = 40

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
        help="Debug rendering: grey boxes/skeletons for non-athletes in the output video, plus the patch strips.",
    )
    parser.add_argument(
        "--no-side-prior", action="store_true",
        help="Ablation: decide AKA/AO from color evidence only (no left/right side-order prior).",
    )
    parser.add_argument(
        "--debug-strips", action="store_true",
        help="Only save belt/glove patch debug strips to <data-dir>/debug_color/ (output video stays clean).",
    )
    parser.add_argument(
        "--det-cache", default=None, metavar="DIR",
        help="Reuse (or save) the raw tracker output per video in DIR, keyed by video + model/conf/imgsz/tracker "
             "config. Everything after detection+tracking still runs on the real frames.",
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
            "ignore_zones": config["classifier"].get("ignore_zones") or [],
            "track_level_dark_clothing_check": True,
            "color_identity": "offline per-frame pair signal + Viterbi orientation over the whole clip",
            "trail_breaking_px": config["visualization"]["trail_break_dist_px"],
            "debug_rendering (grey boxes for non-athletes)": bool(args.debug),
            "identity_side_prior": bool(config.get("identity", {}).get("side_prior", True) and not args.no_side_prior),
            "color_debug_strips": bool(args.debug or args.debug_strips),
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

    # Optional active mat region (data/mats/<clip>.json, scripts/mark_mat.py): only
    # people standing on that mat can hold a slot. Follows a moving camera.
    mat_file = mat_path(Path(config["export"]["data_dir"]), input_path.stem)
    mat = load_mat(mat_file)
    mat_region = registrar = None
    if mat is not None:
        if list(mat["frame_size"]) != [width, height]:
            raise RuntimeError(f"{mat_file} was marked on a {mat['frame_size']} frame, video is {width}x{height}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, mat["frame"])
        ok, ref_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read mat reference frame {mat['frame']}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        registrar = FrameRegistrar(ref_frame)
        mat_region = MatRegion(mat["corners"], config.get("mat", {}).get("margin", 0.1))
        logger.info("Mat region: %s (reference frame %d, margin %.2f)", mat_file, mat["frame"], mat_region.margin)

    cache_path = det_cache_path(
        args.det_cache, input_path, config["model"]["name"], config["model"].get("conf", 0.25),
        config["model"].get("imgsz", 640), config.get("tracker", "bytetrack.yaml"),
    ) if args.det_cache else None
    cached_dets = load_det_cache(cache_path) if cache_path else None
    if cached_dets is not None:
        tracker = None
        logger.info("Detection cache: reusing %s (%d frames)", cache_path, len(cached_dets))
    else:
        tracker = PoseTracker(
            config["model"]["name"], config.get("device", "auto"),
            config.get("tracker", "bytetrack.yaml"), config["model"].get("conf", 0.25),
            config["model"].get("imgsz", 640),
        )
    device = tracker.device if tracker else "cache"

    provenance = build_provenance(config, args, device)
    provenance["det_cache"] = str(cache_path) if cached_dets is not None else None
    provenance["active_features"]["mat_region"] = str(mat_file) if mat is not None else None
    logger.info("RUN PROVENANCE: %s", json.dumps(provenance, indent=2))

    selector = AthleteSelector(
        min_track_len=config["tracking"]["min_track_len"],
        min_bbox_area_frac=config["tracking"]["min_bbox_area_frac"],
        reassoc_max_dist_frac=config["athlete_selection"]["reassoc_max_dist_frac"],
        frame_width=width, frame_height=height,
    )

    classifier_cfg = config["classifier"]
    ignore_zone_cfgs = classifier_cfg.get("ignore_zones") or []
    graphic_frames = 0
    dark_brightness_threshold = classifier_cfg["dark_brightness_threshold"]

    ankle_conf_threshold = config["ground_point"]["ankle_conf_threshold"]
    smoothing_window = config["ground_point"]["smoothing_window"]
    trail_seconds = config["visualization"]["trail_seconds"]
    trail_break_dist = config["visualization"]["trail_break_dist_px"]

    ground_point_history = {AKA: [], AO: []}
    trails = {AKA: [], AO: []}  # each: list of (timestamp, point-or-None)
    color_debug_dir = data_dir / "debug_color"
    color_debug_counts = {AKA: 0, AO: 0}  # keyed by the athlete LABEL (aka/ao)

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

    # Pass 1: detect + track + sample belt/glove color on EVERY frame, keeping
    # per-frame results and per-track cumulative evidence for the whole clip.
    # Pass 2 re-reads the video and runs selection with that full-clip
    # evidence, so AKA/AO is decided from everything seen, not the first frames.
    # Graphic windows first (cheap, no inference): which frames each ignore zone's
    # graphic is up, widened by fade_pad_seconds each side.
    zone_on = [[] for _ in ignore_zone_cfgs]
    while ignore_zone_cfgs:
        ok, frame = cap.read()
        if not ok:
            break
        for z, v in zip(zone_on, zone_flags(frame, ignore_zone_cfgs)):
            z.append(v)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pad = int(round(classifier_cfg.get("graphic_fade_pad_seconds", 0.5) * fps))
    zone_on = [dilate_flags(z, pad) for z in zone_on]

    cached = []
    track_ev: dict[int, list[tuple[int, float, float]]] = {}
    track_dark: dict[int, list[tuple[int, bool]]] = {}
    track_stats: dict[int, tuple[int, float]] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame

        detections = cached_dets[len(cached)] if cached_dets is not None else tracker.track(frame)
        if mat_region is not None:
            mat_region.add_frame(registrar.to_ref(frame))
        raw_ground_points = [ground_point_from_detection(d, ankle_conf_threshold) for d in detections]
        ignore_zones = zone_rects((width, height), ignore_zone_cfgs, [z[len(cached)] for z in zone_on])
        graphic_frames += bool(ignore_zones)
        filter_results = [
            passes_athlete_filter(frame, d.bbox, d.keypoints, ignore_zones, dark_brightness_threshold)
            for d in detections
        ]
        color_results = [classify_belt_color(frame, d.bbox, d.keypoints) for d in detections]
        for det, (_, _, (red, blue)), (_, reject) in zip(detections, color_results, filter_results):
            if det.tracker_id is not None:
                n, total = track_stats.get(det.tracker_id, (0, 0.0))
                track_stats[det.tracker_id] = (
                    n + 1, total + (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]),
                )
                track_ev.setdefault(det.tracker_id, []).append((len(cached), red, blue))
                track_dark.setdefault(det.tracker_id, []).append(
                    (len(cached), bool(reject and reject.startswith("dark_clothing")))
                )
        cached.append((detections, raw_ground_points, filter_results, color_results))
        if len(cached) % 100 == 0:
            logger.info("Pass 1: %d frames", len(cached))
    if cached_dets is not None and len(cached_dets) != len(cached):
        raise RuntimeError(f"detection cache {cache_path} has {len(cached_dets)} frames, video has {len(cached)}")
    if cache_path is not None and cached_dets is None:
        save_det_cache(cache_path, [c[0] for c in cached])
        logger.info("Detection cache: saved %s", cache_path)

    if mat_region is not None:
        logger.info("Mat region: %d/%d frames could not be registered to the reference frame (kept the last good transform)",
                    registrar.failed_frames, len(cached))

    def on_mat_ids(t):
        """Tracks standing on the active mat region at frame t (None = no region)."""
        if mat_region is None:
            return None
        dets, gps = cached[t][0], cached[t][1]
        return {d.tracker_id for d, gp in zip(dets, gps) if mat_region.on_mat(t, gp, d.bbox, height)}

    selector.final_dark_flags = track_dark
    selector.final_track_stats = track_stats
    officials = sorted(
        tid for tid, flags in track_dark.items()
        if any(selector.is_official(TrackStats(tid, dark_flags=flags), f) for f, _ in flags[::5])
    )
    logger.info("Ignore-zone graphic on screen in %d/%d frames; tracks judged an official (dark clothing) at some point: %s",
                graphic_frames, len(cached), officials)
    selector.final_ev = track_ev
    logger.info(
        "Whole-clip color evidence per track (red, blue): %s",
        {tid: (round(sum(r for _, r, _ in rows), 1), round(sum(b for _, _, b in rows), 1))
         for tid, rows in sorted(track_ev.items()) if sum(r + b for _, r, b in rows) >= 1.0},
    )

    # Stage 2a: continuity-only selection of the two slots (no pixels needed),
    # plus the per-frame pair signal x(t) for the identity decision.
    slot_frames = []
    pair_x = []
    slot_centers = []
    for t, (detections, raw_ground_points, filter_results, color_results) in enumerate(cached):
        evidence = [ev for _, _, ev in color_results]
        selector.update_track_stats(t, detections, raw_ground_points, filter_results, evidence)
        selector.select_two(t, tatami_mask_fn=tatami_mask_fn, allowed_ids=on_mat_ids(t))

        slot_of_det = [selector.tracker_to_slot(d.tracker_id) for d in detections]
        smoothed = list(raw_ground_points)
        slot_ev = {AKA: None, AO: None}
        for i, (slot, raw_gp) in enumerate(zip(slot_of_det, raw_ground_points)):
            if slot is None:
                continue
            slot_ev[slot] = evidence[i]
            if raw_gp is not None:
                history = ground_point_history[slot]
                history.append(raw_gp)
                del history[: max(0, len(history) - smoothing_window)]
                smoothed[i] = moving_average_point(history)
                selector.note_ground_point(slot, smoothed[i])
        slot_frames.append((slot_of_det, smoothed))
        pair_x.append(pair_signal(slot_ev[AKA], slot_ev[AO]))
        boxes = {slot_of_det[i]: detections[i].bbox for i in range(len(detections)) if slot_of_det[i] is not None}
        slot_centers.append(
            None if len(boxes) < 2 else (
                (boxes[AKA][0] + boxes[AKA][2]) / 2, (boxes[AO][0] + boxes[AO][2]) / 2,
                ((boxes[AKA][2] - boxes[AKA][0]) + (boxes[AO][2] - boxes[AO][0])) / 2,
            )
        )

    # Stage 2b: which continuity slot is AKA, decided over the whole clip.
    orientation, identity_known, identity_switches = label_orientation(pair_x, side_order(slot_centers) if config.get('identity', {}).get('side_prior', True) and not args.no_side_prior else None)
    selector.color_swap_count = identity_switches
    logger.info(
        "Identity: %d orientation switch(es); %.1f%% of frames have a labeled (non-'?') identity",
        identity_switches, 100.0 * sum(identity_known) / max(len(identity_known), 1),
    )

    # Stage 2c: render + export with those labels.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for (detections, raw_ground_points, filter_results, color_results), (slot_of_det, smoothed_points) in zip(
        cached, slot_frames,
    ):
        ok, frame = cap.read()
        if not ok:
            break
        known = identity_known[frame_idx]
        aka_is_slot_a = orientation[frame_idx] == 1

        def label_of(slot):
            return None if slot is None else (AKA if (slot == AKA) == aka_is_slot_a else AO)

        timestamp = frame_idx / fps
        visible_slots = set()
        slot_points = {AKA: None, AO: None}
        det_styles = {}

        for det, slot, raw_gp, smoothed_gp, (passed, reject_reason), (color, color_samples, color_ev) in zip(
            detections, slot_of_det, raw_ground_points, smoothed_points, filter_results, color_results,
        ):
            athlete = label_of(slot)
            tatami_point = None

            if slot is not None:
                det_styles[det.tracker_id] = slot_style(athlete, known)
            if slot is not None and raw_gp is not None:
                visible_slots.add(slot)
                slot_points[slot] = smoothed_gp
                if calibration is not None:
                    tatami_point = calibration.to_tatami(smoothed_gp)

                if (args.debug or args.debug_strips) and frame_idx % COLOR_DEBUG_EVERY == 0 and color_debug_counts[athlete] < MAX_COLOR_DEBUG_FRAMES and any(
                    s["color"] != "no_keypoint" for s in color_samples
                ):
                    save_color_debug_strip(
                        frame, det.bbox, color_samples, color,
                        color_debug_dir / f"{input_path.stem}_{athlete}_f{frame_idx:05d}.png",
                    )
                    color_debug_counts[athlete] += 1

            record = {
                "frame": frame_idx,
                "timestamp": timestamp,
                "athlete_id": athlete,
                "tracker_id": det.tracker_id,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "keypoints": det.keypoints,
                "ground_point": smoothed_gp,
                "tatami_point": tatami_point,
                "belt_color": color,
                "color_evidence": [round(color_ev[0], 3), round(color_ev[1], 3)],
                "identity_known": known if slot is not None else None,
                "passed_athlete_filter": passed,
                "reject_reason": reject_reason,
                "on_mat": None if mat_region is None else mat_region.on_mat(frame_idx, raw_gp, det.bbox, height),
            }
            jsonl.write(record)
            if athlete is not None:
                records.append(record)

        # Trails follow the continuity slot (the same person), colored by that
        # slot's current label.
        for slot in (AKA, AO):
            append_trail_point(trails[slot], timestamp, slot_points[slot], trail_break_dist)
            trim_trail(trails[slot], timestamp, trail_seconds)
        trail_styles = {slot: slot_style(label_of(slot), known) for slot in (AKA, AO)}

        annotated = draw_frame(
            frame, detections, det_styles, trail_styles, trails, visible_slots,
            config["visualization"], debug=args.debug,
        )
        if mat_region is not None:
            cv2.polylines(annotated, [mat_region.polygon(frame_idx).astype("int32")], True, (0, 200, 200), 1)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info("Rendered %d frames", frame_idx)

    cap.release()
    writer.release()
    jsonl.close()

    elapsed = time.time() - start_time
    effective_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    write_run_meta(data_dir / f"{input_path.stem}_run_meta.json", {
        "provenance": provenance,
        "graphic_frames": graphic_frames,
        "official_tracks": officials,
        "identity_switches": identity_switches,
        "pct_frames_identity_known": 100.0 * sum(identity_known) / max(len(identity_known), 1),
        "model": config["model"]["name"],
        "tracker": config.get("tracker", "bytetrack.yaml"),
        "device": tracker.device if tracker else "cache",
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
        "mat_region": None if mat is None else {
            "file": str(mat_file), "reference_frame": mat["frame"], "corners": mat["corners"],
            "margin": mat_region.margin, "unregistered_frames": registrar.failed_frames,
        },
    })

    logger.info(
        "Done: %d frames in %.1fs (%.1f fps effective). ID switches: %d, color swaps: %d",
        frame_idx, elapsed, effective_fps, selector.id_switch_count, selector.color_swap_count,
    )

    metrics = compute_clip_metrics(jsonl_path, frame_idx, selector.id_switch_count, selector.color_swap_count)
    gt = load_gt(gt_path(Path(config["export"]["data_dir"]), input_path.stem))
    if gt is not None:
        metrics["ground_truth"] = compute_gt_metrics(load_frame_detections(jsonl_path), gt)
        g = metrics["ground_truth"]
        logger.info(
            "GT: identity_acc=%s%% swapped=%d unknown=%d non_athlete_slot_frames=%d missed=%d (%d labeled frames)",
            g["identity_accuracy_pct"], g["identity_swapped"], g["identity_unknown"],
            g["non_athlete_slot_frames"], g["missed_athletes"], g["labeled_frames"],
        )
    else:
        logger.info("GT: no ground-truth labels for %s (run scripts/label_ground_truth.py)", input_path.stem)
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
