"""One command per match, any OS: pipeline -> features -> profiles -> browser-playable videos.

    python scripts/run_match.py --run v1 --clips kumite_01_siroki_kadar kumite_02_siroki_kadar \
        [--corners "185,195,460,195,495,335,150,335"] [--config config.yaml] [--skip-pipeline]

For each clip videos/input/<clip>.mp4:
  1. src/main.py        -> data/<run>/<clip>.jsonl + videos/output/<run>/<clip>_analyzed.mp4
                           (raw detections cached in data/det_cache: re-runs skip the slow model pass)
  2. features.py        -> data/<run>/features/<clip>.json
  3. H.264 copy         -> videos/output/<run>/<clip>_web.mp4 (the dashboard plays this one)
Then scripts/build_profiles.py -> data/<run>/profiles.json (needs the clips in data/roster.csv).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output tag, e.g. v1")
    ap.add_argument("--clips", nargs="+", required=True, help="names in videos/input/ without .mp4")
    ap.add_argument("--config", default="config.yaml", help="config_club.yaml for non-broadcast footage")
    ap.add_argument("--corners", default=None, help="tatami corners x1,y1,...,x4,y4 (used only if no calibration saved)")
    ap.add_argument("--skip-pipeline", action="store_true", help="only recompute features/profiles from existing JSONL")
    args = ap.parse_args()

    data_dir = Path("data") / args.run
    out_dir = Path("videos") / "output" / args.run
    for clip in args.clips:
        if not args.skip_pipeline:
            cmd = [PY, "src/main.py", "--input", f"videos/input/{clip}.mp4",
                   "--output", out_dir / f"{clip}_analyzed.mp4", "--data-dir", data_dir,
                   "--config", args.config, "--det-cache", "data/det_cache", "--no-heatmap"]
            if args.corners:
                cmd += ["--corners", args.corners]
            run(cmd)
        run([PY, "src/analysis/features.py", data_dir / f"{clip}.jsonl"])
        src = ROOT / out_dir / f"{clip}_analyzed.mp4"
        if src.exists():
            import imageio_ffmpeg
            run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", src.with_name(f"{clip}_web.mp4")])
    run([PY, "scripts/build_profiles.py", "--run", args.run])


if __name__ == "__main__":
    main()
