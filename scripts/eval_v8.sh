#!/bin/zsh
# Usage: scripts/eval_v8.sh RUN [extra main.py args]
# Dev loop for round v8: the 6 broadcast clips (config.yaml, their calibration) and
# club_02 (config_club.yaml), all through the detection cache data/det_cache, then
# gt_report against the frozen baseline runs v7 / club_v7.
run=${1:?run}; shift
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p "data/$run" && cp -R data/calibration "data/$run/"
for n in kumite_01_siroki_kadar kumite_02_siroki_kadar kumite_03_siroki_kadar \
         kumite_04_siroki_kadar_pauza kumite_05_siroki_kadar kumite_06_medium_kamera_prati club_02; do
  cfg=config.yaml; [[ $n == club_* ]] && cfg=config_club.yaml
  python src/main.py --input "videos/input/$n.mp4" --output "videos/output/$run/${n}_analyzed.mp4" --data-dir "data/$run" \
    --config $cfg --det-cache data/det_cache --no-heatmap "$@" > "data/$run/${n}.log" 2>&1 || echo "FAILED $n"
done
python scripts/gt_report.py v7 club_v7 "$run"
