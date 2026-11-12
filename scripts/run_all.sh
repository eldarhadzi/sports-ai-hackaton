#!/bin/zsh
# Usage: scripts/run_all.sh v5 [extra main.py args]  -> videos/output/v5/, data/v5/
# Broadcast baseline clips only (explicit list; club footage: scripts/run_club.sh).
tag=${1:?tag}; shift
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p "data/$tag" && cp -R data/calibration "data/$tag/"
clips=(kumite_01_siroki_kadar kumite_02_siroki_kadar kumite_03_siroki_kadar
       kumite_04_siroki_kadar_pauza kumite_05_siroki_kadar kumite_06_medium_kamera_prati)
for n in $clips; do
  f="videos/input/$n.mp4"
  python src/main.py --input "$f" --output "videos/output/$tag/${n}_analyzed.mp4" --data-dir "data/$tag" --debug-strips --no-heatmap "$@" > "data/$tag/${n}.log" 2>&1 || echo "FAILED $n"
done
