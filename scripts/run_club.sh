#!/bin/zsh
# Usage: scripts/run_club.sh club_v7 [extra main.py args]  -> videos/output/club_v7/, data/club_v7/
# Club footage (handheld camera): broadcast graphic zones off (config_club.yaml),
# no tatami calibration, no heatmaps. videos/input/club_NN.mp4 are symlinks into club/.
tag=${1:?tag}; shift
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p "data/$tag" "videos/output/$tag"
for f in videos/input/club_*.mp4; do
  n=$(basename "$f" .mp4)
  python src/main.py --input "$f" --output "videos/output/$tag/${n}_analyzed.mp4" --data-dir "data/$tag" --config config_club.yaml --debug-strips --no-heatmap "$@" > "data/$tag/${n}.log" 2>&1 || echo "FAILED $n"
done
