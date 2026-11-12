#!/bin/zsh
# Usage: scripts/run_all.sh v5 [extra main.py args]  -> videos/output/v5/, data/v5/
tag=${1:?tag}; shift
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p "data/$tag" && cp -R data/calibration "data/$tag/"
for f in videos/input/*.mp4; do
  n=$(basename "$f" .mp4)
  python src/main.py --input "$f" --output "videos/output/$tag/${n}_analyzed.mp4" --data-dir "data/$tag" --debug-strips --no-heatmap "$@" > "data/$tag/${n}.log" 2>&1 || echo "FAILED $n"
done
