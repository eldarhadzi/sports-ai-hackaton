#!/bin/zsh
# Usage: scripts/run_all.sh v4   -> videos/output/v4/, data/v4/
tag=${1:?tag}
cd "$(dirname "$0")/.." && source .venv/bin/activate
for f in videos/input/*.mp4; do
  n=$(basename "$f" .mp4)
  python src/main.py --input "$f" --output "videos/output/$tag/${n}_analyzed.mp4" --data-dir "data/$tag" --debug --no-heatmap > "data/$tag/${n}.log" 2>&1 || echo "FAILED $n"
done
