"""V3 (data/v3) vs V4 (data/v4) metrics + output video hash check.
    python scripts/compare_v3_v4.py
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ["kumite_01_siroki_kadar", "kumite_02_siroki_kadar", "kumite_03_siroki_kadar",
         "kumite_04_siroki_kadar_pauza", "kumite_05_siroki_kadar", "kumite_06_medium_kamera_prati"]
F = [("pct_both_assigned", "both%"), ("pct_one_assigned", "1%"), ("pct_zero_assigned", "0%"),
     ("id_switch_count", "reassign"), ("color_swap_count", "flips"), ("filter_violations", "filt_viol")]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()[:10] if p.exists() else "missing"

print(f"{'clip':<32}" + "".join(f"{l:>16}" for _, l in F) + "  video V3 vs V4")
for c in CLIPS:
    m3 = json.loads((ROOT / f"data/v3/{c}_metrics.json").read_text())
    m4p = ROOT / f"data/v4/{c}_metrics.json"
    m4 = json.loads(m4p.read_text()) if m4p.exists() else {}
    row = f"{c:<32}" + "".join(f"{str(m3[k]) + ' -> ' + str(m4.get(k, '?')):>16}" for k, _ in F)
    h3, h4 = sha(ROOT / f"videos/output/{c}_analyzed.mp4"), sha(ROOT / f"videos/output/v4/{c}_analyzed.mp4")
    print(row + f"  {h3} {'==' if h3 == h4 else '!='} {h4}")
