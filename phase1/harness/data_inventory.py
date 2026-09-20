"""Read-only local data inventory: counts only, no writes outside phase1."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data" / "kitti2015"

def count(d, pat):
    p = D / d
    if not p.exists():
        return f"MISSING {d}"
    files = list(p.glob(pat))
    return f"{d}/{pat}: {len(files)}"

for d, pat in [("training/image_2", "*_10.png"), ("training/image_2", "*.png"),
               ("training/image_3", "*_10.png"), ("training/image_3", "*.png"),
               ("training/disp_occ_0", "*.png"), ("training/disp_noc_0", "*.png"),
               ("training/calib_cam_to_cam", "*.txt"), ("testing/image_2", "*.png"),
               ("testing/image_3", "*.png")]:
    print(count(d, pat))

# split scene lists via repo loader (no image reads beyond names)
import sys
sys.path.insert(0, str(REPO))
from src.datasets.kitti2015 import Kitti2015Stereo
for split in ["hailo_calib", "hailo_val", "all"]:
    ds = Kitti2015Stereo(D, split=split)
    print(f"split={split} n={len(ds)} first={ds.names[0] if ds.names else None} last={ds.names[-1] if ds.names else None}")
# overlap check
calib = set(Kitti2015Stereo(D, split="hailo_calib").names)
val = set(Kitti2015Stereo(D, split="hailo_val").names)
print(f"overlap_calib_val={len(calib & val)} disjoint={not (calib & val)}")
print("val_40_list:")
for n in sorted(val):
    print("  " + n)
