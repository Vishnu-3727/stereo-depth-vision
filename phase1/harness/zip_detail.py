"""Read-only second-level breakdown of data_scene_flow.zip."""
import zipfile
from pathlib import Path
from collections import Counter
REPO = Path(__file__).resolve().parents[2]
zp = REPO / "data" / "kitti2015" / "data_scene_flow.zip"
with zipfile.ZipFile(zp, "r") as z:
    names = z.namelist()
c = Counter()
for n in names:
    parts = n.split("/")
    key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
    c[key] += 1
for k in sorted(c):
    print(f"{k}: {c[k]}")
# SceneFlow markers (must be zero if KITTI archive)
for m in ["frames_cleanpass", "frames_disparity", ".pfm", "monkaa", "flyingthings", "driving"]:
    print(f"marker {m}: {sum(1 for n in names if m.lower() in n.lower())}")
# KITTI markers
for m in ["image_2", "image_3", "disp_occ_0", "disp_noc_0", "disp_occ_1", "flow_occ", "obj_map", "viz"]:
    print(f"marker {m}: {sum(1 for n in names if m in n)}")
