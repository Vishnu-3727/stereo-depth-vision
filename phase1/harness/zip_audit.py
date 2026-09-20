"""Read-only zip audit: list entry names only, no extraction, no writes outside phase1."""
import zipfile
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "kitti2015"

for zp in sorted(DATA.glob("*.zip")):
    print(f"=== {zp.name} size_bytes={zp.stat().st_size} ===")
    with zipfile.ZipFile(zp, "r") as z:
        names = z.namelist()
    print(f"entries={len(names)}")
    tops = Counter(n.split("/")[0] if "/" in n else "(root)" for n in names)
    print("top_level:", dict(tops))
    print("first_25:")
    for n in names[:25]:
        print("  " + n)
    print("last_10:")
    for n in names[-10:]:
        print("  " + n)
    # extension histogram (read-only, from names)
    exts = Counter((Path(n).suffix.lower() or "(dir)" ) for n in names)
    print("extensions:", dict(exts))
    # KITTI vs SceneFlow markers
    markers = ["frames_cleanpass", "frames_disparity", ".pfm", "monkaa", "flying", "driving",
               "image_2", "image_3", "disp_occ", "disp_noc", "calib", "scene_flow", ".txt", ".png"]
    for m in markers:
        c = sum(1 for n in names if m in n)
        if c:
            print(f"marker {m!r}: {c}")
    print()
