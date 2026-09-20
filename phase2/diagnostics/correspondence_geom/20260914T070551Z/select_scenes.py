"""EXP-CORRESPONDENCE-GEOM-A -- GT-only scene selection.

Reads GROUND TRUTH ONLY. Opens no checkpoint, instantiates no model, runs no
inference. Running this script is allowed and required before freezing: it
fixes the scene list that mask_spec.json and run_geomA.py then take as frozen.

Criterion (frozen, from the task brief):
  - split hailo_val (40 scenes);
  - exclude the 4 scenes already used by GEOM-001 {27, 0, 31, 6};
  - from the remaining held-out scenes keep those whose band retention is >= 0.50;
  - take the first N = 4 qualifying scenes by ascending scene index.

Band retention for scene i:
    numerator   = #{GT > 0 AND interior AND GT/16 in [2,9]}
    denominator = #{GT > 0 AND interior}
    retention   = numerator / denominator   (0.0 when the denominator is 0)

where "interior" is the geometric mask interior EDGE <= y < H-EDGE,
EDGE <= x < W-EDGE with EDGE = 96 = M(32) + B(64), in ORIGINAL image
coordinates. GT scale is 256.0 (KITTI convention), matching core.load_scene.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402

W, H = 1232, 368
M = 32
B = 64
EDGE = M + B  # 96
SPLIT = "hailo_val"
EXCLUDED = [27, 0, 31, 6]  # scenes already used by GEOM-001
THRESHOLD = 0.50
N = 4
BAND_LO, BAND_HI = 2.0, 9.0  # GT/16 in [2,9]


def main() -> int:
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=SPLIT,
                         disparity_scale=256.0)
    assert len(ds) == 40, "expected 40 hailo_val scenes, got %d" % len(ds)

    per_scene = {}
    for si in range(len(ds)):
        if si in EXCLUDED:
            continue
        sample = ds[si]
        assert sample.left.shape[:2] == (H, W), (sample.name, sample.left.shape[:2])
        gt = sample.disparity.astype("float64")
        # geometric interior in ORIGINAL coordinates
        inside = np.zeros_like(gt, dtype=bool)
        inside[EDGE:H - EDGE, EDGE:W - EDGE] = True
        valid = (gt > 0) & inside
        band = valid & ((gt / 16.0) >= BAND_LO) & ((gt / 16.0) <= BAND_HI)
        denom = int(valid.sum())
        numer = int(band.sum())
        retention = (numer / denom) if denom else 0.0
        per_scene[si] = {"name": sample.name, "scene_index": si,
                         "valid_interior_pixels": denom,
                         "band_pixels": numer,
                         "band_retention": retention}

    qualifying = sorted(si for si, r in per_scene.items()
                        if r["band_retention"] >= THRESHOLD)
    # NO repair-and-continue: the criterion is frozen. On shortfall the measured
    # table is still written (it is real GT-only data), chosen is left empty,
    # and the script exits nonzero so no downstream freeze can proceed.
    shortfall = len(qualifying) < N
    chosen = qualifying[:N] if not shortfall else []

    payload = {
        "experiment": "EXP-CORRESPONDENCE-GEOM-A",
        "split": SPLIT,
        "criterion": ("exclude GEOM-001 scenes {27, 0, 31, 6}; from the remaining "
                      "held-out hailo_val scenes keep those whose band retention "
                      "(fraction of valid-GT pixels inside the geometric mask "
                      "interior with GT/16 in [2,9]) is >= 0.50; take the first "
                      "N = 4 qualifying scenes by ascending scene index"),
        "criterion_met": (not shortfall),
        "criterion_status": ("OK" if not shortfall else
                             "SHORTFALL: only %d scene(s) meet retention >= %.2f; "
                             "N = %d unmet; no scene list frozen"
                             % (len(qualifying), THRESHOLD, N)),
        "band": {"lo": BAND_LO, "hi": BAND_HI, "quantity": "GT/16",
                 "gt_scale": 256.0},
        "geometry_original_coords": {
            "width": W, "height": H,
            "crop_margin_M": M, "inner_border_B": B, "edge_exclusion_px": EDGE,
            "interior": "EDGE <= y < H-EDGE AND EDGE <= x < W-EDGE",
        },
        "excluded_geom001_scenes": sorted(EXCLUDED),
        "retention_threshold": THRESHOLD,
        "n_requested": N,
        "per_candidate_scene": {str(si): per_scene[si] for si in sorted(per_scene)},
        "n_qualifying": len(qualifying),
        "qualifying_scenes_ascending": qualifying,
        "chosen_scenes": chosen,
        "gt_only": True,
        "checkpoint_opened": False,
    }
    body = json.dumps(payload, indent=2)
    with open(OUT / "scene_selection.json", "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    print("candidates=%d qualifying=%d chosen=%s"
          % (len(per_scene), len(qualifying), chosen))
    for si in sorted(per_scene):
        r = per_scene[si]
        flag = "CHOSEN" if si in chosen else ("qual " if si in qualifying else "     ")
        print("  %s scene %2d %-14s retention=%.4f (%d/%d)"
              % (flag, si, r["name"], r["band_retention"],
                 r["band_pixels"], r["valid_interior_pixels"]))
    if shortfall:
        print("HARD STOP: only %d scene(s) meet retention >= %.2f; need N = %d. "
              "Measured table written; no scene list frozen."
              % (len(qualifying), THRESHOLD, N))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
