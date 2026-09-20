"""HS-BAND -- the pre-freeze hard stop for EXP-CORRESPONDENCE-CP-001.

MODEL-FREE.  No checkpoint is opened, no weight is trained, no optimizer is
imported, no historical record is written.  Frozen modules are imported and used
verbatim; random weights are used deliberately, because the quantity being
measured is a RECEPTIVE-FIELD SUPPORT question, which is a property of the
architecture and not of any trained weights.

THE QUESTION
------------
CP-001 needs a right-image edit band at signed feature-cell offset `delta` from a
tested column `w_test` such that BOTH hold:

    C1  the tested pixel's own cost slice is BIT-IDENTICAL with and without the
        band edit                     ->  max|V1[...,w_test] - V0[...,w_test]| == 0.0
    C2  the band still reaches the tested pixel through the aggregation's 5-cell
        spatial reach                 ->  features differ at some w' with
                                          1 <= |w' - w_test| <= 5, AND the
                                          aggregated cost at w_test differs

C1 is required so that every candidate-POINTWISE mechanism (the class that
defeated O1) has slope exactly 0 by construction.  C2 is required so the
experiment has any signal at all.  The two pull in opposite directions: C1 wants
the edit outside the ~15-cell extractor halo of w_test, C2 wants it inside the
halo of a neighbour within 5 cells.  The predicted band is therefore only a few
cells wide, and the DECISION record flags this bound as DERIVED, NOT MEASURED --
the same defect class that halted GEOM-001.  This script measures it.

CONTENT CHOICE
--------------
Random content, not real KITTI crops.  Random content is the CONSERVATIVE choice
here: every dependence it can express is generically non-zero, so an exact zero
is a statement about support alone.  Real content can only add accidental zeros,
i.e. false PASSes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("C:/Users/vishn/stereo_depth_vision")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch                                                            # noqa: E402

from src.models.stereonet.feature_extractor import FeatureExtractor      # noqa: E402
from src.models.stereonet.cost_volume import build_cost_volume           # noqa: E402
from src.models.stereonet.aggregation import Aggregation                 # noqa: E402

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

# ----------------------------------------------------------- frozen geometry
CROP_H, CROP_W = 272, 1136
STRIDE, D = 16, 12
FW = CROP_W // STRIDE                     # 71
W_TEST = 30                               # inside the frozen STAT band [20,39]
BAND_CELLS = 3                            # band width, feature cells
AGG_REACH = 5                             # 5 padded 3-taps in w
EXT_SEEDS = [11, 12, 13]
AGG_SEED = 21
DELTAS = list(range(-30, 31))

report: dict = {
    "w_test": W_TEST, "band_cells": BAND_CELLS, "agg_reach": AGG_REACH,
    "extractor_seeds": EXT_SEEDS, "content": "random uniform, conservative",
}

torch.manual_seed(7)
left_img = torch.rand(1, 3, CROP_H, CROP_W)
right0 = torch.rand(1, 3, CROP_H, CROP_W)
patch = torch.rand(1, 3, CROP_H, CROP_W)          # replacement content

extractors = []
for s in EXT_SEEDS:
    torch.manual_seed(s)
    extractors.append(FeatureExtractor().eval())
torch.manual_seed(AGG_SEED)
agg = Aggregation().eval()


def edit(delta: int) -> torch.Tensor | None:
    c0, c1 = W_TEST + delta, W_TEST + delta + BAND_CELLS
    if c0 < 0 or c1 > FW:
        return None
    r = right0.clone()
    r[:, :, :, c0 * STRIDE:c1 * STRIDE] = patch[:, :, :, c0 * STRIDE:c1 * STRIDE]
    return r


# ------------------------------------------------ measured one-sided RF radius
# a single-cell edit, which columns of the feature map move at all?
radii = []
with torch.no_grad():
    for ex in extractors:
        r = right0.clone()
        c = 35
        r[:, :, :, c * STRIDE:(c + 1) * STRIDE] = patch[:, :, :, c * STRIDE:(c + 1) * STRIDE]
        d = (ex(r) - ex(right0)).abs().amax(dim=(0, 1, 2))          # (FW,)
        nz = (d != 0).nonzero().flatten().tolist()
        radii.append({"edit_cell": c, "affected_cols": [min(nz), max(nz)],
                      "left_reach": c - min(nz), "right_reach": max(nz) - c})
report["measured_receptive_support_single_cell_edit"] = radii

# ------------------------------------------------------------------ the scan
rows = []
with torch.no_grad():
    for delta in DELTAS:
        r1 = edit(delta)
        if r1 is None:
            continue
        if 0 <= delta + BAND_CELLS - 1 and delta <= 0:
            overlaps = True                     # band covers w_test itself
        else:
            overlaps = False
        per_seed = []
        for ex in extractors:
            f0, f1 = ex(right0), ex(r1)
            dcol = (f1 - f0).abs().amax(dim=(0, 1, 2))              # (FW,)
            d_test = float(dcol[W_TEST])
            lo, hi = max(0, W_TEST - AGG_REACH), min(FW - 1, W_TEST + AGG_REACH)
            near = [float(dcol[w]) for w in range(lo, hi + 1) if w != W_TEST]
            lf = ex(left_img)
            v0 = build_cost_volume(lf, f0, D, "subtract", "left")
            v1 = build_cost_volume(lf, f1, D, "subtract", "left")
            dv_test = float((v1[:, :, :, :, W_TEST] - v0[:, :, :, :, W_TEST]).abs().max())
            c0, c1 = agg(v0), agg(v1)
            dagg = float((c1[:, :, :, W_TEST] - c0[:, :, :, W_TEST]).abs().max())
            per_seed.append({"d_feat_test": d_test, "d_feat_near_max": max(near),
                             "d_V_test": dv_test, "d_agg_test": dagg})
        c1_ok = all(s["d_V_test"] == 0.0 for s in per_seed)
        c2_ok = all(s["d_feat_near_max"] > 0.0 for s in per_seed) and \
                all(s["d_agg_test"] > 0.0 for s in per_seed)
        rows.append({"delta": delta, "band_cells": [W_TEST + delta,
                                                    W_TEST + delta + BAND_CELLS - 1],
                     "covers_w_test": overlaps,
                     "C1_V_bit_identical_at_w_test": c1_ok,
                     "C2_reaches_w_test_via_aggregation": c2_ok,
                     "PASS": bool(c1_ok and c2_ok),
                     "max_d_V_test": max(s["d_V_test"] for s in per_seed),
                     "min_d_feat_near_max": min(s["d_feat_near_max"] for s in per_seed),
                     "min_d_agg_test": min(s["d_agg_test"] for s in per_seed)})

report["scan"] = rows
band = [r["delta"] for r in rows if r["PASS"]]
report["admissible_deltas"] = band
report["band_width_cells"] = len(band)
report["VERDICT"] = "PASS" if band else "FAIL"
report["predicted_band_from_decision_record"] = "|delta| in [15,19] (derived)"

print(json.dumps(report, indent=1))
