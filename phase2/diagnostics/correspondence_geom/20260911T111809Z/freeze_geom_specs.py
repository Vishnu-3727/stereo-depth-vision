"""EXP-CORRESPONDENCE-GEOM-001 -- freeze the translation and mask specifications.

Step 1-2 of the preregistration order (B14). Run ONCE, before PREREGISTRATION.md
is written; its output hashes are recorded in the preregistration.

Reads ground truth only. Loads no checkpoint, instantiates no model, runs no
inference.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

TS = "20260911T111809Z"
OUT = Path(r"C:\Users\vishn\stereo_depth_vision\phase2\diagnostics\correspondence_geom") / TS
REPO = Path(r"C:\Users\vishn\stereo_depth_vision")
sys.path.insert(0, str(REPO))
from phase2.viz import core  # noqa: E402

W, H, STRIDE = 1232, 368, 16
M = 48                       # crop margin, == max|Delta|
B = 64                       # additional inner mask border
EDGE = M + B                 # 112 px from every ORIGINAL edge
DELTAS = [-48, -32, -16, 0, 16, 32, 48]
SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"

# ---------------------------------------------------------------- translation
trans = {
    "experiment": "EXP-CORRESPONDENCE-GEOM-001",
    "mechanism": "crop-based translation: NO fill, NO padding, NO interpolation",
    "rationale": "a zero-filled shift creates an edge discontinuity whose position "
                 "differs between +Delta and -Delta, which would contribute to the ODD "
                 "part and contaminate the primary statistic. Cropping removes the "
                 "vacated region instead of filling it, so every condition sees only "
                 "real image content and every condition has identical output size.",
    "original_size": {"width": W, "height": H},
    "crop_margin_px": M,
    "crop_size": {"width": W - 2 * M, "height": H - 2 * M},
    "crop_size_in_feature_samples": {"width": (W - 2 * M) // STRIDE,
                                     "height": (H - 2 * M) // STRIDE},
    "left_crop": "cols [M, W-M), rows [M, H-M)  -- FIXED for every condition",
    "right_crop_horizontal": "cols [M-dx, W-M-dx), rows [M, H-M)",
    "right_crop_vertical": "cols [M, W-M), rows [M-dy, H-M-dy)",
    "deltas_px": DELTAS,
    "axes": ["horizontal", "vertical"],
    "n_distinct_conditions": 2 * (len(DELTAS) - 1) + 1,
    "delta_zero_shared": True,
    "all_deltas_multiple_of_stride": all(d % STRIDE == 0 for d in DELTAS),
    "max_abs_delta_equals_crop_margin": max(abs(d) for d in DELTAS) == M,
    "disparity_algebra": "d_crop = (x_L - M) - (x_R - M + dx) = d_orig - dx   [exact, integer]",
    "predicted_candidate_response": "Delta(disparity_initial) = -dx / 16 candidates",
    "anchor_slope_candidates_per_pixel": -1.0 / STRIDE,
    "why_not_64": "at |dx| = 64 px (4 candidates) the predicted output range "
                  "3.84-4.00 = -0.16 falls below the candidate floor 0, so the "
                  "soft-argmin would rail. 48 px keeps the predicted range within "
                  "0.84 .. 8.83 for every unit, using the identity medians already "
                  "published in ARCH-001. Rejected on geometry, not on expected results.",
    "why_multiples_of_16": "the feature extractor is fully convolutional with total "
                           "stride 2**4 = 16 (4 x Conv2d(5, stride=2, padding=2), then "
                           "ResBlocks and Conv2d(3, stride=1); no pooling, no "
                           "normalisation, no global op), so a 16n px image translation "
                           "produces EXACTLY an n-sample feature translation in the "
                           "interior. Non-multiples would introduce an interpolation "
                           "confound outside the hypothesis.",
    "interpolation": "none -- all crops are integer-aligned",
}

# ----------------------------------------------------------------------- mask
per_scene = {}
pooled = 0
for si in SCENES:
    sc = core.load_scene(si, split=SPLIT)
    gt = sc.gt_disparity.astype(np.float64)
    m = np.zeros_like(gt, dtype=bool)
    m[EDGE:H - EDGE, EDGE:W - EDGE] = True
    m &= (gt > 0)
    n = int(m.sum())
    pooled += n
    d = gt[m] / float(STRIDE)
    per_scene[sc.name] = {"scene_index": si, "retained_pixels": n,
                          "gt_over_16_mean": float(d.mean()),
                          "gt_over_16_median": float(np.median(d)),
                          "gt_over_16_min": float(d.min()),
                          "gt_over_16_max": float(d.max())}

mask = {
    "experiment": "EXP-CORRESPONDENCE-GEOM-001",
    "definition_original_coords": "retain pixel (y,x) iff GT[y,x] > 0 AND "
                                  "%d <= y < %d AND %d <= x < %d" % (EDGE, H - EDGE, EDGE, W - EDGE),
    "edge_exclusion_px": EDGE,
    "edge_exclusion_decomposition": {"crop_margin_M": M, "inner_border_B": B},
    "crop_frame_mapping": "mask_crop[y, x] = mask_original[y + M, x + M]",
    "single_fixed_mask": True,
    "delta_dependent": False,
    "model_dependent": False,
    "derived_from": "ground truth and image geometry only",
    "why_a_border_exists": "with crop-based translation no fill is ever produced, so the "
                           "border is NOT needed to exclude fill. It is retained to reduce "
                           "the residual influence of differing real content entering the "
                           "feature receptive field near the crop edge. The feature "
                           "receptive field is ~477 px, which no affordable border can "
                           "fully exclude (the image is only 368 px tall); the shift=none "
                           "control measures whatever residual survives.",
    "why_single_mask_is_valid": "the crop window size is identical for every Delta and "
                                "every axis, so one fixed pixel set is well defined for "
                                "all conditions. A Delta-dependent mask would change the "
                                "sample between conditions and bias the paired difference.",
    "band_restriction_applied": False,
    "band_note": "GT/16 in [2,8] is NOT applied to the primary. The provenance audit "
                 "measured that band to be effectively [2,4] (99.4 percent of its pixels), "
                 "and the statistic is a difference of medians over a fixed pixel set, so "
                 "the band narrows the sample without changing the structure. It is kept "
                 "as a preregistered descriptive secondary only.",
    "per_scene": per_scene,
    "pooled_retained_pixels": pooled,
    "scenes": SCENES, "split": SPLIT,
}

for name, payload in (("translation_spec.json", trans), ("mask_spec.json", mask)):
    body = json.dumps(payload, indent=2)
    (OUT / name).write_text(body, encoding="utf-8")
    print("%-24s sha256 %s" % (name, hashlib.sha256(body.encode("utf-8")).hexdigest()))

digests = {n: hashlib.sha256((OUT / n).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
           for n in ("translation_spec.json", "mask_spec.json")}
(OUT / "frozen.sha256").write_text(
    "".join("%s  %s\n" % (d, n) for n, d in digests.items()), encoding="utf-8")
print("pooled retained pixels:", pooled)
print("per-scene:", {k: v["retained_pixels"] for k, v in per_scene.items()})
