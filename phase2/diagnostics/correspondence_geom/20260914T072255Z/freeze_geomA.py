"""EXP-CORRESPONDENCE-GEOM-A -- freeze the translation, mask and scene specs.

Runs BEFORE PREREGISTRATION.md is written; the output hashes are recorded in
the preregistration. Reads ground truth only. Opens no checkpoint, instantiates
no model, runs no inference, writes no checkpoint.

Fail-fast: HS-SELECTION halts before ANY file is written unless
scene_selection.json carries a valid frozen list: criterion_met true with
ALL qualifying held-out scenes (at least 2) in ascending order. No partial
frozen state is ever produced.
"""
import hashlib
import json
import sys
from pathlib import Path

TS = "20260914T072255Z"
OUT = Path(r"C:\Users\vishn\stereo_depth_vision\phase2\diagnostics\correspondence_geom") / TS
REPO = Path(r"C:\Users\vishn\stereo_depth_vision")
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402

W, H, STRIDE = 1232, 368, 16
M = 32                       # crop margin, == max|Delta|
B = 64                       # additional inner mask border (same B as GEOM-001)
EDGE = M + B                 # 96 px from every ORIGINAL edge
BAND_LO, BAND_HI = 2.0, 9.0  # GT/16 in [2,9]
DELTAS = [-32, -16, 0, 16, 32]
SPLIT = "hailo_val"
EXCLUDED = {27, 0, 31, 6}


class HardStop(RuntimeError):
    pass


def write_lf(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ selection
sel = json.loads((OUT / "scene_selection.json").read_text(encoding="utf-8"))
if not sel.get("criterion_met"):
    raise HardStop("HS-SELECTION: scene_selection.json criterion_met is not true "
                   "(%s); freeze refused, nothing written"
                   % sel.get("criterion_status"))
scenes = sel.get("chosen_scenes", [])
qualifying = sel.get("qualifying_scenes_ascending", [])
if len(scenes) < 2 or any(s in EXCLUDED for s in scenes) \
        or scenes != sorted(scenes) or scenes != list(qualifying):
    raise HardStop("HS-SELECTION: chosen_scenes %r is not ALL qualifying "
                   "ascending held-out scenes (need >= 2); freeze refused, "
                   "nothing written" % (scenes,))

# ---------------------------------------------------------------- translation
trans = {
    "experiment": "EXP-CORRESPONDENCE-GEOM-A",
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
    "why_not_48": "at |dx| = 48 px (3 candidates) the symmetric representability band "
                  "would be [3,8], which reaches >= 50% retention on 0 of 40 hailo_val "
                  "scenes (AUDIT.md section 3, MEASURED). 32 px keeps the band at [2,9]. "
                  "The band actually achieved is recorded in mask_spec.json; "
                  "HS-REPRESENTABILITY re-checks it from the specs in this record "
                  "before any model is instantiated.",
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
ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=SPLIT,
                     disparity_scale=256.0)
per_scene = {}
pooled = 0
for si in scenes:
    sample = ds[si]
    if sample.left.shape[:2] != (H, W):
        raise HardStop("scene %d has shape %s, expected %s"
                       % (si, sample.left.shape[:2], (H, W)))
    gt = sample.disparity.astype(np.float64)
    m = np.zeros_like(gt, dtype=bool)
    m[EDGE:H - EDGE, EDGE:W - EDGE] = True
    m &= (gt > 0)
    m &= ((gt / float(STRIDE)) >= BAND_LO) & ((gt / float(STRIDE)) <= BAND_HI)
    n = int(m.sum())
    if n == 0:
        raise HardStop("scene %d retains 0 mask pixels" % si)
    pooled += n
    d = gt[m] / float(STRIDE)
    per_scene[sample.name] = {"scene_index": si, "retained_pixels": n,
                              "gt_over_16_mean": float(d.mean()),
                              "gt_over_16_median": float(np.median(d)),
                              "gt_over_16_min": float(d.min()),
                              "gt_over_16_max": float(d.max())}

mask = {
    "experiment": "EXP-CORRESPONDENCE-GEOM-A",
    "definition_original_coords": "retain pixel (y,x) iff GT[y,x] > 0 AND "
                                  "GT[y,x]/16 in [2,9] AND "
                                  "%d <= y < %d AND %d <= x < %d"
                                  % (EDGE, H - EDGE, EDGE, W - EDGE),
    "band": {"quantity": "GT/16", "lo": BAND_LO, "hi": BAND_HI,
             "gt_scale": 256.0},
    "edge_exclusion_px": EDGE,
    "edge_exclusion_decomposition": {"crop_margin_M": M, "inner_border_B": B},
    "edge_derivation": "EDGE = M + B = 32 + 64 = 96. M = 32 is this experiment's "
                       "crop margin (max|Delta|). B = 64 is the inner border GEOM-001 "
                       "used (its mask_spec.json edge_exclusion_decomposition: "
                       "M = 48, B = 64, EDGE = 112). The brief requires the border to "
                       "be at least M plus the border GEOM-001 used; 32 + 64 = 96 is "
                       "that minimum, adopted exactly, so the crop window "
                       "[M, H-M) x [M, W-M) always contains the mask interior with "
                       "a further 64 px of real image content on every side.",
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
    "per_scene": per_scene,
    "pooled_retained_pixels": pooled,
    "scenes": scenes,
    "split": SPLIT,
}

# ---------------------------------------------------------------------- write
for name, payload in (("translation_spec.json", trans), ("mask_spec.json", mask)):
    body = json.dumps(payload, indent=2)
    write_lf(OUT / name, body)
    print("%-24s sha256 %s" % (name, hashlib.sha256(body.encode("utf-8")).hexdigest()))

names = ("translation_spec.json", "mask_spec.json", "scene_selection.json")
digests = {n: sha256_bytes(OUT / n) for n in names}
write_lf(OUT / "frozen.sha256",
         "".join("%s  %s\n" % (digests[n], n) for n in names))
for n in names:
    print("frozen %-22s %s" % (n, digests[n]))
print("pooled retained pixels:", pooled)
print("per-scene:", {k: v["retained_pixels"] for k, v in per_scene.items()})
