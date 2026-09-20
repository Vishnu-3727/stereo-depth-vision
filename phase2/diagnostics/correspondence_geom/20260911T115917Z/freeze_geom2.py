"""EXP-CORRESPONDENCE-GEOM-002 -- freeze construction and mask specifications.

Steps 1-2 of the preregistration order. Run ONCE, before PREREGISTRATION.md is
written; the resulting file-byte digests are recorded in it.

Consults NO ground truth, NO checkpoint, NO model, and no GEOM-002 response data
(none exists). Pure integer geometry plus a GT-free scene-selection rule.
"""
import hashlib, json, sys
from pathlib import Path

TS = "20260911T115917Z"
OUT = Path(r"C:\Users\vishn\stereo_depth_vision\phase2\diagnostics\correspondence_geom") / TS
REPO = Path(r"C:\Users\vishn\stereo_depth_vision")
sys.path.insert(0, str(REPO))
from phase2.viz import core  # noqa: E402

W, H, STRIDE = 1232, 368, 16
TMAX = 96
CROP_W, CROP_H = W - TMAX, H - TMAX          # 1136 x 272
BORDER = 64
T_LEVELS = [0, 16, 32, 48, 64, 80, 96]
AXES = ["horizontal", "vertical"]
FOCUS_GEOM001 = [27, 0, 31, 6]

names = core.scene_names(split="hailo_val")
SCENES = [i for i in range(len(names)) if i not in FOCUS_GEOM001][:4]

construction = {
    "experiment": "EXP-CORRESPONDENCE-GEOM-002",
    "construction": "synthetic fronto-parallel self-pair; both images are crops of the SAME "
                    "source left image L; no fill, no padding, no interpolation",
    "source_image": "scene.left only (the scene's own right image is never used)",
    "source_size": {"width": W, "height": H},
    "crop_size": {"width": CROP_W, "height": CROP_H},
    "crop_size_in_feature_samples": {"width": CROP_W // STRIDE, "height": CROP_H // STRIDE},
    "left_crop": "rows [0, %d), cols [0, %d)   -- FIXED for every condition" % (CROP_H, CROP_W),
    "right_crop_horizontal": "rows [0, %d), cols [t, %d + t)" % (CROP_H, CROP_W),
    "right_crop_vertical": "rows [t, %d + t), cols [0, %d)" % (CROP_H, CROP_W),
    "t_levels_px": T_LEVELS,
    "axes": AXES,
    "n_distinct_conditions": 2 * (len(T_LEVELS) - 1) + 1,
    "disparity_algebra": "left[x]=L[a_L+x], right[x]=L[a_R+x]; a point at source col X has "
                         "x_L=X-a_L, x_R=X-a_R, so d = x_L-x_R = a_R-a_L. With a_L=0 and "
                         "a_R=t the imposed disparity is EXACTLY t px everywhere.",
    "sign_check": "matches phase2/viz/core.correspondence: x_R = x_L - d",
    "sign_correction_note": "an earlier draft (successor_audit_20260911/RECOMMENDATION.md) "
                            "specified the right crop at cols [96-t, 1136-t), i.e. a_R = a_L - t, "
                            "which imposes d = -t, a NEGATIVE and unrepresentable disparity. "
                            "Corrected here to a_R = a_L + t. The draft is preserved unmodified.",
    "true_disparity_candidates": {str(t): t // STRIDE for t in T_LEVELS},
    "anchor_slope_candidates_per_candidate": 1.0,
    "all_t_multiple_of_stride": all(t % STRIDE == 0 for t in T_LEVELS),
    "all_windows_inside_source": all(0 <= t and t + CROP_W <= W and t + CROP_H <= H
                                     for t in T_LEVELS),
    "rail_clearance": {str(t): {"candidate": t // STRIDE,
                                "dist_to_low_rail": t // STRIDE,
                                "dist_to_high_rail": 11 - t // STRIDE} for t in T_LEVELS},
    "why_t_max_96": "candidate 6 of 0..11 keeps the predicted response clear of both rails "
                    "while the crop stays stride-aligned at 1136 x 272. Chosen from the "
                    "candidate axis and crop arithmetic alone.",
    "ground_truth_used": False,
    "ground_truth_note": "the true disparity is t by construction and uniform over the crop, "
                         "so no GT file, no GT band and no scene-disparity selection is needed. "
                         "This removes the representability defect that ended GEOM-001.",
    "interpolation": "none -- all crops are integer-aligned",
}

mask = {
    "experiment": "EXP-CORRESPONDENCE-GEOM-002",
    "definition_crop_coords": "retain (y,x) iff %d <= y < %d AND %d <= x < %d"
                              % (BORDER, CROP_H - BORDER, BORDER, CROP_W - BORDER),
    "border_px": BORDER,
    "retained_size": {"width": CROP_W - 2 * BORDER, "height": CROP_H - 2 * BORDER},
    "retained_pixels_per_scene": (CROP_W - 2 * BORDER) * (CROP_H - 2 * BORDER),
    "single_fixed_mask": True,
    "t_dependent": False, "axis_dependent": False, "model_dependent": False,
    "scene_dependent": False, "gt_dependent": False,
    "why_no_gt": "the imposed disparity is t everywhere, so ground truth contributes nothing "
                 "and is not consulted. This eliminates the band mask, the scene-selection "
                 "question and the mask-occupancy confound present since INDEX-001.",
    "why_a_border": "the feature receptive field is about 477 px, which no affordable border "
                    "can exclude (the crop is only 272 px tall). The border reduces, but does "
                    "not remove, the influence of differing content near the crop edge. The "
                    "search-free control measures whatever residual survives.",
    "scene_selection_rule": "the first four indices of hailo_val excluding the GEOM-001 focus "
                            "set {27,0,31,6}; no ground truth consulted",
    "scenes": SCENES,
    "scene_names": [names[i] for i in SCENES],
    "geom001_focus_excluded": FOCUS_GEOM001,
    "split": "hailo_val",
}

digests = {}
for name, payload in (("construction_spec.json", construction), ("mask_spec.json", mask)):
    body = json.dumps(payload, indent=2)
    p = OUT / name
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    digests[name] = d
    print("%-26s sha256 %s" % (name, d))

with open(OUT / "frozen.sha256", "w", encoding="utf-8", newline="") as fh:
    for n, d in digests.items():
        fh.write("%s  %s\n" % (d, n))
print("scenes:", SCENES, [names[i] for i in SCENES])
print("conditions:", construction["n_distinct_conditions"])
print("retained pixels per scene:", mask["retained_pixels_per_scene"])
print("forward passes: 4 models x %d scenes x %d conditions = %d"
      % (len(SCENES), construction["n_distinct_conditions"],
         4 * len(SCENES) * construction["n_distinct_conditions"]))
