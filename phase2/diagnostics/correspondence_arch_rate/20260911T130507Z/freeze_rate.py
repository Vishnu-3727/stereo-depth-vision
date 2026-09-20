"""EXP-CORRESPONDENCE-ARCH-RATE-001 -- freeze construction, mask and classification specs.

Steps 1-2 of the preregistration order. Run ONCE, before PREREGISTRATION.md.
Consults NO ground truth, NO checkpoint, NO model, and no response data of any
kind. Pure integer geometry plus a GT-free scene-selection rule.
"""
import hashlib, json, sys
from pathlib import Path

TS = "20260911T130507Z"
OUT = Path(r"C:\Users\vishn\stereo_depth_vision\phase2\diagnostics\correspondence_arch_rate") / TS
REPO = Path(r"C:\Users\vishn\stereo_depth_vision")
sys.path.insert(0, str(REPO))
from phase2.viz import core  # noqa: E402

W, H, STRIDE = 1232, 368, 16
TMAX = 96
CROP_W, CROP_H = W - TMAX, H - TMAX          # 1136 x 272
BORDER = 64
T_LEVELS = [0, 16, 32, 48, 64, 80, 96]
T_FIT = [16, 32, 48, 64, 80, 96]
AXES = ["horizontal", "vertical"]
N_SEEDS = 32
D = 12
USED_GEOM001, USED_GEOM002 = [27, 0, 31, 6], [1, 2, 3, 4]

names = core.scene_names(split="hailo_val")
used = set(USED_GEOM001) | set(USED_GEOM002)
SCENES = [i for i in range(len(names)) if i not in used][:4]

construction = {
    "experiment": "EXP-CORRESPONDENCE-ARCH-RATE-001",
    "construction": "synthetic fronto-parallel self-pair; both images are crops of the SAME "
                    "source image (scene.left); no fill, no padding, no interpolation",
    "inherited_from": "phase2/diagnostics/correspondence_geom/20260911T115917Z/"
                      "construction_spec.json (sha256 a9e8f250...), unchanged",
    "source_size": {"width": W, "height": H},
    "crop_size": {"width": CROP_W, "height": CROP_H},
    "crop_size_in_feature_samples": {"width": CROP_W // STRIDE, "height": CROP_H // STRIDE},
    "left_crop": "rows [0, %d), cols [0, %d)   -- FIXED for every condition" % (CROP_H, CROP_W),
    "right_crop_horizontal": "rows [0, %d), cols [t, %d + t)" % (CROP_H, CROP_W),
    "right_crop_vertical": "rows [t, %d + t), cols [0, %d)" % (CROP_H, CROP_W),
    "disparity_algebra": "d = a_R - a_L = t, uniform over the crop",
    "t_levels_px": T_LEVELS,
    "t_fit_levels_px": T_FIT,
    "true_disparity_candidates": {str(t): t // STRIDE for t in T_LEVELS},
    "axes": AXES,
    "n_distinct_conditions": 2 * (len(T_LEVELS) - 1) + 1,
    "all_t_multiple_of_stride": all(t % STRIDE == 0 for t in T_LEVELS),
    "all_windows_inside_source": all(0 <= t and t + CROP_W <= W and t + CROP_H <= H
                                     for t in T_LEVELS),
    "ground_truth_used": False,
    "interpolation": "none -- all crops are integer-aligned",
}

randomisation = {
    "experiment": "EXP-CORRESPONDENCE-ARCH-RATE-001",
    "randomised_module": "aggregation ONLY",
    "architecture": "4 x [Conv3d(32->32, 3x3x3, padding=1) + LeakyReLU(0.01)] "
                    "then Conv3d(32->1, 3x3x3, padding=1)",
    "constructor": "Aggregation(in_channels=32, channels=32, num_layers=4)",
    "param_count": 111585,
    "initialisation": "framework default (torch.nn.modules.conv._ConvNd.reset_parameters): "
                      "kaiming_uniform_(weight, a=sqrt(5)); bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))",
    "seed_procedure": "torch.manual_seed(seed) immediately before construction",
    "seeds": list(range(N_SEEDS)),
    "n_seeds": N_SEEDS,
    "rate_resolution": 1.0 / N_SEEDS,
    "frozen_NOT_randomised": ["feature_extractor (trained, frozen)",
                              "cost_volume (no parameters)",
                              "regression / readout (zero parameters, verified from source)",
                              "refinement (never invoked)"],
    "feature_extractor_sources": ["POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"],
    "feature_extractor_note": "the trained feature extractor supplies Lf and therefore the cost "
                              "volume; it must stay frozen or the V=0-at-kappa=0 structure is "
                              "destroyed. All three are used so the rate is not tied to one. "
                              "NO trained aggregation is ever run.",
    "trained_aggregation_measured": False,
    "training_performed": False,
    "scope_limitation": "the result characterises FRAMEWORK-DEFAULT initialisation at this "
                        "scale, not all untrained networks.",
}

classification = {
    "experiment": "EXP-CORRESPONDENCE-ARCH-RATE-001",
    "unit": "(feature_extractor, scene, random_seed)",
    "n_units": 3 * len(SCENES) * N_SEEDS,
    "statistic_alpha": "free-intercept OLS slope of m_axis(t) on t/16 over t in T_FIT "
                       "(NOT through the origin; the through-origin form is the proven "
                       "GEOM-002 defect)",
    "conditions": {
        "Q1_sign": "alpha_h > 0",
        "Q2_monotone": "m_h(t) strictly increasing over all 7 levels",
        "Q3_axis_specific": "|alpha_h| > |alpha_v|",
    },
    "conditions_note": "these are exactly the threshold-free conditions that would be used to "
                       "qualify a TRAINED model. If they are not identical, the measurement "
                       "does not bear on diagnosticity.",
    "classification_order": [
        "NONFINITE", "CLIPPED", "FLAT", "ANTI_CORRELATED", "NON_MONOTONE",
        "NON_SPECIFIC", "QUALIFYING"],
    "class_definitions": {
        "NONFINITE": "any m value not finite -> IMPLEMENTATION FAULT -> HARD STOP (not an outcome)",
        "CLIPPED": "min_t m_h(t) <= 1.0 OR max_t m_h(t) >= %.1f" % (D - 2.0),
        "FLAT": "max_t m_h(t) - min_t m_h(t) < 1.0",
        "ANTI_CORRELATED": "alpha_h < 0",
        "NON_MONOTONE": "Q2 fails (some first difference <= 0), not already FLAT",
        "NON_SPECIFIC": "Q1 and Q2 hold but Q3 fails",
        "QUALIFYING": "Q1 and Q2 and Q3 all hold",
    },
    "threshold_provenance": "the two magnitude cuts (1.0 candidate from each rail for CLIPPED; "
                            "range < 1.0 candidate for FLAT) are derived from the CANDIDATE GRID "
                            "SPACING of the architecture (one candidate), not from any observed "
                            "response. No threshold in this specification derives from any "
                            "previous experiment's output.",
    "primary_result": "the RATE k/N_units of QUALIFYING units, reported with the full "
                      "classification tally. NOT a pass/fail verdict.",
    "clipping_must_not_be_folded": True,
    "raw_publication_requirement": "every m_h(t) and m_v(t) value for every unit must be "
                                   "published, so any reader can reclassify under an alternative "
                                   "definition without re-running the experiment.",
    "mask": {
        "definition_crop_coords": "retain (y,x) iff %d <= y < %d AND %d <= x < %d"
                                  % (BORDER, CROP_H - BORDER, BORDER, CROP_W - BORDER),
        "border_px": BORDER,
        "retained_pixels_per_scene": (CROP_W - 2 * BORDER) * (CROP_H - 2 * BORDER),
        "gt_dependent": False, "t_dependent": False, "axis_dependent": False,
        "model_dependent": False, "scene_dependent": False,
    },
    "scenes": SCENES,
    "scene_names": [names[i] for i in SCENES],
    "scene_selection_rule": "first four hailo_val indices excluding BOTH prior correspondence "
                            "sets {27,0,31,6} and {1,2,3,4}; no ground truth consulted",
    "prior_scene_sets_excluded": {"GEOM-001": USED_GEOM001, "GEOM-002": USED_GEOM002},
    "split": "hailo_val",
}

digests = {}
for name, payload in (("construction_spec.json", construction),
                      ("randomisation_spec.json", randomisation),
                      ("classification_spec.json", classification)):
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

print()
print("scenes           :", SCENES, classification["scene_names"])
print("seeds            :", N_SEEDS, " rate resolution 1/%d = %.4f" % (N_SEEDS, 1.0 / N_SEEDS))
print("units            :", classification["n_units"])
print("conditions/unit  :", construction["n_distinct_conditions"])
print("cost volumes     :", 3 * len(SCENES) * construction["n_distinct_conditions"],
      "(cached; independent of the aggregation seed)")
print("agg+readout arms :", classification["n_units"] * construction["n_distinct_conditions"])
print("mask px/scene    :", classification["mask"]["retained_pixels_per_scene"])
