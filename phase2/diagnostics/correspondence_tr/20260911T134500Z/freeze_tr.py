"""EXP-CORRESPONDENCE-TR-001 -- freeze spec.json and the random reference.

Run ONCE, before PREREGISTRATION.md and before ANY trained aggregation is
executed. Reads only already-published records and source; computes no trained
response.
"""
import hashlib, json, shutil, sys
from pathlib import Path

TS = "20260911T134500Z"
ROOT = Path(r"C:\Users\vishn\stereo_depth_vision")
OUT = ROOT / "phase2/diagnostics/correspondence_tr" / TS
AR = ROOT / "phase2/diagnostics/correspondence_arch_rate/20260911T130507Z"
SCRATCH = Path(r"C:\Users\vishn\AppData\Local\Temp\claude\C--WINDOWS-system32\4a814cae-d81d-4498-a709-595455f230cd\scratchpad")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def fh(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- copy the frozen random reference verbatim -------------------------------
shutil.copy2(SCRATCH / "random_reference.json", OUT / "random_reference.json")

CKPT = {
    "H2_seed0": {"path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
                 "sha256": "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a"},
    "H2_seed1": {"path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
                 "sha256": "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf"},
    "H2_seed2": {"path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
                 "sha256": "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69"},
    "NEG_shift_none": {"path": "phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth",
                       "sha256": "d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1"},
}
for k, v in CKPT.items():
    a = fh(ROOT / v["path"])
    if a != v["sha256"]:
        raise SystemExit("HARD STOP: %s checkpoint hash mismatch" % k)

ref = json.loads((OUT / "random_reference.json").read_text(encoding="utf-8"))

spec = {
    "experiment_id": "EXP-CORRESPONDENCE-TR-001",
    "title": "Trained-vs-Random Geometric Separation",
    "kind": "inference-only; no training; distributional comparison against a "
            "hash-verified pre-existing random population",

    "primary_question": "Under the same synthetic geometric construction, does the trained "
                        "aggregation produce a response distinguishable from the empirically "
                        "observed random-weight population?",

    "checkpoints": CKPT,
    "trained_checkpoints": ["H2_seed0", "H2_seed1", "H2_seed2"],
    "search_free_control": "NEG_shift_none",

    "construction": {
        "inherited_from": "correspondence_arch_rate/20260911T130507Z/construction_spec.json",
        "inherited_sha256": ref["construction_spec_sha256"],
        "rule": "self-pair crops of scene.left; a_R = a_L + t; imposed disparity d = t px "
                "exactly and uniformly; candidate coordinate k_true = t/16",
        "left_crop": "rows [0,272), cols [0,1136)   FIXED every condition",
        "right_crop_horizontal": "rows [0,272), cols [t, 1136+t)",
        "right_crop_vertical": "rows [t, 272+t), cols [0,1136)",
        "crop_size": {"width": 1136, "height": 272},
        "t_levels_px": [0, 16, 32, 48, 64, 80, 96],
        "u_levels": [t / 16 for t in [0, 16, 32, 48, 64, 80, 96]],
        "fill_padding_interpolation": "NONE -- integer slicing only",
        "sign_verification_required": True,
    },

    "mask": {
        "inherited_from": "correspondence_arch_rate/20260911T130507Z/classification_spec.json",
        "inherited_sha256": ref["classification_spec_sha256"],
        "definition_crop_coords": "retain (y,x) iff 64 <= y < 208 AND 64 <= x < 1072",
        "retained_pixels_per_scene": 145152,
        "gt_dependent": False,
    },

    "scenes": ref["scenes"],
    "scene_names": ref["scene_names"],
    "scene_note": "IDENTICAL to the random reference. Required for an apples-to-apples "
                  "comparison under identical input construction.",

    "primary_statistic": {
        "name": "alpha_h",
        "definition": "FREE-INTERCEPT ordinary-least-squares slope of m_h(t) on u = t/16 "
                      "over ALL SEVEN levels t in {0,16,32,48,64,80,96}; m_h(t) = median "
                      "predicted candidate disparity over the fixed mask",
        "intercept": "FREE -- the through-origin form is prohibited (GEOM-002 defect)",
        "identical_to_reference": True,
        "reference_definition_sha256": ref["classification_spec_sha256"],
    },

    "secondary_statistics": {
        "alpha_v": "same fit on the vertical arm; control axis; NO expectation that it is zero",
        "R_h": "max(m_h) - min(m_h); recorded, no unregistered threshold applied",
        "first_differences": "Delta_i = m_h(t_i) - m_h(t_{i-1}), all six; positive count, "
                             "negative count, min, max",
    },

    "classification": {
        "inherited_verbatim_from_ARCH_RATE": True,
        "order": ["NONFINITE", "CLIPPED", "FLAT", "ANTI_CORRELATED", "NON_MONOTONE",
                  "NON_SPECIFIC", "QUALIFYING"],
        "CLIPPED": "min_t m_h(t) <= 1.0 OR max_t m_h(t) >= 10.0",
        "FLAT": "max_t m_h(t) - min_t m_h(t) < 1.0",
        "threshold_provenance": "candidate grid spacing (1 candidate); inherited unchanged "
                                "from ARCH-RATE-001; NOT derived from any trained observation",
        "precedence_note": "CLIPPED is evaluated before every ordinary class. A clipped "
                           "response can never be classified monotonic.",
    },

    "unit_structure": {
        "primary": "MATCHED units: checkpoint c supplies BOTH its feature extractor and its "
                   "aggregation. 3 checkpoints x 4 scenes = 12 units. These are the real models.",
        "secondary": "FULL CROSS: 3 feature extractors x 3 trained aggregations x 4 scenes = "
                     "36 units, closest analogue of the ARCH-RATE 3 x 32 x 4 structure. "
                     "Off-diagonal pairs are chimeras, not real models.",
        "n_matched": 12, "n_cross": 36,
        "independence": "Units are NOT independent replicates. They share three checkpoints "
                        "and four scenes, and the three checkpoints originate from related "
                        "training conditions. NO p-value is computed anywhere.",
        "reporting_required": ["per-unit", "grouped by checkpoint", "grouped by scene"],
        "pooling_prohibited": "matched and cross populations are never pooled",
    },

    "random_reference": {
        "source": ref["source_record"],
        "results_sha256": ref["source_results_sha256"],
        "n": ref["n"],
        "alpha_h_observed_range": [ref["alpha_h"]["min"], ref["alpha_h"]["max"]],
        "alpha_v_observed_range": [ref["alpha_v"]["min"], ref["alpha_v"]["max"]],
        "range_h_observed_range": [ref["range_h"]["min"], ref["range_h"]["max"]],
        "alpha_h_summary": ref["alpha_h"], "alpha_v_summary": ref["alpha_v"],
        "regeneration_prohibited": True,
        "seed18_retained_unchanged": True,
    },

    "comparison_procedure": {
        "per_trained_unit": ["empirical percentile rank of alpha_h within the 384 random values",
                             "empirical CDF position",
                             "nearest random observations above and below",
                             "outside-observed-random-range flag"],
        "key_descriptive_quantity": "trained_outside_random_range_rate = "
                                    "(# trained units outside the complete observed random "
                                    "alpha_h range) / (total trained units)",
        "also_computed_for": ["alpha_v", "R_h", "positive first-difference count"],
        "binary_conversion_prohibited": "secondary quantities are reported, never converted "
                                        "into a new binary claim",
        "distributional_requirement": "report trained median/IQR/min/max, random "
                                      "median/IQR/min/max, overlap, outside-range fraction, "
                                      "and a nonparametric effect size (rank-biserial from "
                                      "Mann-Whitney U, reported as an EFFECT SIZE ONLY, "
                                      "with NO p-value)",
        "invalid_inference_explicitly_rejected": "'0/384 random qualified therefore any "
                                                 "trained qualification proves learning' is "
                                                 "INVALID and is not used",
    },

    "decision_logic": {
        "no_arbitrary_pass_threshold": True,
        "CASE_A_CLEAR_SEPARATION": "trained distribution substantially displaced from random, "
                                   "little or no overlap, high outside-range fraction, AND "
                                   "consistent across all three checkpoints and all four scenes",
        "CASE_B_OVERLAP": "trained responses substantially overlap the random population",
        "CASE_C_AMBIGUOUS": "partial separation, mixed checkpoints, strong scene dependence, "
                            "clipping, or other structure preventing a clean conclusion",
        "heterogeneity_rule": "if one checkpoint separates while others overlap -> MIXED -> "
                              "CASE C, never overall success. If one scene drives the result, "
                              "report it explicitly. Heterogeneity is never averaged away.",
    },

    "wiring_checks": {
        "W1": "left input byte-identical across all translation conditions (hash compared)",
        "W2": "right-image construction matches the frozen spec at every level "
              "(crop origin asserted = (0,t) horizontal, (t,0) vertical)",
        "W3": "record k_true = t/16; the model output is NOT required to equal it",
        "W4": "deterministic repeat of one complete trained unit after process restart; "
              "the complete response vector must be BIT-IDENTICAL",
        "no_numeric_t0_expectation": "the invalid 5.5 expectation is never reinstated",
    },

    "search_free_control": {
        "checkpoint": "NEG_shift_none",
        "purpose": "tests whether the trained geometric statistic requires the "
                   "candidate-dependent construction",
        "reporting": "SEPARATE. Never mixed into the random-weight null.",
        "note": "not previously measured on these scenes; run fresh under identical conditions",
    },

    "hard_stops": [
        "trained weights modified", "trained aggregation retrained",
        "random reference not hash-verifiable", "construction differs from this spec",
        "translation levels differ", "mask differs", "preprocessing differs",
        "deterministic repeat fails", "trained curves inspected before freeze",
        "post-hoc statistic introduced", "post-hoc threshold introduced",
        "historical results overwritten",
        "-> VERDICT = NO DETERMINATION; do not continue merely to obtain a result",
    ],

    "claim_ceiling": {
        "maximum": "Under the specified synthetic translation construction, the trained "
                   "aggregation response is / is not distinguishable from the empirically "
                   "sampled random aggregation population.",
        "unavailable": ["genuine stereo correspondence", "correct disparity estimation",
                        "true disparity search", "correct correspondence on natural scenes",
                        "metric depth correctness", "generalisation",
                        "biological/physical scene correspondence"],
        "language_requirement": "'outside the observed random population' -- NEVER 'impossible "
                                "under random weights' and NEVER 'proves learned correspondence'. "
                                "The reference contains 384 sampled parameterisations, not the "
                                "space of random weights.",
    },

    "compute": {
        "cost_volumes": 156,
        "trained_readout_passes": 36 * 14,
        "search_free_readout_passes": 4 * 14,
        "deterministic_repeat_passes": 14,
        "training": False,
    },
}

body = json.dumps(spec, indent=2)
with open(OUT / "spec.json", "w", encoding="utf-8", newline="") as f:
    f.write(body)

digests = {n: fh(OUT / n) for n in ("spec.json", "random_reference.json")}
with open(OUT / "frozen.sha256", "w", encoding="utf-8", newline="") as f:
    for n, d in digests.items():
        f.write("%s  %s\n" % (d, n))

for n, d in digests.items():
    print("%-24s sha256 %s" % (n, d))
print()
print("scenes            :", spec["scenes"], spec["scene_names"])
print("matched units     :", spec["unit_structure"]["n_matched"])
print("cross units       :", spec["unit_structure"]["n_cross"])
print("random reference  : N=%d  alpha_h range [%.6f, %.6f]"
      % (ref["n"], ref["alpha_h"]["min"], ref["alpha_h"]["max"]))
print("checkpoint hashes : all 4 verified")
