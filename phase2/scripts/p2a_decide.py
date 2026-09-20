"""EXP-P2A-SCALE-COVERAGE-001 — apply the preregistered gates and emit the record.

Reads only the frozen-evaluation JSON produced by eval_p2a.py. Applies the
thresholds exactly as written in phase2/docs/PHASE2_HYPOTHESIS_01.md. No
threshold is recomputed, reinterpreted or fitted here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

E = json.loads((REPO / "phase2" / "runs" / "p2a_scale_coverage"
                / "p2a_eval_best.json").read_text())
S = E["seeds"]

# --- frozen ARM-V control, quoted, not recomputed ---
ARMV = {"epe": [1.8903392, 1.5493088, 1.8785827], "mean": 1.7727436,
        "spread": 0.3410304, "gt_lt_64_mean": 1.2897,
        "gt_ge_64_epe": [13.740, 8.083, 13.443],
        "gt_ge_96_epe": [39.287, 21.708, 38.935],
        "slope_hi": [0.2951, 0.2390, 0.0747],
        "max_pred_valid": [91.53, 113.95, 101.66]}
ACCEPT_GATE = 1.4317132
REFUTE_GATE = 1.7727436
MECH_MEAN_GATE = 0.60
MECH_SEED_GATE = 0.45
REFERENCE = 1.3134471

g = lambda s, *p: (lambda o: o)(
    [S[str(s)]["hailo_val"], *p] and _dig(S[str(s)]["hailo_val"], p))


def _dig(o, path):
    for k in path:
        o = o[k]
    return o


epe = [S[str(s)]["hailo_val"]["metrics"]["epe"] for s in (0, 1, 2)]
d1 = [S[str(s)]["hailo_val"]["metrics"]["d1"] for s in (0, 1, 2)]
slope = [S[str(s)]["hailo_val"]["gt_ge_96"]["slope"] for s in (0, 1, 2)]
r2 = [S[str(s)]["hailo_val"]["gt_ge_96"]["r2"] for s in (0, 1, 2)]
lo64 = [S[str(s)]["hailo_val"]["gt_lt_64"]["epe"] for s in (0, 1, 2)]
hi64 = [S[str(s)]["hailo_val"]["gt_ge_64"]["epe"] for s in (0, 1, 2)]
hi96 = [S[str(s)]["hailo_val"]["gt_ge_96"]["epe"] for s in (0, 1, 2)]
sg64 = [S[str(s)]["hailo_val"]["gt_ge_64"]["signed_err"] for s in (0, 1, 2)]
sg96 = [S[str(s)]["hailo_val"]["gt_ge_96"]["signed_err"] for s in (0, 1, 2)]
tr = [S[str(s)]["hailo_calib"]["metrics"]["epe"] for s in (0, 1, 2)]
trslope = [S[str(s)]["hailo_calib"]["gt_ge_96"].get("slope") for s in (0, 1, 2)]

mean_epe = float(np.mean(epe))
mean_slope = float(np.mean(slope))

if mean_epe < ACCEPT_GATE:
    acc = "ACCEPTED"
elif mean_epe < REFUTE_GATE:
    acc = "INCONCLUSIVE"
else:
    acc = "REFUTED"
mech = ("MECHANISM CONFIRMED"
        if (mean_slope >= MECH_MEAN_GATE and min(slope) >= MECH_SEED_GATE)
        else "MECHANISM NOT CONFIRMED")

print("=" * 74)
print("EXP-P2A-SCALE-COVERAGE-001 — preregistered decisions")
print("=" * 74)
print("contract_match per seed:", [S[str(s)]["hailo_val"]["guard"]["contract_match"]
                                   for s in (0, 1, 2)])
print("valid pixels per seed :", [S[str(s)]["hailo_val"]["metrics"]["valid_pixels"]
                                  for s in (0, 1, 2)])
print("train pixels per seed :", [S[str(s)]["hailo_calib"]["metrics"]["valid_pixels"]
                                  for s in (0, 1, 2)])
print("params / keys         :", [(S[str(s)]["params"], S[str(s)]["n_keys"],
                                   S[str(s)]["state_dict_keys_match_arm_v"])
                                  for s in (0, 1, 2)])
print()
print("global EPE   P2A: %.7f  %.7f  %.7f   mean %.7f  spread %.7f"
      % (*epe, mean_epe, max(epe) - min(epe)))
print("global EPE ARM-V: %.7f  %.7f  %.7f   mean %.7f  spread %.7f"
      % (*ARMV["epe"], ARMV["mean"], ARMV["spread"]))
print("delta vs ARM-V mean: %+.7f px" % (mean_epe - ARMV["mean"]))
print("gap to frozen reference %.7f: %+.7f px" % (REFERENCE, mean_epe - REFERENCE))
print()
print("ACCURACY GATE  (<%.7f ACCEPTED | <%.7f INCONCLUSIVE | else REFUTED)"
      % (ACCEPT_GATE, REFUTE_GATE))
print("   mean %.7f  ->  %s   (margin to ACCEPTED: %+.7f px)"
      % (mean_epe, acc, mean_epe - ACCEPT_GATE))
print()
print("MECHANISM GATE (mean slope_hi >= %.2f AND every seed >= %.2f)"
      % (MECH_MEAN_GATE, MECH_SEED_GATE))
print("   slope_hi P2A : %+.4f  %+.4f  %+.4f   mean %+.4f  min %+.4f"
      % (*slope, mean_slope, min(slope)))
print("   slope_hi ARMV: %+.4f  %+.4f  %+.4f   mean %+.4f"
      % (*ARMV["slope_hi"], float(np.mean(ARMV["slope_hi"]))))
print("   -> %s   (mean margin %+.4f, min-seed margin %+.4f)"
      % (mech, mean_slope - MECH_MEAN_GATE, min(slope) - MECH_SEED_GATE))
print()
print("SECONDARY COST  GT<64 EPE  P2A %.4f %.4f %.4f  mean %.4f  vs ARM-V %.4f  (%+.4f)"
      % (*lo64, float(np.mean(lo64)), ARMV["gt_lt_64_mean"],
         float(np.mean(lo64)) - ARMV["gt_lt_64_mean"]))
print()
out = {
    "experiment": "EXP-P2A-SCALE-COVERAGE-001",
    "control_arm_v": ARMV, "frozen_reference": REFERENCE,
    "gates": {"accept": ACCEPT_GATE, "refute": REFUTE_GATE,
              "mech_mean": MECH_MEAN_GATE, "mech_seed": MECH_SEED_GATE},
    "p2a": {"epe": epe, "epe_mean": mean_epe,
            "epe_spread": float(max(epe) - min(epe)), "d1": d1,
            "rmse": [S[str(s)]["hailo_val"]["metrics"]["rmse"] for s in (0, 1, 2)],
            "bad1": [S[str(s)]["hailo_val"]["metrics"]["bad1"] for s in (0, 1, 2)],
            "bad2": [S[str(s)]["hailo_val"]["metrics"]["bad2"] for s in (0, 1, 2)],
            "bad3": [S[str(s)]["hailo_val"]["metrics"]["bad3"] for s in (0, 1, 2)],
            "gt_lt_64_epe": lo64, "gt_ge_64_epe": hi64, "gt_ge_96_epe": hi96,
            "gt_ge_64_signed": sg64, "gt_ge_96_signed": sg96,
            "slope_hi": slope, "slope_hi_mean": mean_slope, "r2_hi": r2,
            "max_pred_valid": [S[str(s)]["hailo_val"]["max_pred_valid_px"] for s in (0, 1, 2)],
            "max_pred_all": [S[str(s)]["hailo_val"]["max_pred_all_px"] for s in (0, 1, 2)],
            "train_epe": tr, "train_slope_hi": trslope,
            "wall_clock_s": [S[str(s)]["wall_clock_s"] for s in (0, 1, 2)],
            "sha256": [S[str(s)]["sha256"] for s in (0, 1, 2)]},
    "decisions": {"accuracy": acc, "mechanism": mech,
                  "accuracy_margin_to_accepted_px": mean_epe - ACCEPT_GATE,
                  "mechanism_mean_margin": mean_slope - MECH_MEAN_GATE},
}
(REPO / "phase2" / "runs" / "p2a_scale_coverage" / "P2A_DECISIONS.json").write_text(
    json.dumps(out, indent=2))

print("per-bin EPE (val)")
print("| GT bin | ARM-V-era px | P2A s0 | P2A s1 | P2A s2 | signed s0/s1/s2 |")
print("|---|---|---|---|---|---|")
for i, b in enumerate(S["0"]["hailo_val"]["bins"]):
    if "epe" not in b:
        print("| [%d,%d) | %d | INSUFFICIENT | | | |" % (b["lo"], b["hi"], b["px"]))
        continue
    e = [S[str(s)]["hailo_val"]["bins"][i]["epe"] for s in (0, 1, 2)]
    sg = [S[str(s)]["hailo_val"]["bins"][i]["signed_err"] for s in (0, 1, 2)]
    print("| [%d,%d) | %d | %.3f | %.3f | %.3f | %+.2f / %+.2f / %+.2f |"
          % (b["lo"], b["hi"], b["px"], *e, *sg))
print()
print("train split EPE: %.4f %.4f %.4f  | train slope_hi: %s"
      % (*tr, ["%.4f" % x if x is not None else "n/a" for x in trslope]))
print("max pred valid : %s" % ["%.2f" % S[str(s)]["hailo_val"]["max_pred_valid_px"]
                               for s in (0, 1, 2)])
print("max pred all   : %s" % ["%.2f" % S[str(s)]["hailo_val"]["max_pred_all_px"]
                               for s in (0, 1, 2)])
print("wall clock (s) : %s" % ["%.1f" % S[str(s)]["wall_clock_s"] for s in (0, 1, 2)])
