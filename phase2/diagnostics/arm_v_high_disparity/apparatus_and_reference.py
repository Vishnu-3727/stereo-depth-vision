"""PHASE-2 STEP 1d — (i) how resolvable is each candidate instrument, and
(ii) the frozen reference ONNX on the same GT bins. INFERENCE ONLY, NO TRAINING.

(i) For every arm with three frozen seeds, report the 3-seed mean and spread of
    global EPE, of EPE over GT >= 96 px, and of the high-stratum slope, so that
    a Phase-2 experiment can be judged on an instrument whose noise is known
    rather than assumed.

(ii) Re-score the frozen reference ONNX on the Phase-2 GT bins, and place both
     against the TRAINING pixel density in each bin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

BINS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]

# ---------- (i) instrument resolvability ----------
cen = json.loads((HERE / "collapse_census.json").read_text())["runs"]
by_arm = {}
for r in cen:
    if r.get("status") == "ok":
        by_arm.setdefault(r["arm"], []).append(r)

print("### instrument resolvability (3-seed arms only)")
print()
print("| arm | global EPE mean | global spread | EPE(GT>=96) mean | spread | spread/mean | slope_hi mean | spread |")
print("|---|---|---|---|---|---|---|---|")
appar = {}
for arm, rs in by_arm.items():
    if len(rs) != 3:
        continue
    g = np.array([r["global_epe"] for r in rs])
    h = np.array([r["epe_ge_96"] for r in rs])
    s = np.array([r["slope_hi"] for r in rs])
    appar[arm] = {"global_mean": float(g.mean()), "global_spread": float(np.ptp(g)),
                  "hi_mean": float(h.mean()), "hi_spread": float(np.ptp(h)),
                  "hi_rel_spread": float(np.ptp(h) / h.mean()),
                  "slope_mean": float(s.mean()), "slope_spread": float(np.ptp(s))}
    print("| %s | %.4f | %.4f | %.3f | %.3f | %.3f | %+.4f | %.4f |" % (
        arm, g.mean(), np.ptp(g), h.mean(), np.ptp(h), np.ptp(h) / h.mean(),
        s.mean(), np.ptp(s)))

print()
print("relative 3-seed spread (spread / mean), lower = more resolvable instrument")
print("| arm | global EPE | EPE(GT>=96) |")
print("|---|---|---|")
for arm, a in appar.items():
    print("| %s | %.3f | %.3f |" % (arm, a["global_spread"] / a["global_mean"],
                                    a["hi_rel_spread"]))

# ---------- (ii) reference ONNX on the Phase-2 bins ----------
import onnxruntime as ort  # noqa: E402
from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402

sess = ort.InferenceSession(str(REPO / "reference" / "onnx" / "stereonet.onnx"),
                            providers=["CPUExecutionProvider"])
ins = [i.name for i in sess.get_inputs()]
outn = sess.get_outputs()[0].name
ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                     disparity_scale=256.0, occluded=True)
P, G = [], []
pmax = -1e30
for i in range(len(ds)):
    s = ds[i]
    o = sess.run([outn], {ins[0]: normalize(s.left), ins[1]: normalize(s.right)})[0]
    p = o[0, 0].astype(np.float64)
    pmax = max(pmax, float(p.max()))
    v = s.disparity > 0
    P.append(p[v]); G.append(s.disparity[v].astype(np.float64))
P, G = np.concatenate(P), np.concatenate(G)
m = pooled_metrics([P], [G])
hi = G >= 96
a, b = np.polyfit(G[hi], P[hi], 1)

# training pixel density per bin
tr = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib",
                     disparity_scale=256.0, occluded=True)
tv = []
for i in range(len(tr)):
    d = tr[i].disparity
    tv.append(d[d > 0].astype(np.float32))
TV = np.concatenate(tv)

print()
print("### frozen reference ONNX, Phase-2 bins  (EPE %.7f, D1 %.4f%%, contract %s)"
      % (m["epe"], m["d1"], refuse_unless_contract(len(ds), m["valid_pixels"], 256.0,
                                                   "hailo_val", "disp_occ_0")["contract_match"]))
print("reference max predicted disparity: %.2f px (valid px %.2f)" % (pmax, P.max()))
print("reference GT>=96: EPE %.3f  mean pred %.2f  slope %+.4f  pearson %+.4f"
      % (np.abs(P[hi] - G[hi]).mean(), P[hi].mean(), a,
         float(np.corrcoef(G[hi], P[hi])[0, 1])))
print()
print("| GT bin | train px frac | val px frac | reference EPE | reference signed |")
print("|---|---|---|---|---|")
rows = []
for lo, hi_ in zip(BINS[:-1], BINS[1:]):
    tm = (TV >= lo) & (TV < hi_)
    vm = (G >= lo) & (G < hi_)
    row = {"lo": lo, "hi": hi_, "train_frac": float(tm.mean()),
           "val_frac": float(vm.mean()),
           "ref_epe": float(np.abs(P[vm] - G[vm]).mean()) if vm.sum() >= 1000 else None,
           "ref_signed": float((P[vm] - G[vm]).mean()) if vm.sum() >= 1000 else None}
    rows.append(row)
    print("| [%d,%d) | %.5f | %.5f | %s | %s |" % (
        lo, hi_, row["train_frac"], row["val_frac"],
        "INSUFFICIENT" if row["ref_epe"] is None else "%.3f" % row["ref_epe"],
        "—" if row["ref_signed"] is None else "%+.3f" % row["ref_signed"]))

(HERE / "apparatus_and_reference.json").write_text(json.dumps(
    {"instrument": appar,
     "reference": {"epe": m["epe"], "d1": m["d1"], "max_pred_all_px": pmax,
                   "max_pred_valid_px": float(P.max()),
                   "epe_ge_96": float(np.abs(P[G >= 96] - G[G >= 96]).mean()),
                   "slope_hi": float(a),
                   "pearson_hi": float(np.corrcoef(G[G >= 96], P[G >= 96])[0, 1])},
     "bins": rows}, indent=2))
