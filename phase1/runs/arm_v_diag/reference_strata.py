"""Same strata decomposition applied to the FROZEN REFERENCE ONNX. Read-only.

Purpose: find out whether the reference model's 0.459 px advantage over the
ARM-V mean lives in the same strata where ARM-V's error mass lives. Uses the
unmodified frozen-contract dataset/metric path.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
import onnxruntime as ort

from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract
from src.datasets.kitti2015 import Kitti2015Stereo, normalize

sess = ort.InferenceSession(str(REPO / "reference" / "onnx" / "stereonet.onnx"),
                            providers=["CPUExecutionProvider"])
ins = [i.name for i in sess.get_inputs()]
outn = sess.get_outputs()[0].name

ds_occ = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
ds_noc = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=False)
P, G, OCC = [], [], []
pmax = -1e9
for i in range(len(ds_occ)):
    s = ds_occ[i]
    o = sess.run([outn], {ins[0]: normalize(s.left), ins[1]: normalize(s.right)})[0]
    p = o[0, 0].astype(np.float64)
    pmax = max(pmax, float(p.max()))
    v = s.disparity > 0
    P.append(p[v]); G.append(s.disparity[v].astype(np.float64))
    OCC.append((v & (ds_noc[i].disparity <= 0))[v])
    print(i + 1, s.name, flush=True)

P, G, OCC = np.concatenate(P), np.concatenate(G), np.concatenate(OCC)
err = np.abs(P - G)
m = pooled_metrics([P], [G])
res = {"epe": m["epe"], "d1": m["d1"],
       "guard": refuse_unless_contract(len(ds_occ), m["valid_pixels"], 256.0,
                                       "hailo_val", "disp_occ_0"),
       "pred_pct": {q: float(np.percentile(P, q)) for q in (50, 90, 99, 99.9, 100)},
       "pred_max_all_px": pmax,
       "gt_bins": [], "occluded": {}}
for a, b in zip([0, 8, 16, 32, 64, 96, 128], [8, 16, 32, 64, 96, 128, 160]):
    k = (G >= a) & (G < b)
    res["gt_bins"].append({"lo": a, "hi": b, "frac_px": float(k.mean()),
                           "epe": float(err[k].mean()) if k.any() else None,
                           "mean_signed_err": float((P[k] - G[k]).mean()) if k.any() else None,
                           "epe_mass_frac": float(err[k].sum() / err.sum())})
res["gt_over_64"] = {"frac_px": float((G > 64).mean()),
                     "epe": float(err[G > 64].mean()),
                     "epe_mass_frac": float(err[G > 64].sum() / err.sum())}
res["occluded"] = {"frac_px": float(OCC.mean()), "epe": float(err[OCC].mean()),
                   "epe_mass_frac": float(err[OCC].sum() / err.sum()),
                   "complement_epe": float(err[~OCC].mean())}
res["error_percentiles"] = {str(q): float(np.percentile(err, q))
                            for q in (50, 75, 90, 95, 99, 99.9)}
(Path(__file__).resolve().parent / "reference_strata.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
