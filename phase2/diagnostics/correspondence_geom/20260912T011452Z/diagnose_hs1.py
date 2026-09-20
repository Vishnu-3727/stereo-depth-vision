"""HS1 DIAGNOSTIC -- why the matched cost volume is not exactly zero at k = tau.

This measures the ZERO-SET OF THE COST VOLUME ONLY -- a wiring quantity.
It does NOT compute the experiment's statistic (no aggregation is run, no
interaction, no argmin, no h).  It exists to decide whether the fired hard stop
is a harness bug or a false premise in the stop's derivation.

Inference only. No training. Nothing in any record is modified.
"""
import sys
from pathlib import Path

REC = Path(r"C:\Users\vishn\stereo_depth_vision\phase2\diagnostics"
           r"\correspondence_geom\20260912T011452Z")
sys.path.insert(0, str(REC))

import json                                          # noqa: E402
import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import run_geom as G                                 # noqa: E402

dev = G.setup_determinism()
spec = json.loads((REC / "spec.json").read_text(encoding="utf-8"))
model, _ = G.load_model(spec, "H2_seed0", dev)

sc = G.scene_crops(0, dev)
A = sc["A"]
print("scene:", A["name"], " left crop", A["left"].shape)

from src.datasets.kitti2015 import normalize          # noqa: E402

lf = model.feature_extractor(torch.from_numpy(normalize(A["left"])).to(dev))
print("Lf shape", tuple(lf.shape), " |Lf| max %.4g" % float(lf.abs().max()))

for tau in (1, 3, 6):
    t = tau * G.STRIDE
    rc = G.crop(A["src"], 0, t)
    # pixel-level identity (this is what HS1's array check asserts)
    px_ok = np.array_equal(rc[:, :G.CROP_W - t], A["left"][:, t:])
    rf = model.feature_extractor(torch.from_numpy(normalize(rc)).to(dev))

    # feature-level equivariance:  Rf[w] ?= Lf[w + tau]
    fe = (rf[0, :, :, :G.FW - tau] - lf[0, :, :, tau:]).abs()
    per_w = fe.amax(dim=(0, 1)).detach().cpu().numpy()      # max over C,H per column

    v = model.cost_volume(lf, rf)                           # (1,32,12,17,71)
    vz = v[0, :, tau, :, :].abs()                           # the k = tau slice
    vz_w = vz.amax(dim=(0, 1)).detach().cpu().numpy()       # max over C,H per column
    vz_rows = vz.amax(dim=(0, 2)).detach().cpu().numpy()    # max over C,W per row

    zero_cols = np.flatnonzero(vz_w == 0.0)
    print("\ntau=%d t=%dpx   pixel identity right[:, :-t]==left[:, t:] -> %s"
          % (tau, t, px_ok))
    print("  |Lf[w+tau] - Rf[w]| per column, max over C,H:")
    print("    w=0..5   ", np.array2string(per_w[:6], precision=3))
    print("    w=28..34 ", np.array2string(per_w[28:35], precision=3))
    print("    w=50..56 ", np.array2string(per_w[50:57], precision=3))
    print("  |V[:,k=tau,:,w]| exactly-zero columns: n=%d  range=%s"
          % (zero_cols.size,
             ("%d..%d" % (zero_cols.min(), zero_cols.max())) if zero_cols.size else "-"))
    print("  |V| max over the whole k=tau slice: %.6g" % float(vz.max()))
    print("  per-row max (rows 0..16): %s"
          % np.array2string(vz_rows, precision=2, max_line_width=200))
    nz = np.flatnonzero(vz_w != 0.0)
    print("  non-zero columns: %s%s"
          % (nz[:12], " ... " + str(nz[-8:]) if nz.size > 12 else ""))

# does an exactly-zero interior rectangle exist at all?
print("\n--- exact-zero interior rectangle search (tau=1..6) ---")
for tau in G.TAUS:
    t = tau * G.STRIDE
    rf = model.feature_extractor(
        torch.from_numpy(normalize(G.crop(A["src"], 0, t))).to(dev))
    vz = model.cost_volume(lf, rf)[0, :, tau, :, :].abs()
    ok = (vz.amax(dim=0) == 0.0).detach().cpu().numpy()      # (17, 71) bool
    rows = np.flatnonzero(ok.all(axis=1))
    cols = np.flatnonzero(ok.all(axis=0))
    print("tau=%d  all-zero rows %s   all-zero cols %s   total zero cells %d/%d"
          % (tau, rows if rows.size else "none", cols if cols.size else "none",
             int(ok.sum()), ok.size))
