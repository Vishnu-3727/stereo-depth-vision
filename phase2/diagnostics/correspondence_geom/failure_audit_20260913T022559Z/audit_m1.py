"""Audit measurement M1: is the response profile tau-covariant?

Model A (shift-covariant)   Ibar_tau(k) = Phi(k - tau)
Model B (tau-constant)      Ibar_tau(k) = Psi(k)
Both fitted by cell-means on the SAME set of (tau,k) cells, so R^2 is comparable.
Random weights only.  No trained response anywhere in this file.
"""
import json, sys, collections
import numpy as np
from pathlib import Path

D = Path(sys.argv[1])
raw = json.loads((D / "gate_raw.json").read_text(encoding="utf-8"))
TAUS = [1, 2, 3, 4, 5, 6]
LAGS = list(range(-1, 6))          # d common to every tau: k = tau+d in [0,12)

rows = []
for r in raw:
    P = np.array([[r["tau"][str(t)]["Ibar"][t + d] for d in LAGS] for t in TAUS])
    rms = np.sqrt((P ** 2).mean())
    if rms == 0:
        continue
    P = P / rms                                   # per-unit scale normalisation
    tot = ((P - P.mean()) ** 2).sum()
    # Model A: one value per lag d
    phi = P.mean(axis=0)
    resA = ((P - phi[None, :]) ** 2).sum()
    # Model B: one value per absolute candidate k
    ks = np.array([[t + d for d in LAGS] for t in TAUS])
    psi = {}
    for k in np.unique(ks):
        psi[k] = P[ks == k].mean()
    resB = sum((P[i, j] - psi[ks[i, j]]) ** 2 for i in range(len(TAUS))
               for j in range(len(LAGS)))
    rows.append((1 - resA / tot, 1 - resB / tot, r["h"],
                 r["extractor"], r["seed"], r["pair"]))

a = np.array([x[0] for x in rows]); b = np.array([x[1] for x in rows])
h = np.array([x[2] for x in rows])
print("units analysed:", len(rows))
print("R^2 shift-covariant Phi(k-tau) : mean %.4f  median %.4f  p05 %.4f  min %.4f"
      % (a.mean(), np.median(a), np.percentile(a, 5), a.min()))
print("R^2 tau-constant   Psi(k)      : mean %.4f  median %.4f  p95 %.4f  max %.4f"
      % (b.mean(), np.median(b), np.percentile(b, 95), b.max()))
print("units where Phi beats Psi      : %d / %d" % (int((a > b).sum()), len(rows)))
for lo, hi in [(0.99, 1.01), (0.95, 0.99), (0.90, 0.95), (0.0, 0.90)]:
    print("  R^2_Phi in [%.2f,%.2f): %d units" % (lo, hi, int(((a >= lo) & (a < hi)).sum())))
for hv in range(7):
    m = h == hv
    if m.sum():
        print("  h=%d : n=%3d  R^2_Phi mean %.4f  R^2_Psi mean %.4f"
              % (hv, int(m.sum()), a[m].mean(), b[m].mean()))
