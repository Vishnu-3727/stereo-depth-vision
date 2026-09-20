"""Audit check: is tau a COORDINATE CHANGE on the 4-cell cost-volume family?

Claim: with Rf[w] = Lf[w+tau], every cell is a function of (u, d) alone, where
       u = w + tau  and  d = k - tau:
           V_AA[k,w] = Af[u+d] - Af[u]      V_AB[k,w] = Af[u+d] - Bf[u]
           V_BB[k,w] = Bf[u+d] - Bf[u]      V_BA[k,w] = Bf[u+d] - Af[u]
Test: V^{tau1}[k1,w1] == V^{tau2}[k2,w2] exactly whenever k1-tau1 == k2-tau2
      and w1+tau1 == w2+tau2.

Uses the FROZEN cost-volume operator on RANDOM feature tensors.  No checkpoint
is opened, no trained weight exists in this process, no aggregation runs.
"""
import sys
from pathlib import Path
import torch

REPO = Path("/c/Users/vishn/stereo_depth_vision".replace("/c/", "C:/"))
sys.path.insert(0, str(REPO))
from src.models.stereonet.cost_volume import build_cost_volume   # noqa: E402

torch.manual_seed(0)
FW, FH, C, Dd = 71, 17, 32, 12
TAUS = [1, 2, 3, 4, 5, 6]
# one "long" feature field per scene, wide enough that every tau-shifted crop is
# a true translation of the same underlying field (this is what the synthetic
# pixel construction produces inside the clean band)
LONG = {w: torch.randn(1, C, FH, FW + max(TAUS)) for w in ("A", "B")}
Lf = {w: LONG[w][..., :FW] for w in ("A", "B")}

vols = {}
for tau in TAUS:
    Rf = {w: LONG[w][..., tau:tau + FW] for w in ("A", "B")}   # Rf[w] = Lf[w+tau]
    for cell, (li, ri) in {"AA": ("A", "A"), "BB": ("B", "B"),
                           "AB": ("A", "B"), "BA": ("B", "A")}.items():
        vols[(cell, tau)] = build_cost_volume(Lf[li], Rf[ri], Dd, shift="left")

CV_LO, CV_HI = 15, 44
worst = 0.0
n = 0
for cell in ("AA", "BB", "AB", "BA"):
    for t1 in TAUS:
        for t2 in TAUS:
            if t2 <= t1:
                continue
            s = t2 - t1
            # u = w+tau, d = k-tau  =>  k1 = k2 - s  and  w1 = w2 + s
            k2lo, k2hi = s, Dd - 1
            w2lo, w2hi = CV_LO, CV_HI - s
            a = vols[(cell, t1)][0, :, k2lo - s:k2hi + 1 - s, :, w2lo + s:w2hi + 1 + s]
            b = vols[(cell, t2)][0, :, k2lo:k2hi + 1, :, w2lo:w2hi + 1]
            d = float((a - b).abs().max())
            worst = max(worst, d)
            n += 1
print("reparameterisation checks: %d  (4 cells x 15 tau pairs)" % n)
print("max |V^{t1}[k-s, w+s] - V^{t2}[k, w]| over the clean band = %r" % worst)
print("EXACT" if worst == 0.0 else "NOT EXACT")

# and the d=0 anchor, for the record
z = max(float(vols[(c, t)][0, :, t, :, CV_LO:CV_HI + 1].abs().max())
        for c in ("AA", "BB") for t in TAUS)
print("max |V[:, tau, :, clean band]| for AA/BB = %r" % z)
