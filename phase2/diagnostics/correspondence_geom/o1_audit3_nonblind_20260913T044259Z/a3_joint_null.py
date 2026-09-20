"""THIRD-PASS AUDIT -- two measurements the brief demands.

M-A  THE JOINT SIGNATURE NULL.  The O1 signature is a CONJUNCTION:
     small E_orig AND positive Delta.  Marginal nulls are not enough.
     Is the CONJUNCTION obtainable from a randomly initialised aggregation?

M-B  THE DEGENERACY, QUANTIFIED.  V[tau] is identically zero.  How isolated
     is that slice?  If it is a singularity, 'find the match' and 'detect the
     anomaly' are the same observable.

No checkpoint is opened.  Random feature fields (the frozen extractor lives
inside a checkpoint and is therefore out of bounds for this audit).
"""
import sys
import numpy as np, torch
sys.path.insert(0, "C:/Users/vishn/stereo_depth_vision")
from src.models.stereonet.cost_volume import build_cost_volume            # noqa
from src.models.stereonet.aggregation import Aggregation                  # noqa
from phase2.models.scaled_regression import standardise_across_disparity  # noqa

C, FH, FW, D = 32, 17, 71, 12
TAUS = [1, 2, 3, 4, 5, 6]; W_LO, W_HI = 20, 39

def readout(c):
    z = standardise_across_disparity(c, dim=1); p = torch.softmax(-z, 1)
    k = torch.arange(D, dtype=c.dtype).view(1, D, 1, 1)
    return (p * k).sum(dim=1)
def stat(m): return float(np.median(m[0, :, W_LO:W_HI + 1].numpy()))

def fields(seed):
    torch.manual_seed(500 + seed)
    LONG = torch.randn(1, C, FH, FW + 6)
    Lf = LONG[..., :FW]
    return {t: build_cost_volume(Lf, LONG[..., t:t + FW], D, shift="left") for t in TAUS}

print("=" * 78)
print("M-B  THE DEGENERACY, QUANTIFIED")
print("=" * 78)
V = fields(0)
for t in (2, 4):
    n = V[t][0].pow(2).sum(0).sqrt()          # (D, H, W) L2 norm across channels
    band = n[:, :, W_LO:W_HI + 1]
    matched = float(band[t].abs().max())
    others = torch.cat([band[:t], band[t + 1:]], dim=0)
    print("  tau=%d :  ||V[tau]|| max over the band = %r" % (t, matched))
    print("           ||V[k!=tau]||  min %.3f   median %.3f   max %.3f"
          % (float(others.min()), float(others.median()), float(others.max())))
    print("           gap ratio (min over k!=tau) / (median over k!=tau) = %.3f"
          % (float(others.min()) / float(others.median())))
print()
print("  -> the matched slice is EXACTLY 0 while the nearest competitor sits at")
print("     an appreciable fraction of the typical slice norm.  It is an")
print("     isolated singularity, not merely the smallest of a graded set.")
print("     'smallest' and 'anomalous' therefore select the SAME slice.")

print()
print("=" * 78)
print("M-A  THE JOINT SIGNATURE NULL AT RANDOM INITIALISATION")
print("=" * 78)
print("signature = ( E_orig small )  AND  ( Delta_P1 > 0 )")
print("random aggregations, 40 seeds x 3 random scenes x 6 tau, one fixed pi")
rng = np.random.default_rng(2024)
pi = torch.as_tensor(rng.permutation(D))
print("pi =", list(pi.numpy()))
SC = {s: fields(s) for s in range(3)}
rows = []
for s in range(40):
    torch.manual_seed(s)
    agg = Aggregation(in_channels=C, channels=C, num_layers=4).eval()
    with torch.no_grad():
        for sc in range(3):
            eo, dl = [], []
            for t in TAUS:
                m0 = stat(readout(agg(SC[sc][t])))
                m1 = stat(readout(agg(SC[sc][t][:, :, pi])))
                eo.append(abs(m0 - t)); dl.append(abs(m1 - t) - abs(m0 - t))
            rows.append((s, sc, float(np.mean(eo)), float(np.mean(dl))))
R = np.array([(r[2], r[3]) for r in rows])
print()
print("  units = %d   E_orig: min %.3f  p05 %.3f  median %.3f  max %.3f"
      % (len(R), R[:, 0].min(), np.percentile(R[:, 0], 5), np.median(R[:, 0]), R[:, 0].max()))
print("            Delta_P1: min %+.3f  median %+.3f  p95 %+.3f  max %+.3f"
      % (R[:, 1].min(), np.median(R[:, 1]), np.percentile(R[:, 1], 95), R[:, 1].max()))
for e_thr in (0.5, 1.0, 1.5):
    for d_thr in (0.5, 1.0):
        n = int(((R[:, 0] <= e_thr) & (R[:, 1] >= d_thr)).sum())
        print("    joint: E_orig <= %.1f AND Delta >= %.1f  ->  %3d / %d units"
              % (e_thr, d_thr, n, len(R)))
best = rows[int(np.argmin(R[:, 0]))]
print("  best random unit by E_orig: seed %d scene %d  E_orig %.3f  Delta %+.3f"
      % (best[0], best[1], best[2], best[3]))
print()
print("  LIMITATION, STATED: this null uses RANDOM feature fields, because the")
print("  frozen feature extractor lives inside a checkpoint and is out of bounds")
print("  for an audit.  Whether real features change it is UNKNOWN here.")
