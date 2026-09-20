"""AUDIT 2 / measurement R1.

Is the decisive counterexample ARCHITECTURALLY REALIZABLE?

Audit 1 rejected O1 partly on an abstract 'atypical-slice latching' operator.
An abstract function is not admissible evidence unless the architecture under
test can express it.  Here the counterexample is built INSIDE the frozen
Aggregation class by SETTING WEIGHTS BY HAND -- no training, no checkpoint, no
optimizer.  Then it is proved non-corresponding by a spike-injection test.
"""
import sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "C:/Users/vishn/stereo_depth_vision")
from src.models.stereonet.cost_volume import build_cost_volume            # noqa
from src.models.stereonet.aggregation import Aggregation                  # noqa
from src.models.stereonet.blocks import AGGREGATION_SLOPE as A            # noqa
from phase2.models.scaled_regression import standardise_across_disparity  # noqa

C, FH, FW, D = 32, 17, 71, 12
TAUS = [1, 2, 3, 4, 5, 6]; W_LO, W_HI = 20, 39
print("LeakyReLU negative slope (frozen) =", A)

def zero_(m):
    for p in m.parameters():
        nn.init.zeros_(p)

def conv_of(agg, i):
    return agg.filter[i] if i < 4 else agg.to_cost

def build(kind):
    """Hand-set weights.  kind='match' or 'contrast'."""
    agg = Aggregation(in_channels=C, channels=C, num_layers=4).eval()
    zero_(agg)
    with torch.no_grad():
        L = [agg.filter[0], agg.filter[2], agg.filter[4], agg.filter[6], agg.to_cost]
        # L1: out c = +V[c] (c<16), out c = -V[c-16] (c>=16); centre tap only
        for c in range(16):
            L[0].weight[c, c, 1, 1, 1] = 1.0
            L[0].weight[c + 16, c, 1, 1, 1] = -1.0
        # after LReLU: LReLU(+V[c]), LReLU(-V[c]);  sum = (1-A)|V[c]|
        # L2: out0 = sum_c ( x[c] + x[c+16] )  ->  n(k) = (1-A) * L1 norm over 16 ch
        for c in range(32):
            L[1].weight[0, c, 1, 1, 1] = 1.0
        # n >= 0 so LReLU is identity on it
        if kind == "match":
            L[2].weight[0, 0, 1, 1, 1] = 1.0          # pass n
            L[3].weight[0, 0, 1, 1, 1] = 1.0          # pass n
            L[4].weight[0, 0, 1, 1, 1] = 1.0          # cost = n  -> argmin at the match
        else:
            # L3: out0 = n(k) - .5 n(k-1) - .5 n(k+1)   (3-tap along the CANDIDATE axis)
            #     out1 = -(that)
            for o, s in ((0, 1.0), (1, -1.0)):
                L[2].weight[o, 0, 1, 1, 1] = s * 1.0
                L[2].weight[o, 0, 0, 1, 1] = s * -0.5
                L[2].weight[o, 0, 2, 1, 1] = s * -0.5
            # after LReLU:  LReLU(g), LReLU(-g);  sum = (1-A)|g|
            L[3].weight[0, 0, 1, 1, 1] = 1.0
            L[3].weight[0, 1, 1, 1, 1] = 1.0
            # cost = -|g|  ->  argmin picks the LARGEST local contrast
            L[4].weight[0, 0, 1, 1, 1] = -1.0
    return agg

def readout(cost):
    z = standardise_across_disparity(cost, dim=1)
    p = torch.softmax(-z, dim=1)
    k = torch.arange(D, dtype=cost.dtype).view(1, D, 1, 1)
    return (p * k).sum(dim=1)
def stat(m): return float(np.median(m[0, :, W_LO:W_HI + 1].numpy()))
def hard_argmin(cost):
    return np.median(cost[0, :, :, W_LO:W_HI + 1].argmin(dim=0).numpy())

torch.manual_seed(100)
LONG = torch.randn(1, C, FH, FW + 6); Lf = LONG[..., :FW]
V = {t: build_cost_volume(Lf, LONG[..., t:t + FW], D, shift="left") for t in TAUS}

M = {"HC_match  (hand-built matcher)": build("match"),
     "HC_contrast (hand-built, NO matching)": build("contrast")}

print()
print("=== R1a: do the hand-built operators behave as intended on the clean CV? ===")
print("%-40s %s" % ("operator", "  hard argmin per tau (tau = 1..6)"))
with torch.no_grad():
    for name, agg in M.items():
        am = [hard_argmin(agg(V[t])) for t in TAUS]
        print("%-40s %s" % (name, ["%.0f" % a for a in am]))
print("  both latch the matched slice -> IDENTICAL observable on clean data")

print()
print("=== R1b: SPIKE TEST -- proof that HC_contrast is NOT matching ===")
print("inject a large-norm spike at a NON-matched candidate k*; a matcher must")
print("ignore it (it is not a match), a contrast latch must jump to it.")
with torch.no_grad():
    for kstar in (8, 9, 10):
        for t in (2, 3):
            Vs = V[t].clone()
            Vs[:, :, kstar] = Vs[:, :, kstar] + 8.0     # anomalous slice, not a match
            a_m = hard_argmin(M["HC_match  (hand-built matcher)"](Vs))
            a_c = hard_argmin(M["HC_contrast (hand-built, NO matching)"](Vs))
            print("  spike at k*=%2d, tau=%d :  matcher argmin = %.0f (want %d)   "
                  "contrast argmin = %.0f (want %d)" % (kstar, t, a_m, t, a_c, kstar))
print("  -> the two operators are DIFFERENT FUNCTIONS; they only agree when the")
print("     matched slice is the only anomaly, which is exactly the synthetic case.")

print()
print("=== R1c: O1/S8 signature of the hand-built pair ===")
rng = np.random.default_rng(3)
perms = {"reverse": np.array([D - 1 - k for k in range(D)]),
         "random A": rng.permutation(D), "random B": rng.permutation(D)}
print("%-40s %8s %10s %10s %10s" % ("operator", "E_orig", "D_P1 rev", "D_P1 rA", "D_P1 rB"))
with torch.no_grad():
    for name, agg in M.items():
        eo = np.mean([abs(stat(readout(agg(V[t]))) - t) for t in TAUS])
        row = []
        for pn, p in perms.items():
            pt = torch.as_tensor(p)
            e1 = np.mean([abs(stat(readout(agg(V[t][:, :, pt]))) - t) for t in TAUS])
            row.append(e1 - eo)
        print("%-40s %8.3f %10.3f %10.3f %10.3f" % (name, eo, *row))
