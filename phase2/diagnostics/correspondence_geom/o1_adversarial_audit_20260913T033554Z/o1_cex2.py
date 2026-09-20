"""O1 AUDIT part 2: (a) readout equivariance, (b) per-tau structure of Delta,
(c) the point prediction m_pi == pi^-1(tau), (d) a translation INSIDE G.
No checkpoint opened."""
import sys
import numpy as np
import torch
sys.path.insert(0, "C:/Users/vishn/stereo_depth_vision")
from src.models.stereonet.cost_volume import build_cost_volume            # noqa
from src.models.stereonet.aggregation import Aggregation                  # noqa
from phase2.models.scaled_regression import standardise_across_disparity  # noqa

C, FH, FW, D = 32, 17, 71, 12
TAUS = [1, 2, 3, 4, 5, 6]; W_LO, W_HI = 20, 39

def readout(cost):
    z = standardise_across_disparity(cost, dim=1)
    p = torch.softmax(-z, dim=1)
    k = torch.arange(D, dtype=cost.dtype).view(1, D, 1, 1)
    return (p * k).sum(dim=1)

def stat(m): return float(np.median(m[0, :, W_LO:W_HI + 1].numpy()))
def energy(V): return (V ** 2).sum(dim=1)

torch.manual_seed(100)
LONG = torch.randn(1, C, FH, FW + 6); Lf = LONG[..., :FW]
V = {t: build_cost_volume(Lf, LONG[..., t:t + FW], D, shift="left") for t in TAUS}

print("=== (a) IS THE READOUT PERMUTATION-EQUIVARIANT?  (A7) ===")
rng = np.random.default_rng(3); pi = torch.as_tensor(rng.permutation(D))
c = energy(V[3])
lhs = standardise_across_disparity(c[:, pi], dim=1)
rhs = standardise_across_disparity(c, dim=1)[:, pi]
print("  max |z(c o pi) - z(c) o pi|           = %r" % float((lhs - rhs).abs().max()))
l2 = torch.softmax(-lhs, 1); r2 = torch.softmax(-rhs, 1)
print("  max |softmax(-z(c o pi)) - (...) o pi| = %r" % float((l2 - r2).abs().max()))
print("  -> the z-score and softmax are EXACTLY equivariant; the readout adds no")
print("     artifact of its own.  It converts pi into a relabelling of the")
print("     soft-argmin weights:  m(c o pi) = E_{j~p}[pi^-1(j)].")

print()
print("=== (b) A TRANSLATION IS *INSIDE* G AND STILL GIVES A LARGE POSITIVE Delta ===")
for s in (1, 2, 3):
    p_t = torch.as_tensor(np.array([(k - s) % D for k in range(D)]))
    d = [abs(stat(readout(energy(V[t])[:, p_t])) - t) - abs(stat(readout(energy(V[t]))) - t)
         for t in TAUS]
    print("  cyclic translation by %+d (IN G):  mean Delta_P1 = %+.3f" % (s, np.mean(d)))
print("  -> Delta_P1 > 0 does NOT certify that the transformation left G.")

print()
print("=== (c) PER-TAU STRUCTURE OF Delta  (readout-centre artifact) ===")
pi = torch.as_tensor(rng.permutation(D)); inv = torch.argsort(pi)
print("  tau :  m_orig   m_perm   E_orig   E_perm    Delta   pi^-1(tau)")
for t in TAUS:
    mo = stat(readout(energy(V[t]))); mp = stat(readout(energy(V[t])[:, pi]))
    print("   %d  : %7.3f %8.3f %8.3f %8.3f %+8.3f %10d"
          % (t, mo, mp, abs(mo - t), abs(mp - t), abs(mp - t) - abs(mo - t),
             int(inv[t])))
print("  -> E_perm tracks |m_perm - tau| with m_perm pulled toward the window")
print("     centre 5.5, so Delta is systematically SMALLER at tau ~ 5-6.")
print("     The effect size is set by the tau grid and the readout centre,")
print("     not by the mechanism.")

print()
print("=== (d) DOES THE POINT PREDICTION m_pi = pi^-1(tau) SURVIVE THE SOFT READOUT? ===")
for name, P in [("random", torch.as_tensor(rng.permutation(D))),
                ("reverse", torch.as_tensor(np.array([D - 1 - k for k in range(D)])))]:
    iv = torch.argsort(P)
    pred = [int(iv[t]) for t in TAUS]
    obs = [stat(readout(energy(V[t])[:, P])) for t in TAUS]
    err = [abs(o - p) for o, p in zip(obs, pred)]
    print("  %-8s predicted pi^-1(tau) = %s" % (name, pred))
    print("           observed  m_pi     = %s" % ["%.2f" % o for o in obs])
    print("           mean |obs - pred|  = %.3f   (a HARD matcher would give 0)" % np.mean(err))

print()
print("=== (e) RANDOM-CONV NULL FOR Delta_P1: is it a control or a flatness artifact? ===")
pi = torch.as_tensor(rng.permutation(D))
sharp, flat = [], []
for s in range(8):
    torch.manual_seed(s)
    agg = Aggregation(in_channels=C, channels=C, num_layers=4).eval()
    with torch.no_grad():
        for t in TAUS:
            c0 = agg(V[t]); cp = agg(V[t][:, :, pi])
            z = standardise_across_disparity(c0, dim=1)
            pmax = float(torch.softmax(-z, 1).max(dim=1).values.median())
            mo, mp = stat(readout(c0)), stat(readout(cp))
            sharp.append(pmax); flat.append(abs(mp - t) - abs(mo - t))
print("  median max softmax prob (sharpness) : %.4f   (uniform would be %.4f)"
      % (np.median(sharp), 1 / D))
print("  mean Delta_P1 over 8 random aggregations x 6 tau : %+.4f" % np.mean(flat))
print("  -> the random-weight null for Delta_P1 sits at ~0 because the softmax is")
print("     nearly flat and m is pinned near the window centre in BOTH arms.")
print("     It is a FLATNESS artifact, not a demonstration of geometric blindness.")
