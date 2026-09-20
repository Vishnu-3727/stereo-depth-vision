"""AUDIT 2 / measurements R2 and R3.

R2 -- the P2 fallback claim ("the aggregation uses candidate-axis neighbourhood
      structure") : is it a TRAINED property, or already true at random init?
R3 -- the k-axis zero padding : permutation changes which slice sits at the
      padded boundary k in {0, 11}.  How big is that nuisance?
No checkpoint opened.
"""
import sys, itertools
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

torch.manual_seed(100)
LONG = torch.randn(1, C, FH, FW + 6); Lf = LONG[..., :FW]
V = {t: build_cost_volume(Lf, LONG[..., t:t + FW], D, shift="left") for t in TAUS}

print("=" * 76)
print("R2  IS THE P2 FALLBACK CLAIM ALREADY TRUE AT RANDOM INITIALISATION?")
print("=" * 76)
print("claim under test: 'the aggregation is NOT a pointwise function of each")
print("candidate slice' -- detected by Delta_P2 != 0 (P2 = permute, then apply")
print("pi^-1 to the aggregated cost before readout).")
print()
rng = np.random.default_rng(11)
PERMS = [rng.permutation(D) for _ in range(6)]
rows = []
for s in range(12):
    torch.manual_seed(s)
    agg = Aggregation(in_channels=C, channels=C, num_layers=4).eval()
    with torch.no_grad():
        for pi in PERMS:
            pt = torch.as_tensor(pi); inv = torch.argsort(pt)
            for t in TAUS:
                c0 = agg(V[t]); cp = agg(V[t][:, :, pt])[:, inv]
                m0, m2 = stat(readout(c0)), stat(readout(cp))
                # the direct, assumption-free form of the same question:
                dev = float((c0 - cp).abs().max() / c0.abs().max())
                rows.append((abs(m2 - t) - abs(m0 - t), abs(m2 - m0), dev))
R = np.array(rows)
print("random-weight aggregations: %d seeds x %d permutations x %d tau = %d units"
      % (12, len(PERMS), len(TAUS), len(R)))
print("  |Delta_P2|            : mean %.3f  median %.3f  p95 %.3f"
      % (np.abs(R[:, 0]).mean(), np.median(np.abs(R[:, 0])), np.percentile(np.abs(R[:, 0]), 95)))
print("  |m_P2 - m_orig|       : mean %.3f  median %.3f  p95 %.3f"
      % (R[:, 1].mean(), np.median(R[:, 1]), np.percentile(R[:, 1], 95)))
print("  relative cost deviation max|A(Vopi)opi^-1 - A(V)| / max|A(V)|:")
print("        mean %.4f  median %.4f  min %.4f" % (R[:, 2].mean(), np.median(R[:, 2]), R[:, 2].min()))
print("  units with a NON-ZERO cost deviation: %d of %d"
      % (int((R[:, 2] > 1e-6).sum()), len(R)))
print()
print("  -> a RANDOMLY INITIALISED aggregation is already non-pointwise in k, in")
print("     100%% of units.  The P2 claim is a property of the ARCHITECTURE, not")
print("     of training.  It cannot separate a trained model from random weights.")

print()
print("=" * 76)
print("R3  THE k-AXIS ZERO PADDING IS NOT NUISANCE-MATCHED")
print("=" * 76)
print("Conv3d pads k with zeros, so k=0 and k=11 are treated differently from the")
print("interior.  A permutation changes WHICH slice sits there.")
print()
# permutations that send the matched slice to a boundary vs to the interior
def perm_sending(tau, dest):
    p = list(range(D))
    # we need pi^-1(tau) = dest, i.e. pi(dest) = tau
    src = p.index(tau)
    p[dest], p[src] = p[src], p[dest]
    return np.array(p)

torch.manual_seed(0)
agg = Aggregation(in_channels=C, channels=C, num_layers=4).eval()
print("  single transposition moving the matched slice to position `dest`")
print("  (everything else identical).  Random Conv3d aggregation, tau = 3.")
print("  dest :  m_P1     |cost change|rel   (dest 0 and 11 are the PADDED ends)")
t = 3
with torch.no_grad():
    base = agg(V[t]); nb = base.abs().max()
    for dest in range(D):
        pi = torch.as_tensor(perm_sending(t, dest))
        cp = agg(V[t][:, :, pi])
        mark = "  <-- padded boundary" if dest in (0, D - 1) else ""
        print("   %2d  : %7.3f   %14.4f%s"
              % (dest, stat(readout(cp)), float((cp[:, torch.argsort(pi)] - base).abs().max() / nb), mark))
print()
print("  -> the size of the permutation's effect depends on WHERE the matched")
print("     slice lands, including whether it lands against the zero padding.")
print("     'same weights, same input' does not make the two arms exchangeable.")
