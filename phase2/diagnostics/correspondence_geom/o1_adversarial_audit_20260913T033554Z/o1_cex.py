"""O1 AUDIT -- synthetic counterexample suite.

NO CHECKPOINT IS OPENED.  Toy mechanisms + the frozen cost-volume operator +
the frozen readout functions, on random feature fields.  Random-weight
Aggregation is constructed fresh (same class GEOM-002's gate used); it is not
a trained checkpoint.

Question: do materially different mechanisms give DIFFERENT O1/S8 predictions?
"""
import sys, itertools
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "C:/Users/vishn/stereo_depth_vision")
from src.models.stereonet.cost_volume import build_cost_volume          # noqa: E402
from src.models.stereonet.aggregation import Aggregation                # noqa: E402
from phase2.models.scaled_regression import standardise_across_disparity  # noqa: E402

torch.manual_seed(0)
C, FH, FW, D = 32, 17, 71, 12
TAUS = [1, 2, 3, 4, 5, 6]
W_LO, W_HI = 20, 39                      # frozen statistic band (feature cells)

def readout(cost):
    """Frozen readout, k-axis only: z across k, softmax(-z), soft-argmin.

    Upsampling is omitted: it is linear and acts only on (y,w), so it commutes
    exactly with any permutation of k.  DERIVED, not assumed.
    """
    z = standardise_across_disparity(cost, dim=1)
    p = torch.softmax(-z, dim=1)
    k = torch.arange(D, dtype=cost.dtype, device=cost.device).view(1, D, 1, 1)
    return (p * k).sum(dim=1)            # (1, H, W)

def stat(m):
    return float(np.median(m[0, :, W_LO:W_HI + 1].cpu().numpy()))

# ---------------------------------------------------------------- mechanisms
def energy(V):                            # ||V[k]||^2 over channels -> (1,D,H,W)
    return (V ** 2).sum(dim=1)

def M_corr(V):                            # genuine match readout: cost = residual energy
    return energy(V)

def M_lag(V, c0):                         # pointwise matcher reading a SHIFTED slice
    e = energy(V)
    idx = torch.clamp(torch.arange(D) - c0, 0, D - 1)
    return e[:, idx]

def M_pos(V):                             # absolute-index positional bias, content-blind
    k = torch.arange(D, dtype=V.dtype, device=V.device).view(1, D, 1, 1)
    return ((k - 5.5) ** 2).expand(1, D, V.shape[3], V.shape[4]).clone()

def M_adj(V):                             # adjacency sensitivity, NOT correspondence
    nxt = torch.cat([V[:, :, 1:], V[:, :, -1:]], dim=2)
    return ((V - nxt) ** 2).sum(dim=1)

def M_outlier(V):                         # latch the most ATYPICAL slice (no matching)
    e = energy(V)
    return -(e - e.mean(dim=1, keepdim=True)).abs()

def make_randconv(seed):
    torch.manual_seed(seed)
    return Aggregation(in_channels=C, channels=C, num_layers=4).eval()

MECH = [("M_corr   genuine match", M_corr),
        ("M_lag+3  sharp wrong lag", lambda V: M_lag(V, 3)),
        ("M_lag+0  pointwise, centred", lambda V: M_lag(V, 0)),
        ("M_pos    index bias only", M_pos),
        ("M_adj    adjacency, no match", M_adj),
        ("M_outlier atypical-slice latch", M_outlier)]

# ---------------------------------------------------------------- permutations
rng = np.random.default_rng(7)
PERMS = {
    "translate+3 (IN G, forbidden)": np.array([(k - 3) % D for k in range(D)]),
    "reverse      (adjacency kept)": np.array([D - 1 - k for k in range(D)]),
    "random A     (adjacency gone)": rng.permutation(D),
    "random B     (adjacency gone)": rng.permutation(D),
    "fix-taus     (1..6 fixed)": None,
}
p = np.arange(D); tail = np.array([0, 7, 8, 9, 10, 11]); p[tail] = rng.permutation(tail)
PERMS["fix-taus     (1..6 fixed)"] = p

def volumes(seed):
    torch.manual_seed(100 + seed)
    LONG = {w: torch.randn(1, C, FH, FW + max(TAUS)) for w in ("A",)}
    Lf = LONG["A"][..., :FW]
    out = {}
    for tau in TAUS:
        Rf = LONG["A"][..., tau:tau + FW]
        out[tau] = build_cost_volume(Lf, Rf, D, shift="left")
    return out

print("=" * 78)
print("O1 COUNTEREXAMPLE SUITE -- does O1 give DISTINCT predictions per mechanism?")
print("=" * 78)
print("m reported as median over the frozen band w in [20,39]; E = |m - tau|")
print("P1 = permute positions, read out in place  (the proposal as written)")
print("P2 = permute positions, apply pi^-1 to the aggregated cost before readout")
print()

for pname, pi in PERMS.items():
    pi_t = torch.as_tensor(pi, dtype=torch.long)
    inv = torch.argsort(pi_t)
    print("-" * 78)
    print("PERMUTATION: %s   pi = %s" % (pname, list(pi)))
    print("%-30s %7s %7s %7s %8s %8s" % ("mechanism", "E_orig", "E_P1", "E_P2",
                                          "D_P1", "D_P2"))
    V = volumes(0)
    for mname, f in MECH:
        eo, e1, e2 = [], [], []
        for tau in TAUS:
            c = f(V[tau]); mo = stat(readout(c))
            cp = f(V[tau][:, :, pi_t]); m1 = stat(readout(cp))
            m2 = stat(readout(cp[:, inv]))
            eo.append(abs(mo - tau)); e1.append(abs(m1 - tau)); e2.append(abs(m2 - tau))
        eo, e1, e2 = np.mean(eo), np.mean(e1), np.mean(e2)
        print("%-30s %7.3f %7.3f %7.3f %+8.3f %+8.3f"
              % (mname, eo, e1, e2, e1 - eo, e2 - eo))
    # random Conv3d aggregation, 4 seeds
    eo, e1, e2 = [], [], []
    for s in range(4):
        agg = make_randconv(s)
        with torch.no_grad():
            for tau in TAUS:
                c = agg(V[tau]); mo = stat(readout(c))
                cp = agg(V[tau][:, :, pi_t]); m1 = stat(readout(cp))
                m2 = stat(readout(cp[:, inv]))
                eo.append(abs(mo - tau)); e1.append(abs(m1 - tau)); e2.append(abs(m2 - tau))
    print("%-30s %7.3f %7.3f %7.3f %+8.3f %+8.3f"
          % ("M_randconv frozen-arch random", np.mean(eo), np.mean(e1), np.mean(e2),
             np.mean(e1) - np.mean(eo), np.mean(e2) - np.mean(eo)))
