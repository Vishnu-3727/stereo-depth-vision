"""FINAL BLIND O1 IDENTIFIABILITY AUDIT -- numerical section.

AUDIT ONLY.  No checkpoint is opened, no training, no O1 execution.
Random synthetic features + hand-set / random weights inside the FROZEN
Aggregation class, to test architectural realizability and the permutation
algebra.  Frozen modules imported, never modified.
"""
import sys, itertools, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn

REPO = Path("C:/Users/vishn/stereo_depth_vision")
sys.path.insert(0, str(REPO))
from src.models.stereonet.cost_volume import build_cost_volume, shift_left
from src.models.stereonet.aggregation import Aggregation
from phase2.models.scaled_regression import StandardisedDisparityRegression

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

D, FH, FW, C = 12, 17, 71, 32
CROP_H, CROP_W = 272, 1136
MY0, MY1, MX0, MX1 = 64, 208, 325, 633
TAUS = [1, 2, 3, 4, 5, 6]
SLOPE = 0.01
out = {}

# ---------------------------------------------------------------- T0  |x| identity
x = torch.randn(10000, dtype=torch.float64)
lr = nn.LeakyReLU(SLOPE)
resid = (lr(x) + lr(-x) - (1 - SLOPE) * x.abs()).abs().max().item()
out["T0_abs_identity_max_resid"] = resid

# ---------------------------------------------------------------- frozen readout
reg = StandardisedDisparityRegression(upsample_first=True).eval()

def readout(cost):
    with torch.no_grad():
        m = reg(cost, (CROP_H, CROP_W))
    return float(m[0, 0, MY0:MY1, MX0:MX1].double().median())

# ---------------------------------------------------------------- latch weights
def latch_aggregation():
    """Sum_{c<16} |V_c|, POINTWISE in (k,y,x): centre taps only, zero spatial and
    zero candidate-axis reach.  Never compares left and right (the L-R subtraction
    is upstream, in the frozen cost-volume op); never uses candidate adjacency;
    never uses the candidate index."""
    agg = Aggregation().eval()
    for p in agg.parameters():
        torch.nn.init.zeros_(p)
    cv = [m for m in agg.filter if isinstance(m, nn.Conv3d)]
    with torch.no_grad():
        for j in range(16):                       # layer 1: +V_j and -V_j
            cv[0].weight[j, j, 1, 1, 1] = 1.0
            cv[0].weight[16 + j, j, 1, 1, 1] = -1.0
        for j in range(16):                       # layer 2: sum -> 0.99*sum|V_j|
            cv[1].weight[0, j, 1, 1, 1] = 1.0
            cv[1].weight[0, 16 + j, 1, 1, 1] = 1.0
        cv[2].weight[0, 0, 1, 1, 1] = 1.0         # layers 3,4: identity on ch 0
        cv[3].weight[0, 0, 1, 1, 1] = 1.0
        agg.to_cost.weight[0, 0, 1, 1, 1] = 1.0
    return agg

# ---------------------------------------------------------------- data
Lf = torch.randn(1, C, FH, FW)
vols = {}
for tau in TAUS:
    Rf = shift_left(Lf, tau)
    V = build_cost_volume(Lf, Rf, D, "subtract", "left")
    a = V[:, :, tau, :, 15:45].abs().max().item()
    vols[tau] = (V, a)
out["T3_anchor_max_abs_V_at_tau_over_taus"] = max(v[1] for v in vols.values())
# how distinguishable is the matched slice?  L1 profile of the raw volume
prof = {tau: [float(vols[tau][0][:, :, k, :, 20:40].abs().mean()) for k in range(D)]
        for tau in TAUS}
out["T3_raw_L1_profile_tau3"] = prof[3]

# ---------------------------------------------------------------- permutations
g = torch.Generator().manual_seed(20260913)
PIS = [torch.randperm(D, generator=g) for _ in range(8)]

def perm_apply(V, pi):
    return V.index_select(2, pi).contiguous()

def run(agg, tag):
    rows = []
    for tau in TAUS:
        V, _ = vols[tau]
        with torch.no_grad():
            m0 = readout(agg(perm_apply(V, torch.arange(D))))
        for i, pi in enumerate(PIS):
            pinv = torch.argsort(pi)
            with torch.no_grad():
                Cp = agg(perm_apply(V, pi))
                m1 = readout(Cp)                                   # P1
                m2 = readout(Cp.index_select(1, pinv).contiguous())  # P2
            rows.append(dict(tag=tag, tau=tau, pi=i, m0=m0, m1=m1, m2=m2,
                             E0=abs(m0 - tau),
                             S8_P1=abs(m1 - tau) - abs(m0 - tau),
                             S8_P2=abs(m2 - tau) - abs(m0 - tau),
                             pinv_tau=int(pinv[tau]),
                             pred_latch=abs(int(pinv[tau]) - tau),
                             pred_collapse=abs(5.5 - tau)))
    return rows

latch = latch_aggregation()
rows = run(latch, "latch")

# random-weight aggregations (default init) -- full 3-tap kernels
rand_rows = []
for s in range(3):
    torch.manual_seed(1000 + s)
    ra = Aggregation().eval()
    rand_rows += run(ra, "rand%d" % s)

def summ(rs, key):
    v = np.array([r[key] for r in rs], dtype=np.float64)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))

out["T5_latch"] = dict(
    E0=summ(rows, "E0"), S8_P1=summ(rows, "S8_P1"), S8_P2=summ(rows, "S8_P2"),
    m1_vs_pinv_tau_maxabs=float(max(abs(r["m1"] - r["pinv_tau"]) for r in rows)),
    frac_S8_P1_ge_1=float(np.mean([r["S8_P1"] >= 1 for r in rows])),
    per_tau_E0={r["tau"]: r["E0"] for r in rows})
out["T7_random"] = dict(
    E0=summ(rand_rows, "E0"), S8_P1=summ(rand_rows, "S8_P1"),
    S8_P2=summ(rand_rows, "S8_P2"),
    m0_mean=float(np.mean([r["m0"] for r in rand_rows])),
    frac_S8_P1_ge_1=float(np.mean([r["S8_P1"] >= 1 for r in rand_rows])),
    frac_S8_P2_nonzero=float(np.mean([abs(r["S8_P2"]) > 1e-9 for r in rand_rows])))

# ---------------------------------------------------------------- T11/T13 collapse arithmetic
out["T13_collapse_S8_if_m1_eq_5p5"] = {t: abs(5.5 - t) for t in TAUS}
out["T13_mean_collapse_S8"] = float(np.mean([abs(5.5 - t) for t in TAUS]))
out["T13_mean_latch_S8_pred"] = float(np.mean([r["pred_latch"] for r in rows]))

# ---------------------------------------------------------------- T9/T10 padding reach
# which candidate indices are padding-free after 5 padded 3-taps?
reach = 5
out["T9_padding_free_k"] = [k for k in range(D) if k - reach >= 0 and k + reach <= D - 1]

print(json.dumps(out, indent=1, default=str))
