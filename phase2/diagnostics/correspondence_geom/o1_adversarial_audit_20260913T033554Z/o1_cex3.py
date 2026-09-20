"""O1 AUDIT part 3: is the point-prediction test capable IN PRINCIPLE, and how
does sharpness drive both Delta and the point prediction?  No checkpoint."""
import sys
import numpy as np, torch
sys.path.insert(0, "C:/Users/vishn/stereo_depth_vision")
from src.models.stereonet.cost_volume import build_cost_volume            # noqa
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
rng = np.random.default_rng(3)
pi = torch.as_tensor(rng.permutation(D)); inv = torch.argsort(pi)

print("sharpness sweep: cost scaled by g before the readout (g=1 is the frozen scale)")
print("  g    E_orig   Delta_P1   mean|m_pi - pi^-1(tau)|   maxprob")
for g in (0.25, 1.0, 4.0, 16.0, 64.0):
    eo, dl, pp, mp_ = [], [], [], []
    for t in TAUS:
        c = energy(V[t]) * g
        mo = stat(readout(c)); m1 = stat(readout(c[:, pi]))
        eo.append(abs(mo - t)); dl.append(abs(m1 - t) - abs(mo - t))
        pp.append(abs(m1 - int(inv[t])))
        z = standardise_across_disparity(c, dim=1)
        mp_.append(float(torch.softmax(-z, 1).max(dim=1).values.median()))
    print("  %5.2f %7.3f %+10.3f %22.3f %9.4f"
          % (g, np.mean(eo), np.mean(dl), np.mean(pp), np.mean(mp_)))
print()
print("note: the frozen readout has NO temperature knob -- g is not a free")
print("parameter of the experiment.  The sweep only shows that BOTH the size of")
print("Delta and the fidelity of the point prediction are governed by how peaked")
print("the model's standardised profile happens to be, which is a property of")
print("the trained weights and is UNKNOWN without loading them.")
