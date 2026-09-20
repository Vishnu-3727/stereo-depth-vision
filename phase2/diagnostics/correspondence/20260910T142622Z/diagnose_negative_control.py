import os, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
REPO = Path(r"C:\Users\vishn\stereo_depth_vision")
sys.path.insert(0, str(REPO))
import numpy as np, torch
from src.models.stereonet.stereonet import StereoNet, StereoNetConfig
from src.datasets.kitti2015 import normalize
from phase2.models import scaled_regression
from phase2.viz import core

ck = REPO/"phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth"
m = StereoNet(StereoNetConfig(cost_volume_shift="none"))
scaled_regression.apply_to(m)
m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
m.eval().to(dev)

sc = core.load_scene(27, split="hailo_val")
m.regression.capture = True
with torch.no_grad():
    _, st = m(torch.from_numpy(normalize(sc.left)).to(dev),
              torch.from_numpy(normalize(sc.right)).to(dev), return_stages=True)
soft = m.regression.last["softmax_input"]

vol = st["cost_volume"]        # (B,C,D,H,W)
agg = st["aggregated_cost"]    # (B,D,h,w)
print("=== 1. is the cost volume degenerate (EXP-010)? ===")
d0 = vol[:, :, 0]
mx = max(float((vol[:, :, k] - d0).abs().max()) for k in range(vol.shape[2]))
print("   max |slice_k - slice_0| over all 12 slices =", mx)

print()
print("=== 2. aggregated cost across the disparity axis (mean over pixels) ===")
per_d = agg[0].mean(dim=(1, 2)).double().cpu().numpy()
for k, v in enumerate(per_d):
    print("   d=%2d  %+.6f" % (k, v))
interior = per_d[1:11]
print("   interior d=1..10 spread = %.3e   endpoints d0=%.6f d11=%.6f"
      % (interior.max()-interior.min(), per_d[0], per_d[11]))

print()
print("=== 3. per-pixel across-d structure of aggregated cost ===")
a = agg[0].double()
std_d = a.std(dim=0, unbiased=False)
print("   mean over pixels of std_d(aggregated) = %.6e" % float(std_d.mean()))
inner = a[1:11]
std_inner = inner.std(dim=0, unbiased=False)
print("   same, restricted to d=1..10          = %.6e" % float(std_inner.mean()))

print()
print("=== 4. the tensor the softmax consumes (standardised, full-res) ===")
s = soft[0].double()
sd = s.mean(dim=(1, 2)).cpu().numpy()
for k, v in enumerate(sd):
    print("   d=%2d  standardised mean %+.6f" % (k, v))
w = torch.softmax(-s, dim=0)
idx = torch.arange(12, dtype=torch.float64, device=s.device).view(12, 1, 1)
di = (w*idx).sum(dim=0)
print("   soft-argmin mean %.6f  (uniform anchor would be 5.5)" % float(di.mean()))
print("   soft-argmin min %.4f max %.4f std %.4f" % (float(di.min()), float(di.max()), float(di.std())))
print("   mean max-softmax-weight %.6f  (uniform would be %.6f)" % (float(w.max(dim=0).values.mean()), 1/12))
