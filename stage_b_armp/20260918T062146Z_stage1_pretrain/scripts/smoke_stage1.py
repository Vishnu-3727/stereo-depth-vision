"""Smoke check: FT3D triplet -> scale-crop -> P2A forward/backward. No training."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch

from ft3d import FlyingThings3DTrain
from src.models.stereonet import StereoNet, StereoNetConfig
from src.losses.disparity import masked_smooth_l1
from train_armp_stage1 import ScaledCroppedFT3D, integrity_guard, P2A_CONFIG, P2A_PARAMS

t0 = time.time()
base = FlyingThings3DTrain(HERE.parent / "dataset_manifest.json", split="train")
print("train triplets:", len(base), flush=True)
smp = base[0]
print("sample:", smp.name, smp.left.shape, smp.disparity.shape, smp.disparity.dtype,
      "disp_min=%.2f disp_max=%.2f frac_pos=%.4f" % (
          float(smp.disparity.min()), float(smp.disparity.max()),
          float((smp.disparity > 0).mean())), flush=True)
ds = ScaledCroppedFT3D(base, seed=0)
t1 = time.time()
left, right, disp = ds[0]
t2 = time.time()
print("crop:", tuple(left.shape), tuple(disp.shape), "load+crop %.1fs" % (t2 - t1), flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StereoNet(StereoNetConfig(**P2A_CONFIG)).to(device)
guard = integrity_guard(model, StereoNetConfig(**P2A_CONFIG))
print("guard:", guard["param_count"], guard["all_ok"], flush=True)
assert guard["all_ok"] and guard["param_count"] == P2A_PARAMS
model.train()
out = model(left[None].to(device), right[None].to(device))
loss, n = masked_smooth_l1(out, disp[None].to(device), max_disparity=184.0)
loss.backward()
print("device=%s fwd+bwd loss=%.4f valid=%d total_smoke=%.1fs" % (device, float(loss), n, time.time() - t0), flush=True)
