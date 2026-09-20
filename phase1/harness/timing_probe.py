"""Timing probe: time N training steps only, discard weights, seconds-scale."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from src.datasets.kitti2015 import Kitti2015Stereo
from src.models.stereonet import StereoNet, StereoNetConfig
from src.losses.disparity import masked_smooth_l1
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.datasets.kitti2015 import normalize

CROP_H, CROP_W = 256, 512

class CroppedKitti(Dataset):
    def __init__(self, base, seed=0):
        self.base = base
        self.rng = np.random.default_rng(seed)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        s = self.base[i]
        h, w = s.disparity.shape
        y = int(self.rng.integers(0, h - CROP_H + 1))
        x = int(self.rng.integers(0, w - CROP_W + 1))
        sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
        left = normalize(s.left[sl])[0]
        right = normalize(s.right[sl])[0]
        disp = s.disparity[sl][None]
        return (torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(disp, dtype=np.float32)))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    base = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
    ds = CroppedKitti(base, seed=0)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0)
    model = StereoNet(StereoNetConfig()).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    it = iter(loader)
    # warmup 1 step (not timed)
    left, right, disp = next(it)
    left, right, disp = left.to(device), right.to(device), disp.to(device)
    out = model(left, right)
    loss, n = masked_smooth_l1(out, disp, max_disparity=float(StereoNetConfig().max_disparity_px))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # timed steps
    times = []
    for k in range(a.steps):
        left, right, disp = next(it)
        left, right, disp = left.to(device), right.to(device), disp.to(device)
        t0 = time.perf_counter()
        out = model(left, right)
        loss, n = masked_smooth_l1(out, disp, max_disparity=float(StereoNetConfig().max_disparity_px))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    import numpy as np
    print(f"steps={a.steps} batch={a.batch} times_s={[round(t,3) for t in times]}")
    print(f"mean_per_step_s={float(np.mean(times)):.4f} median={float(np.median(times)):.4f}")
    steps_per_epoch = len(loader)
    print(f"steps_per_epoch_batch{a.batch}={steps_per_epoch}")
    print(f"extrap_epoch_s={float(np.mean(times))*steps_per_epoch:.1f}")
    for epochs in [20, 200, 300]:
        print(f"extrap_{epochs}ep_s={float(np.mean(times))*steps_per_epoch*epochs:.1f} h={float(np.mean(times))*steps_per_epoch*epochs/3600:.2f}")
    print("WEIGHTS DISCARDED: probe weights were never saved (by design).")

if __name__ == "__main__":
    main()
