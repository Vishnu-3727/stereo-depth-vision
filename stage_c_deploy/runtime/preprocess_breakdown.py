"""Q2 probe: break down bench load_preprocess with perf_counter (MEASUREMENT ONLY).

Mirrors bench_host.run_pipeline's load_preprocess region primitive-for-primitive:
  left PNG decode | right PNG decode | disparity GT read | pad+crop |
  normalize L+R | tensor construct + .to(device) | calib parse + fy read
Reports median ms over 10 hailo_val scenes (2 warmup iters, like the bench).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "spatial_perception"))
from src.datasets.kitti2015 import (Kitti2015Stereo, normalize, pad_and_crop,
                                    read_disparity_png, read_image)
from src.geometry.stereo import parse_kitti_cam_to_cam
from metric_depth import read_fy_from_calib
import torch

STEPS = ("left_png_decode", "right_png_decode", "disp_gt_read", "pad_crop",
         "normalize_LR", "tensor_to_device", "calib_parse_fy")

def run_once(ds, dev: str, idx: int):
    t = {}
    name = ds.names[idx]
    root = ds.root
    s = time.perf_counter()
    left = read_image(root / "training" / "image_2" / name)
    t["left_png_decode"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter()
    right = read_image(root / "training" / "image_3" / name)
    t["right_png_decode"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter()
    disp = read_disparity_png(ds.gt_dir / name, scale=ds.disparity_scale)
    t["disp_gt_read"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter()
    left = pad_and_crop(left); right = pad_and_crop(right); disp = pad_and_crop(disp)
    t["pad_crop"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter()
    tl_np = normalize(left); tr_np = normalize(right)
    t["normalize_LR"] = (time.perf_counter() - s) * 1e3
    if dev == "cuda":
        torch.cuda.synchronize()
    s = time.perf_counter()
    tl = torch.from_numpy(tl_np).to(dev); tr = torch.from_numpy(tr_np).to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t["tensor_to_device"] = (time.perf_counter() - s) * 1e3
    s = time.perf_counter()
    cal_path = ds.calibration_path(name)
    calib = parse_kitti_cam_to_cam(cal_path); fy = read_fy_from_calib(cal_path)
    t["calib_parse_fy"] = (time.perf_counter() - s) * 1e3
    assert tl.shape == (1, 3, 368, 1232) and fy > 0
    return t

def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    n = 10
    for _ in range(2):
        run_once(ds, dev, 0)
    acc = {k: [] for k in STEPS}
    for i in range(n):
        t = run_once(ds, dev, i)
        for k in STEPS:
            acc[k].append(t[k])
    print(f"load_preprocess breakdown: device={dev} scenes={n} torch={torch.__version__}")
    tot_med = 0.0
    for k in STEPS:
        v = np.asarray(acc[k])
        m = float(np.median(v))
        tot_med += m
        print(f"{k:18s} median={m:8.2f} ms  min={float(v.min()):8.2f}  max={float(v.max()):8.2f}")
    print(f"{'sum_of_medians':18s} median={tot_med:8.2f} ms")

if __name__ == "__main__":
    main()
