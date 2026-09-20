"""Host benchmark for the Stage C deployment pipeline (MEASUREMENT ONLY).

Times the exact stage composition of stage_c_deploy/demo/pipeline_demo.py:

    stereo pair -> ARM-P disparity -> metric depth (m) -> spatial cells,
    occupancy, depth discontinuities

Every stage calls the same frozen functions the demo calls; this file only
adds perf_counter regions around them. No optimization: no torch.compile, no
channels_last, no batching, no resolution change. fp16 applies to the model
only; the geometry stages stay fp64/fp32 as they are.

Usage:
    python stage_c_deploy/runtime/bench_host.py --device cpu --scenes 10
    python stage_c_deploy/runtime/bench_host.py --device cuda --scenes 10
    python stage_c_deploy/runtime/bench_host.py --device cuda --dtype fp16 --scenes 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "spatial_perception"))

from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.geometry.stereo import parse_kitti_cam_to_cam  # noqa: E402

import armp_depth as AD  # noqa: E402
from metric_depth import (disparity_to_depth,  # noqa: E402
                          read_fy_from_calib, reproject_xyz)
from discontinuity import depth_discontinuity  # noqa: E402
from occupancy import occupancy_grid  # noqa: E402
from pointcloud import depth_to_pointcloud, physical_mask  # noqa: E402
from spatial import spatial_cells  # noqa: E402

import torch  # noqa: E402

# Same constants the demo uses (visualization subsample only; depth never
# subsampled; stored depth never modified).
PHYSICAL_MAX_M = 60.0
NEAR_THRESHOLD_M = 10.0
CLOUD_STRIDE = 3

STAGES = ("load_preprocess", "inference", "metric_depth",
          "cloud_spatial_occupancy", "discontinuity")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="cpu | cuda (default: cuda if available else cpu)")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"),
                    help="fp32 (default) | fp16 (model only) | bf16 (model only)")
    ap.add_argument("--scenes", type=int, default=10,
                    help="number of hailo_val scenes from the head of the list")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed full-pipeline warmup iterations on scene 0")
    ap.add_argument("--out", default=None, help="JSON output path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda"):
        print(f"unknown --device {device!r}: use cpu or cuda", file=sys.stderr)
        sys.exit(2)
    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device available for --device cuda", file=sys.stderr)
        sys.exit(2)

    kitti_root = REPO / "data" / "kitti2015"
    if not (kitti_root / "training" / "image_2").exists():
        print(f"KITTI data missing: {kitti_root / 'training' / 'image_2'} not found",
              file=sys.stderr)
        sys.exit(2)
    ckpt = REPO / AD.ARMP_REL
    if not ckpt.exists():
        print(f"checkpoint missing: {ckpt} not found", file=sys.stderr)
        sys.exit(2)

    try:
        ds = Kitti2015Stereo(kitti_root, split="hailo_val")
    except FileNotFoundError as e:
        print(f"KITTI data missing: {e}", file=sys.stderr)
        sys.exit(2)
    n = min(args.scenes, len(ds))
    if args.scenes > len(ds):
        print(f"requested {args.scenes} scenes, only {len(ds)} in hailo_val; using {n}")
    names = list(ds.names[:n])

    # Asserts the checkpoint SHA-256 exactly as the demo does (inside).
    try:
        net, dev = AD.load_frozen_net(device)
    except FileNotFoundError as e:
        print(f"checkpoint missing: {e}", file=sys.stderr)
        sys.exit(2)
    use_half = args.dtype == "fp16"
    use_bf16 = args.dtype == "bf16"
    if use_half:
        net = net.half()
    elif use_bf16:
        net = net.to(torch.bfloat16)
    net.eval()

    is_cuda = dev == "cuda"

    def run_pipeline(idx: int) -> tuple[dict, np.ndarray]:
        """Run one scene, timing the five stages. Returns (times_ms, disparity)."""
        t = {}
        if is_cuda:
            torch.cuda.synchronize()
        s = time.perf_counter()
        sample = ds[idx]
        tl_np = normalize(sample.left)
        tr_np = normalize(sample.right)
        tl = torch.from_numpy(tl_np).to(dev)
        tr = torch.from_numpy(tr_np).to(dev)
        if use_half:
            tl = tl.half()
            tr = tr.half()
        elif use_bf16:
            tl = tl.to(torch.bfloat16)
            tr = tr.to(torch.bfloat16)
        cal_path = ds.calibration_path(sample.name)
        calib = parse_kitti_cam_to_cam(cal_path)
        fy = read_fy_from_calib(cal_path)
        if is_cuda:
            torch.cuda.synchronize()
        t["load_preprocess"] = (time.perf_counter() - s) * 1e3

        if is_cuda:
            torch.cuda.synchronize()
        s = time.perf_counter()
        with torch.no_grad():
            out = net(tl, tr)
        disparity = out[0, 0].detach().float().cpu().numpy().astype(np.float64)
        if is_cuda:
            torch.cuda.synchronize()
        t["inference"] = (time.perf_counter() - s) * 1e3

        s = time.perf_counter()
        depth, valid = disparity_to_depth(calib, disparity)
        xs, ys, zs = reproject_xyz(calib, depth, fy)
        t["metric_depth"] = (time.perf_counter() - s) * 1e3

        s = time.perf_counter()
        phys = physical_mask(depth, valid, PHYSICAL_MAX_M)
        cells = spatial_cells(depth, valid, 1, 3, physical_valid=phys)
        occ = occupancy_grid(depth, phys, 4, 6, near_threshold_m=NEAR_THRESHOLD_M)
        cloud = depth_to_pointcloud(xs, ys, zs, phys, stride=CLOUD_STRIDE)
        t["cloud_spatial_occupancy"] = (time.perf_counter() - s) * 1e3

        s = time.perf_counter()
        mag, mag_valid = depth_discontinuity(depth, valid)
        t["discontinuity"] = (time.perf_counter() - s) * 1e3

        # Touch every result so nothing is optimized away and shapes stay honest.
        assert disparity.shape == (368, 1232)
        assert len(cells) and len(occ) and cloud["count"] >= 0 and mag.shape == depth.shape
        return t, disparity

    # Warmup (untimed, excluded from statistics).
    for _ in range(max(0, args.warmup)):
        run_pipeline(0)

    per_stage: dict[str, list[float]] = {k: [] for k in STAGES}
    totals: list[float] = []
    disp0 = None
    for i in range(n):
        t, disp = run_pipeline(i)
        if i == 0:
            disp0 = disp
        for k in STAGES:
            per_stage[k].append(t[k])
        totals.append(sum(t[k] for k in STAGES))

    stats = {}
    for k in STAGES:
        v = np.asarray(per_stage[k], dtype=np.float64)
        stats[k] = {"median_ms": float(np.median(v)), "min_ms": float(v.min()),
                    "max_ms": float(v.max()), "samples": per_stage[k]}
    tot = np.asarray(totals, dtype=np.float64)
    total_median = float(np.median(tot))
    fps = 1000.0 / total_median if total_median > 0 else float("nan")

    # CORRECTNESS GUARD (diagnostic only, no threshold): disparity for scene 0
    # vs the unmodified demo path at fp32/cpu.
    max_abs_diff = None
    if n > 0:
        ref_net, _ = AD.load_frozen_net("cpu")
        ref_net.eval()
        with torch.no_grad():
            ref_disp = AD.infer_disparity_scene(ref_net, "cpu", names[0])
        max_abs_diff = float(np.max(np.abs(disp0 - ref_disp)))

    device_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    out_path = Path(args.out) if args.out else (
        REPO / "stage_c_deploy" / "runtime" / "out" /
        f"bench_{dev}_{args.dtype}_s{n}_w{args.warmup}.json")
    payload = {
        "torch_version": torch.__version__, "device": dev, "device_name": device_name,
        "dtype": args.dtype, "scenes": n, "warmup": args.warmup,
        "scene_names": names, "checkpoint_sha256": AD.ARMP_SHA,
        "stages_ms": stats, "total_median_ms": total_median,
        "total_min_ms": float(tot.min()), "total_max_ms": float(tot.max()),
        "fps": fps, "scene0_max_abs_disparity_diff_vs_fp32cpu": max_abs_diff,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    w = 26
    print(f"host benchmark: device={dev} ({device_name})  dtype={args.dtype}  "
          f"scenes={n}  warmup={args.warmup}  torch={torch.__version__}")
    print(f"checkpoint sha256: {AD.ARMP_SHA}")
    print(f"{'stage':<{w}} {'median_ms':>10} {'min_ms':>10} {'max_ms':>10}")
    for k in STAGES:
        s = stats[k]
        print(f"{k:<{w}} {s['median_ms']:>10.2f} {s['min_ms']:>10.2f} "
              f"{s['max_ms']:>10.2f}")
    print(f"{'total_end_to_end':<{w}} {total_median:>10.2f} "
          f"{float(tot.min()):>10.2f} {float(tot.max()):>10.2f}")
    print(f"FPS (from total median): {fps:.2f}")
    if max_abs_diff is not None:
        print(f"scene0 ({names[0]}) max abs disparity diff vs fp32/cpu demo path: "
              f"{max_abs_diff:.6f} px  (diagnostic, no threshold)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
