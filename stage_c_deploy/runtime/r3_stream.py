#!/usr/bin/env python
"""R3: sequence throughput of the host runtime path, with and without a
decode prefetch thread.

The per-frame benchmark in ``runtime_path.py`` times one frame at a time, so
the CPU-bound PNG decode and the GPU-bound inference are strictly serialised.
Over a *sequence* they need not be: frame N+1 can be decoded while frame N is
on the GPU. This script measures the end-to-end wall clock of a whole
sequence under both schedules and checks that the disparity is unchanged.

The per-frame work is identical in both modes: concurrent cv2 decode, GPU
normalize, the same frozen net, the torch-CUDA geometry ports with the frozen
numpy occupancy, cudnn.benchmark off. Only the *schedule* differs. No frozen
artefact is written; the checkpoint is read and its SHA-256 asserted.

Usage:
    python r3_stream.py --mode serial   --scenes 10 --repeats 3
    python r3_stream.py --mode prefetch --scenes 10 --repeats 3
    python r3_stream.py --mode both     --scenes 10 --repeats 3 --out <json>
    python r3_stream.py --mode both --dump-dir <dir>   # write disparities
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runtime_path as RP  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(HERE.parent / "metric_depth"))
import armp_depth as AD  # noqa: E402

from occupancy import occupancy_grid  # noqa: E402
from pointcloud import physical_mask  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, pad_and_crop  # noqa: E402
from src.geometry.stereo import parse_kitti_cam_to_cam  # noqa: E402

REPO = HERE.parents[1]


def _decode_one(ds: Kitti2015Stereo, name: str) -> dict:
    """CPU-side work for one frame: decode both PNGs, pad/crop, read calib.

    This is exactly the CPU half of the timed path; it touches no GPU, so it
    is what a prefetch thread can legitimately overlap with inference.
    """
    root = ds.root
    left, right = RP.decode_pair(root / "training" / "image_2" / name,
                                 root / "training" / "image_3" / name,
                                 concurrent=True)
    cal_path = ds.calibration_path(name)
    return {"name": name,
            "left": pad_and_crop(left), "right": pad_and_crop(right),
            "calib": parse_kitti_cam_to_cam(cal_path),
            "fy": RP.read_fy_from_calib(cal_path)}


def _gpu_frame(net, dev: str, item: dict) -> torch.Tensor:
    """GPU-side work for one frame: normalize, infer, full geometry.

    Returns the disparity tensor (kept on device, as the timed path does).
    """
    tl = RP.normalize_gpu(item["left"], dev)
    tr = RP.normalize_gpu(item["right"], dev)
    with torch.no_grad():
        disp_t = net(tl, tr)[0, 0].detach()
    calib, fy = item["calib"], item["fy"]
    depth_t, valid_t = RP.disparity_to_depth_gpu(disp_t, calib.fB)
    xs_t, ys_t, zs_t = RP.reproject_xyz_gpu(depth_t, calib.focal_px, fy,
                                            calib.cx, calib.cy)
    phys_t = RP.physical_mask_gpu(depth_t, valid_t, RP.PHYSICAL_MAX_M)
    cells = RP.spatial_cells_gpu(depth_t, valid_t, 1, 3, phys_t)
    # occupancy stays on frozen numpy: the torch port is correct but slower
    # (see README section 5), so one D2H copy of depth+valid, as in the
    # timed --gpu-geometry path.
    depth_np = depth_t.cpu().numpy()
    valid_np = valid_t.cpu().numpy()
    phys_np = physical_mask(depth_np, valid_np, RP.PHYSICAL_MAX_M)
    occ = occupancy_grid(depth_np, phys_np, 4, 6,
                         near_threshold_m=RP.NEAR_THRESHOLD_M)
    cloud = RP.pointcloud_gpu(xs_t, ys_t, zs_t, phys_t, stride=RP.CLOUD_STRIDE)
    mag_t, _ = RP.discontinuity_gpu(depth_t, valid_t)
    assert len(cells) and len(occ) and cloud["count"] >= 0 \
        and mag_t.shape == depth_t.shape
    return disp_t


def run_serial(net, dev, ds, names, dump_dir: Path | None) -> float:
    """Decode and GPU work strictly interleaved, one frame at a time."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for name in names:
        disp_t = _gpu_frame(net, dev, _decode_one(ds, name))
        if dump_dir is not None:
            np.save(dump_dir / f"{Path(name).stem}.npy",
                    disp_t.float().cpu().numpy())
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def run_prefetch(net, dev, ds, names, dump_dir: Path | None,
                 depth: int = 2) -> float:
    """One background thread decodes ahead while the main thread runs the GPU.

    cv2's decode releases the GIL, so the overlap is real rather than
    cooperative-only. Queue depth is bounded so memory stays flat.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)

    def producer() -> None:
        try:
            for name in names:
                q.put(_decode_one(ds, name))
        finally:
            q.put(None)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is None:
            break
        disp_t = _gpu_frame(net, dev, item)
        if dump_dir is not None:
            np.save(dump_dir / f"{Path(item['name']).stem}.npy",
                    disp_t.float().cpu().numpy())
    th.join()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="both",
                    choices=("serial", "prefetch", "both"))
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-frames", type=int, default=5)
    ap.add_argument("--queue-depth", type=int, default=2)
    ap.add_argument("--dump-dir", default=None,
                    help="write per-scene disparity .npy of the LAST repeat "
                         "of each mode (into <dir>/<mode>)")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("R3 needs CUDA")
    torch.backends.cudnn.benchmark = False
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    names = list(ds.names[:args.scenes])
    net, dev = AD.load_frozen_net("cuda")
    net.eval()

    for _ in range(args.warmup_frames):
        _gpu_frame(net, dev, _decode_one(ds, names[0]))

    modes = ("serial", "prefetch") if args.mode == "both" else (args.mode,)
    results: dict[str, dict] = {}
    for mode in modes:
        runner = run_serial if mode == "serial" else run_prefetch
        walls = []
        for r in range(args.repeats):
            dump = None
            if args.dump_dir and r == args.repeats - 1:
                dump = Path(args.dump_dir) / mode
                dump.mkdir(parents=True, exist_ok=True)
            kw = {} if mode == "serial" else {"depth": args.queue_depth}
            walls.append(runner(net, dev, ds, names, dump, **kw))
        w = np.asarray(walls, dtype=np.float64)
        results[mode] = {
            "wall_ms_per_repeat": walls,
            "median_wall_ms": float(np.median(w)),
            "spread_ms": float(w.max() - w.min()),
            "ms_per_frame": float(np.median(w)) / len(names),
            "fps": len(names) / (float(np.median(w)) / 1e3),
        }
        print(f"{mode:<9} median {results[mode]['median_wall_ms']:9.1f} ms "
              f"for {len(names)} frames = "
              f"{results[mode]['ms_per_frame']:6.2f} ms/frame, "
              f"{results[mode]['fps']:5.2f} FPS "
              f"(spread over {args.repeats} repeats: "
              f"{results[mode]['spread_ms']:.1f} ms)")

    payload = {
        "torch_version": torch.__version__, "device_name": torch.cuda.get_device_name(0),
        "dtype": "fp32", "scenes": len(names), "scene_names": names,
        "repeats": args.repeats, "warmup_frames": args.warmup_frames,
        "queue_depth": args.queue_depth,
        "checkpoint_sha256": AD.ARMP_SHA, "results": results,
    }
    if "serial" in results and "prefetch" in results:
        s, p = results["serial"], results["prefetch"]
        payload["gain_ms_per_frame"] = s["ms_per_frame"] - p["ms_per_frame"]
        payload["worst_spread_ms"] = max(s["spread_ms"], p["spread_ms"])
        print(f"gain {payload['gain_ms_per_frame']:.2f} ms/frame; worst "
              f"repeat spread {payload['worst_spread_ms']:.1f} ms over "
              f"{len(names)} frames "
              f"({payload['worst_spread_ms'] / len(names):.2f} ms/frame)")
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
