"""FAST runtime path for live use (Stage C deploy benchmark).

Live operation has no ground truth, so the GT disparity read is dropped from
the timed path (kept behind --with-gt for accuracy benchmarking only).

Optimizations vs bench_host.py (model, checkpoint, dtype=fp32, resolution,
and evaluation contract all unchanged; no torch.compile):
  1. no GT read in the timed load_preprocess region (default; --with-gt opts in)
  2. left/right PNG decode with cv2.imread, CONCURRENTLY via ThreadPoolExecutor
     (PNG decode releases the GIL). --decode seq|par (default par).
  3. normalize / float-convert / transpose on the GPU: upload uint8 HWC and do
     the arithmetic in torch float32 on cuda. --norm host|gpu (default gpu).

Opt-in round-2 flags (all default OFF; frozen numpy path stays the default):
  --gpu-geometry      torch-CUDA ports of metric_depth, cloud/spatial/
                      occupancy and discontinuity (this file only).
                      --verify-gpugeometry reports per-stage numbers-only
                      equivalence vs the frozen numpy on N scenes.
                      MEASURED EXCEPTION: the occupancy torch port is kept
                      OUT of the timed path (correct but slower than numpy,
                      15.0 vs 6.4 ms); the flag uses frozen numpy occupancy
                      on one D2H copy. See the inline comment.
  --cudnn-bench       torch.backends.cudnn.benchmark = True (default OFF:
                      deterministic benchmark=False; True stays measurable
                      but is nondeterministic)
  --channels-last     channels_last model + inputs
  --compile           torch.compile(model), mode=default, with clean fallback

REQUIRED CHECKS (run explicitly):
  --verify-decode   assert concurrent-cv2 arrays are bit-identical (exact ==)
                    to read_image() for all N scenes. Stops (exit 3) on mismatch.
  --verify-gpunorm  report max abs diff of the model INPUT tensor (host vs GPU
                    normalize) and of the resulting DISPARITY, for all N scenes.
                    Numbers only; no pass/fail threshold is invented.

Timing method matches bench_host.py exactly: 5 stages, N scenes from the head
of hailo_val, W warmup iters on scene 0, perf_counter regions, cuda synchronize
around load_preprocess and inference.

Usage:
    python stage_c_deploy/runtime/runtime_path.py --device cuda --scenes 10
    python stage_c_deploy/runtime/runtime_path.py --verify-decode
    python stage_c_deploy/runtime/runtime_path.py --verify-gpunorm --device cuda
    python stage_c_deploy/runtime/runtime_path.py --with-gt   # bench accuracy only
    python stage_c_deploy/runtime/runtime_path.py --profile   # step-5 cProfile probe

Opt-in checkpoint override (R4): --checkpoint <repo-relative|absolute path>
loads that .pth instead of the frozen one, asserting its measured sha256 at
load and recording path + measured sha in the JSON. Absent, the frozen
checkpoint is used exactly as before.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "spatial_perception"))

from src.datasets.kitti2015 import (  # noqa: E402
    NORM_MEAN,
    NORM_STD,
    Kitti2015Stereo,
    normalize,
    pad_and_crop,
    read_disparity_png,
    read_image,
)
from src.geometry.stereo import parse_kitti_cam_to_cam  # noqa: E402

import armp_depth as AD  # noqa: E402
from metric_depth import (  # noqa: E402
    disparity_to_depth,
    read_fy_from_calib,
    reproject_xyz,
)
from discontinuity import depth_discontinuity  # noqa: E402
from occupancy import occupancy_grid  # noqa: E402
from pointcloud import depth_to_pointcloud, physical_mask  # noqa: E402
from spatial import spatial_cells  # noqa: E402

import torch  # noqa: E402

PHYSICAL_MAX_M = 60.0
NEAR_THRESHOLD_M = 10.0
CLOUD_STRIDE = 3

STAGES = ("load_preprocess", "inference", "metric_depth",
          "cloud_spatial_occupancy", "discontinuity")


def resolve_checkpoint(arg: str | None) -> tuple[Path, str]:
    """Resolve --checkpoint to (absolute path, JSON label).

    None (flag absent) -> the frozen checkpoint (REPO / AD.ARMP_REL,
    label AD.ARMP_REL): behaviour is exactly as before. Otherwise the
    value may be repo-relative or absolute; the label is the repo-relative
    posix path when the file sits under REPO, else the path as given.
    """
    if arg is None:
        return REPO / AD.ARMP_REL, AD.ARMP_REL
    p = Path(arg)
    if not p.is_absolute():
        p = REPO / p
    try:
        label = p.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        label = str(p)
    return p, label


def load_net(device: str, ckpt: Path, ckpt_sha: str, override: bool):
    """Load the net, asserting the checkpoint sha without a vacuous check.

    override=False (no --checkpoint flag) -> AD.load_frozen_net(device) with
    no checkpoint arguments, so the frozen ARMP_SHA is asserted exactly as
    before. override=True -> the measured sha of the override file is
    asserted at load. The measured sha is always recorded in the JSON by the
    caller; this helper only chooses which assert runs.
    """
    if override:
        return AD.load_frozen_net(device, checkpoint=ckpt,
                                  expected_sha256=ckpt_sha)
    return AD.load_frozen_net(device)


def _cv2_read_rgb(path: Path) -> np.ndarray:
    """One PNG decode with cv2, BGR->RGB, identical to read_image()."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return bgr[:, :, ::-1].copy()


def decode_pair(left_path: Path, right_path: Path,
                concurrent: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Decode left+right PNGs; concurrent=True uses 2 threads (GIL-free)."""
    if not concurrent:
        return _cv2_read_rgb(left_path), _cv2_read_rgb(right_path)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fl = ex.submit(_cv2_read_rgb, left_path)
        fr = ex.submit(_cv2_read_rgb, right_path)
        return fl.result(), fr.result()


def normalize_gpu(u8_hwc: np.ndarray, dev: str) -> torch.Tensor:
    """Upload uint8 HWC RGB and normalize in torch float32 on `dev`.

    Mirrors src.datasets.kitti2015.normalize: (x - MEAN) / STD per channel,
    NCHW, contiguous, float32. `u8_hwc` must already be pad_and_crop'ed.
    """
    t = torch.from_numpy(u8_hwc).to(dev, dtype=torch.float32)  # HWC float32 on dev
    mean = torch.from_numpy(np.asarray(NORM_MEAN, dtype=np.float32)).to(dev)
    std = torch.from_numpy(np.asarray(NORM_STD, dtype=np.float32)).to(dev)
    t = (t - mean) / std
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()


def verify_decode(ds: Kitti2015Stereo, n: int) -> None:
    """REQUIRED CHECK step 2: bit-identical concurrent decode on N scenes."""
    print(f"verify-decode: {n} scenes, concurrent cv2 vs read_image()")
    n_bad = 0
    for i in range(n):
        name = ds.names[i]
        root = ds.root
        ref_l = read_image(root / "training" / "image_2" / name)
        ref_r = read_image(root / "training" / "image_3" / name)
        got_l, got_r = decode_pair(root / "training" / "image_2" / name,
                                   root / "training" / "image_3" / name,
                                   concurrent=True)
        same_l = (got_l.shape == ref_l.shape and got_l.dtype == ref_l.dtype
                  and bool(np.array_equal(got_l, ref_l)))
        same_r = (got_r.shape == ref_r.shape and got_r.dtype == ref_r.dtype
                  and bool(np.array_equal(got_r, ref_r)))
        nl = int(np.sum(got_l != ref_l)) if not same_l else 0
        nr = int(np.sum(got_r != ref_r)) if not same_r else 0
        status = "IDENTICAL" if (same_l and same_r) else (
            f"MISMATCH left_diff_elems={nl} right_diff_elems={nr}")
        print(f"  [{i}] {name}: {status}")
        if not (same_l and same_r):
            n_bad += 1
    if n_bad:
        print(f"verify-decode: {n_bad}/{n} scenes DIFFER -> keep original loader, STOP",
              file=sys.stderr)
        sys.exit(3)
    print(f"verify-decode: all {n} scenes bit-identical (exact ==).")


def verify_gpunorm(net: torch.nn.Module, ds: Kitti2015Stereo, dev: str,
                   n: int) -> None:
    """REQUIRED CHECK step 3: host-vs-GPU normalize input + disparity diffs."""
    is_cuda = dev == "cuda"
    print(f"verify-gpunorm: {n} scenes, host normalize vs GPU normalize ({dev})")
    worst_in = 0.0
    for i in range(n):
        name = ds.names[i]
        root = ds.root
        left = pad_and_crop(read_image(root / "training" / "image_2" / name))
        right = pad_and_crop(read_image(root / "training" / "image_3" / name))
        # host path (reference): numpy normalize then to(device)
        tl_np = normalize(left)
        tr_np = normalize(right)
        tl_ref = torch.from_numpy(tl_np).to(dev)
        tr_ref = torch.from_numpy(tr_np).to(dev)
        # GPU path
        if is_cuda:
            torch.cuda.synchronize()
        tl_gpu = normalize_gpu(left, dev)
        tr_gpu = normalize_gpu(right, dev)
        if is_cuda:
            torch.cuda.synchronize()
        dl = float((tl_gpu.detach().float().cpu()
                    - tl_ref.detach().float().cpu()).abs().max())
        dr = float((tr_gpu.detach().float().cpu()
                    - tr_ref.detach().float().cpu()).abs().max())
        din = max(dl, dr)
        worst_in = max(worst_in, din)
        with torch.no_grad():
            if is_cuda:
                torch.cuda.synchronize()
            d_ref = net(tl_ref, tr_ref)[0, 0].detach().float().cpu().numpy()
            if is_cuda:
                torch.cuda.synchronize()
            d_gpu = net(tl_gpu, tr_gpu)[0, 0].detach().float().cpu().numpy()
            if is_cuda:
                torch.cuda.synchronize()
        dd = float(np.max(np.abs(d_gpu - d_ref)))
        print(f"  [{i}] {name}: input_max_abs_diff={din:.6e} "
              f"disparity_max_abs_diff={dd:.6e} px")
    print(f"verify-gpunorm: worst input diff over {n} scenes = {worst_in:.6e}. "
          f"(numbers only, no threshold)")


# ---------------------------------------------------------------------------
# GPU geometry ports (opt-in via --gpu-geometry; torch on cuda, inside this
# file only). The frozen numpy implementations in stage_c_deploy/metric_depth
# and stage_c_deploy/spatial_perception stay untouched and remain the default
# timed path. These ports mirror them op-for-op in float64 so the numbers can
# be compared directly (see --verify-gpugeometry, numbers only, no threshold).
#
# NOTE on median: torch.median does NOT match numpy.median for even-sized
# inputs (lower-middle vs average-of-two-middles). torch.quantile(..., 0.5)
# with default linear interpolation DOES match numpy.median / percentile, so
# all cell statistics below use torch.quantile / torch.nanquantile.
# ---------------------------------------------------------------------------

def disparity_to_depth_gpu(disp_t: torch.Tensor, fB: float,
                           min_disp: float = 1e-3
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """torch-CUDA port of metric_depth.disparity_to_depth (float64)."""
    d = disp_t.to(dtype=torch.float64)
    valid = (d > min_disp) & torch.isfinite(d)
    fb = torch.tensor(fB, device=d.device, dtype=torch.float64)
    nan = torch.tensor(float("nan"), device=d.device, dtype=torch.float64)
    depth = torch.where(valid, fb / d, nan)
    return depth, valid


def reproject_xyz_gpu(depth_t: torch.Tensor, fx: float, fy: float,
                      cx: float, cy: float
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch-CUDA port of metric_depth.reproject_xyz (float64).

    Uses 1-D aranges + broadcasting instead of materialised full-res u/v
    grids; per-element op order ((u-cx)*z/fx) matches the numpy version.
    """
    h, w = depth_t.shape
    dev = depth_t.device
    uc = torch.arange(w, device=dev, dtype=torch.float64) - cx
    vc = torch.arange(h, device=dev, dtype=torch.float64) - cy
    xs = uc[None, :] * depth_t / fx
    ys = vc[:, None] * depth_t / fy
    return xs, ys, depth_t.clone()


def physical_mask_gpu(depth_t: torch.Tensor, valid_t: torch.Tensor,
                      max_m: float) -> torch.Tensor:
    """torch-CUDA port of pointcloud.physical_mask."""
    return valid_t & torch.isfinite(depth_t) & (depth_t <= max_m)


def _cell_edges(h: int, w: int, n_rows: int, n_cols: int):
    """Same edge formula as the frozen spatial/occupancy code (host, cheap)."""
    row_edges = np.linspace(0, h, n_rows + 1).astype(int)
    col_edges = np.linspace(0, w, n_cols + 1).astype(int)
    return row_edges, col_edges


def spatial_cells_gpu(depth_t: torch.Tensor, valid_t: torch.Tensor,
                      n_rows: int = 1, n_cols: int = 3,
                      physical_valid_t: torch.Tensor | None = None) -> list[dict]:
    """torch-CUDA port of spatial.spatial_cells (float64, 1x3 default).

    Per-cell min/p5/median via one batched torch.nanquantile over NaN-padded
    rows (matches numpy percentile/median with linear interpolation).
    """
    from spatial import cell_name  # frozen; import only, never modified

    use_t = valid_t & physical_valid_t if physical_valid_t is not None \
        else valid_t
    filt = physical_valid_t is not None
    h, w = depth_t.shape
    row_edges, col_edges = _cell_edges(h, w, n_rows, n_cols)
    dev = depth_t.device

    bounds, val_list, n_pix_list, n_valid_list = [], [], [], []
    for r in range(n_rows):
        for c in range(n_cols):
            v0, v1 = int(row_edges[r]), int(row_edges[r + 1])
            u0, u1 = int(col_edges[c]), int(col_edges[c + 1])
            cell_use = use_t[v0:v1, u0:u1]
            n_pix = int(cell_use.numel())
            n_valid = int(cell_use.sum().item())
            vals = depth_t[v0:v1, u0:u1][cell_use] if n_valid \
                else None
            bounds.append((r, c, v0, v1, u0, u1))
            val_list.append(vals)
            n_pix_list.append(n_pix)
            n_valid_list.append(n_valid)

    K = len(bounds)
    q = torch.tensor([0.05, 0.5], device=dev, dtype=torch.float64)
    lmax = max((int(v.numel()) for v in val_list if v is not None),
               default=0)
    if lmax > 0:
        nan = float("nan")
        padded = torch.full((K, lmax), nan, device=dev,
                            dtype=torch.float64)
        for i, v in enumerate(val_list):
            if v is not None:
                padded[i, :v.numel()] = v
        qq = torch.nanquantile(padded, q, dim=1)  # (2, K)
        mins = torch.where(torch.isnan(padded),
                           torch.tensor(float("inf"), device=dev,
                                        dtype=torch.float64),
                           padded).min(dim=1).values
    else:
        qq, mins = None, None

    cells = []
    for i, (r, c, v0, v1, u0, u1) in enumerate(bounds):
        n_pix = n_pix_list[i]
        n_valid = n_valid_list[i]
        if n_valid and qq is not None:
            mn = float(mins[i].item())
            p5 = float(qq[0, i].item())
            med = float(qq[1, i].item())
        else:
            mn = p5 = med = float("nan")
        cells.append({"name": cell_name(r, c, n_rows, n_cols),
                      "row": r, "col": c,
                      "bounds": (v0, v1, u0, u1),
                      "pixels": n_pix, "valid_count": n_valid,
                      "percent_valid": 100.0 * n_valid / n_pix if n_pix else 0.0,
                      "min_m": mn,
                      "nearest_valid_surface_m": p5,
                      "median_m": med,
                      "physical_filter_applied": filt})
    return cells


def occupancy_grid_gpu(depth_t: torch.Tensor, valid_t: torch.Tensor,
                       n_rows: int = 4, n_cols: int = 6,
                       near_threshold_m: float = 10.0,
                       min_valid_fraction: float = 0.1) -> list[dict]:
    """torch-CUDA port of occupancy.occupancy_grid (float64).

    Per-cell medians via one batched torch.nanquantile over NaN-padded rows.
    """
    from occupancy import INSUFFICIENT_DATA, NEAR, FAR  # frozen; import only

    h, w = depth_t.shape
    row_edges, col_edges = _cell_edges(h, w, n_rows, n_cols)
    dev = depth_t.device

    bounds, val_list, frac_list = [], [], []
    for r in range(n_rows):
        for c in range(n_cols):
            v0, v1 = int(row_edges[r]), int(row_edges[r + 1])
            u0, u1 = int(col_edges[c]), int(col_edges[c + 1])
            cell_valid = valid_t[v0:v1, u0:u1]
            frac = float(cell_valid.to(torch.float64).mean().item())
            vals = depth_t[v0:v1, u0:u1][cell_valid] \
                if int(cell_valid.sum().item()) else None
            bounds.append((r, c, v0, v1, u0, u1))
            val_list.append(vals)
            frac_list.append(frac)

    K = len(bounds)
    lmax = max((int(v.numel()) for v in val_list if v is not None),
               default=0)
    if lmax > 0:
        padded = torch.full((K, lmax), float("nan"), device=dev,
                            dtype=torch.float64)
        for i, v in enumerate(val_list):
            if v is not None:
                padded[i, :v.numel()] = v
        meds = torch.nanquantile(
            padded, torch.tensor(0.5, device=dev, dtype=torch.float64),
            dim=1)
    else:
        meds = None

    out = []
    for i, (r, c, v0, v1, u0, u1) in enumerate(bounds):
        frac = frac_list[i]
        if frac < min_valid_fraction or val_list[i] is None:
            label, med = INSUFFICIENT_DATA, float("nan")
        else:
            med = float(meds[i].item())
            label = NEAR if med <= near_threshold_m else FAR
        out.append({"row": r, "col": c, "bounds": (v0, v1, u0, u1),
                    "label": label, "median_m": med,
                    "valid_fraction": frac,
                    "near_threshold_m": float(near_threshold_m),
                    "min_valid_fraction": float(min_valid_fraction)})
    return out


def pointcloud_gpu(xs_t: torch.Tensor, ys_t: torch.Tensor, zs_t: torch.Tensor,
                   valid_t: torch.Tensor, stride: int = 1) -> dict:
    """torch-CUDA port of pointcloud.depth_to_pointcloud.

    Points/us/vs stay on the GPU in the timed path (only the scalar count is
    synced); the equivalence check copies them to host explicitly.
    """
    assert stride >= 1 and int(stride) == stride, stride
    h, w = valid_t.shape
    sv = valid_t[::stride, ::stride]
    sx = xs_t[::stride, ::stride]
    sy = ys_t[::stride, ::stride]
    sz = zs_t[::stride, ::stride]
    vs_sub, us_sub = torch.nonzero(sv, as_tuple=True)
    us = (us_sub * stride).to(torch.int64)
    vs = (vs_sub * stride).to(torch.int64)
    points = torch.stack([sx[sv], sy[sv], sz[sv]], dim=-1).to(torch.float32)
    count = int(sv.sum().item())
    assert points.shape == (count, 3), points.shape
    return {"points": points, "us": us, "vs": vs,
            "count": count,
            "nbytes": int(points.numel() * 4 + us.numel() * 8 + vs.numel() * 8),
            "points_nbytes": int(points.numel() * 4),
            "stride": int(stride),
            "full_shape": (int(h), int(w))}


def discontinuity_gpu(field_t: torch.Tensor,
                      valid_t: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """torch-CUDA port of discontinuity._forward_mag (float64)."""
    f = field_t.to(dtype=torch.float64)
    v = valid_t
    h, w = f.shape
    dev = f.device
    nan = torch.tensor(float("nan"), device=dev, dtype=torch.float64)
    mag = torch.full_like(f, float("nan"))
    if h < 2 or w < 2:
        return mag, torch.zeros_like(v)
    du_ok = v[:, :-1] & v[:, 1:]
    dv_ok = v[:-1, :] & v[1:, :]
    du = torch.full_like(f, float("nan"))
    dv = torch.full_like(f, float("nan"))
    du[:, :-1] = torch.where(du_ok, torch.abs(f[:, 1:] - f[:, :-1]), nan)
    dv[:-1, :] = torch.where(dv_ok, torch.abs(f[1:, :] - f[:-1, :]), nan)
    both = torch.zeros_like(v)
    both[:-1, :-1] = du_ok[:-1, :] & dv_ok[:, :-1]
    m = both & torch.isfinite(du) & torch.isfinite(dv)
    mag[m] = torch.hypot(du[m], dv[m])
    return mag, m


def fast_run_scene(index: int, device: str | None = None) -> dict:
    """Demo entry point for the deterministic fast path (opt-in).

    Same contract as ``stage_c_deploy/demo/pipeline_demo.py::run_scene``:
    identical keys and value semantics (numpy arrays of the same dtype and
    shape the demo draws and click-measures). Deterministic configuration:
    concurrent-cv2 decode, GPU normalize, torch-CUDA geometry ports with the
    frozen numpy occupancy on one D2H copy (the timed ``--gpu-geometry``
    path), ``cudnn.benchmark = False``. On CPU (no CUDA) the frozen numpy
    geometry is used instead. The model, checkpoint, dtype (fp32),
    resolution and evaluation contract are unchanged.
    """
    from metric_depth import depth_stats  # frozen; local import only

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    assert 0 <= index < len(ds), f"scene {index} outside 0..{len(ds) - 1}"
    name = ds.names[index]

    net, dev = AD.load_frozen_net(device)
    net.eval()
    is_cuda = dev == "cuda"
    if is_cuda:
        torch.backends.cudnn.benchmark = False

    root = ds.root
    left_u8, right_u8 = decode_pair(root / "training" / "image_2" / name,
                                    root / "training" / "image_3" / name,
                                    concurrent=True)
    gt = read_disparity_png(ds.gt_dir / name, scale=ds.disparity_scale)
    left_u8 = pad_and_crop(left_u8)
    right_u8 = pad_and_crop(right_u8)
    gt = pad_and_crop(gt)
    cal_path = ds.calibration_path(name)
    calib = parse_kitti_cam_to_cam(cal_path)
    fy = read_fy_from_calib(cal_path)

    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if is_cuda:
        tl = normalize_gpu(left_u8, dev)
        tr = normalize_gpu(right_u8, dev)
    else:
        tl = torch.from_numpy(normalize(left_u8)).to(dev)
        tr = torch.from_numpy(normalize(right_u8)).to(dev)
    with torch.no_grad():
        out = net(tl, tr)
    if is_cuda:
        disp_t = out[0, 0].detach()
        torch.cuda.synchronize()
    else:
        disp_np_cpu = out[0, 0].detach().float().cpu().numpy().astype(np.float64)
        disp_t = None
    t_infer = time.perf_counter() - t0

    t1 = time.perf_counter()
    if is_cuda:
        assert disp_t is not None
        depth_t, valid_t = disparity_to_depth_gpu(disp_t, calib.fB)
        xs_t, ys_t, zs_t = reproject_xyz_gpu(depth_t, calib.focal_px, fy,
                                             calib.cx, calib.cy)
        phys_t = physical_mask_gpu(depth_t, valid_t, PHYSICAL_MAX_M)
        cells = spatial_cells_gpu(depth_t, valid_t, 1, 3, phys_t)
        # Frozen numpy occupancy on one D2H copy: the torch port is correct
        # but slower, so the timed --gpu-geometry path keeps numpy here.
        depth_np = depth_t.cpu().numpy()
        valid_np = valid_t.cpu().numpy()
        phys_np = physical_mask(depth_np, valid_np, PHYSICAL_MAX_M)
        occ = occupancy_grid(depth_np, phys_np, 4, 6,
                             near_threshold_m=NEAR_THRESHOLD_M)
        cloud_g = pointcloud_gpu(xs_t, ys_t, zs_t, phys_t,
                                 stride=CLOUD_STRIDE)
        mag_t, mag_valid_t = discontinuity_gpu(depth_t, valid_t)
        torch.cuda.synchronize()
        disparity = disp_t.detach().float().cpu().numpy().astype(np.float64)
        depth = depth_np
        valid = valid_np
        phys = phys_t.cpu().numpy()
        xs = xs_t.cpu().numpy()
        ys = ys_t.cpu().numpy()
        zs = zs_t.cpu().numpy()
        mag = mag_t.cpu().numpy()
        mag_valid = mag_valid_t.cpu().numpy()
        cloud = {"points": cloud_g["points"].float().cpu().numpy(),
                 "us": cloud_g["us"].cpu().numpy(),
                 "vs": cloud_g["vs"].cpu().numpy(),
                 "count": int(cloud_g["count"]),
                 "nbytes": int(cloud_g["nbytes"]),
                 "points_nbytes": int(cloud_g["points_nbytes"]),
                 "stride": int(cloud_g["stride"]),
                 "full_shape": (int(cloud_g["full_shape"][0]),
                                int(cloud_g["full_shape"][1]))}
    else:
        disparity = disp_np_cpu
        depth, valid = disparity_to_depth(calib, disparity)
        xs, ys, zs = reproject_xyz(calib, depth, fy)
        phys = physical_mask(depth, valid, PHYSICAL_MAX_M)
        cells = spatial_cells(depth, valid, 1, 3, physical_valid=phys)
        occ = occupancy_grid(depth, phys, 4, 6,
                             near_threshold_m=NEAR_THRESHOLD_M)
        mag, mag_valid = depth_discontinuity(depth, valid)
        cloud = depth_to_pointcloud(xs, ys, zs, phys, stride=CLOUD_STRIDE)
    t_post = time.perf_counter() - t1

    stats = depth_stats(depth, phys)
    gtm = gt > 0
    gn = int(gtm.sum())
    if gn:
        gerr = np.abs(disparity[gtm] - gt[gtm])
        gd1 = float(np.mean((gerr > 3.0) & (gerr > 0.05 * gt[gtm])) * 100.0)
        accuracy = {"n": gn, "epe": float(gerr.mean()), "d1": gd1}
    else:
        accuracy = {"n": 0, "epe": float("nan"), "d1": float("nan")}

    return {
        "name": name, "index": index, "device": dev,
        "left": left_u8, "gt": gt,
        "disparity": disparity, "depth": depth, "valid": valid,
        "phys": phys, "xs": xs, "ys": ys, "zs": zs,
        "calib": calib, "fy": fy,
        "cells": cells, "occupancy": occ, "mag": mag, "mag_valid": mag_valid,
        "cloud": cloud,
        "stats": stats,
        "accuracy": accuracy,
        "t_infer": t_infer, "t_post": t_post,
    }


def _nanaware_maxdiff(a: np.ndarray, b: np.ndarray,
                      mask: np.ndarray) -> float:
    """Max |a-b| over mask, ignoring NaN==NaN (both-NaN treated as 0)."""
    if not bool(mask.any()):
        return float("nan")
    da = a[mask].astype(np.float64)
    db = b[mask].astype(np.float64)
    both_nan = np.isnan(da) & np.isnan(db)
    d = np.abs(da - db)
    d[both_nan] = 0.0
    if np.isnan(d).any():  # one-sided NaN: report as inf, count via masks
        return float("inf")
    return float(d.max())


def verify_gpugeometry(net: torch.nn.Module, ds: Kitti2015Stereo, dev: str,
                       n: int) -> dict:
    """MANDATORY CHECK: GPU ports vs frozen numpy on N scenes, numbers only.

    Reports, per scene and worst-over-N: depth max abs diff (m), valid-mask
    mismatch fraction, discontinuity mag_valid mismatch fraction + magnitude
    max abs diff, spatial/occupancy max abs diff of every reported float value
    (+ label/name mismatch counts), cloud count diff + points max abs diff.
    No pass/fail threshold is invented.
    """
    from spatial import spatial_cells  # noqa: E402  (frozen reference)
    from occupancy import occupancy_grid  # noqa: E402  (frozen reference)
    from pointcloud import (  # noqa: E402  (frozen reference)
        depth_to_pointcloud, physical_mask)
    from metric_depth import (  # noqa: E402  (frozen reference)
        disparity_to_depth, read_fy_from_calib, reproject_xyz)
    from discontinuity import depth_discontinuity  # noqa: E402 (frozen ref)

    is_cuda = dev == "cuda"
    print(f"verify-gpugeometry: {n} scenes, frozen numpy vs torch-CUDA ports")
    worst: dict[str, float] = {
        "depth_max_abs_diff_m": 0.0, "valid_mismatch_frac": 0.0,
        "xs_max_abs_diff_m": 0.0, "ys_max_abs_diff_m": 0.0,
        "disc_valid_mismatch_frac": 0.0, "disc_mag_max_abs_diff": 0.0,
        "spatial_max_abs_diff": 0.0, "occupancy_max_abs_diff": 0.0,
        "cloud_points_max_abs_diff": 0.0,
    }
    worst_counts = {"spatial_name_mismatch": 0, "occupancy_label_mismatch": 0,
                    "cloud_count_diff": 0, "cloud_uv_mismatch": 0}
    for i in range(n):
        name = ds.names[i]
        root = ds.root
        left = pad_and_crop(read_image(root / "training" / "image_2" / name))
        right = pad_and_crop(read_image(root / "training" / "image_3" / name))
        tl = normalize_gpu(left, dev)
        tr = normalize_gpu(right, dev)
        cal_path = ds.calibration_path(name)
        calib = parse_kitti_cam_to_cam(cal_path)
        fy = read_fy_from_calib(cal_path)
        with torch.no_grad():
            if is_cuda:
                torch.cuda.synchronize()
            out = net(tl, tr)
            disp_t = out[0, 0].detach()
            if is_cuda:
                torch.cuda.synchronize()
        disp_np = disp_t.float().cpu().numpy().astype(np.float64)

        # frozen numpy reference
        depth_np, valid_np = disparity_to_depth(calib, disp_np)
        xs_np, ys_np, zs_np = reproject_xyz(calib, depth_np, fy)
        phys_np = physical_mask(depth_np, valid_np, PHYSICAL_MAX_M)
        cells_np = spatial_cells(depth_np, valid_np, 1, 3,
                                 physical_valid=phys_np)
        occ_np = occupancy_grid(depth_np, phys_np, 4, 6,
                                near_threshold_m=NEAR_THRESHOLD_M)
        cloud_np = depth_to_pointcloud(xs_np, ys_np, zs_np, phys_np,
                                       stride=CLOUD_STRIDE)
        mag_np, mv_np = depth_discontinuity(depth_np, valid_np)

        # GPU ports (same disparity)
        if is_cuda:
            torch.cuda.synchronize()
        depth_t, valid_t = disparity_to_depth_gpu(disp_t, calib.fB)
        xs_t, ys_t, zs_t = reproject_xyz_gpu(depth_t, calib.focal_px, fy,
                                             calib.cx, calib.cy)
        phys_t = physical_mask_gpu(depth_t, valid_t, PHYSICAL_MAX_M)
        cells_g = spatial_cells_gpu(depth_t, valid_t, 1, 3, phys_t)
        occ_g = occupancy_grid_gpu(depth_t, phys_t, 4, 6,
                                   near_threshold_m=NEAR_THRESHOLD_M)
        cloud_g = pointcloud_gpu(xs_t, ys_t, zs_t, phys_t,
                                 stride=CLOUD_STRIDE)
        mag_t, mv_t = discontinuity_gpu(depth_t, valid_t)
        if is_cuda:
            torch.cuda.synchronize()
        depth_g = depth_t.cpu().numpy()
        valid_g = valid_t.cpu().numpy()
        xs_g = xs_t.cpu().numpy()
        ys_g = ys_t.cpu().numpy()
        mag_g = mag_t.cpu().numpy()
        mv_g = mv_t.cpu().numpy()

        vm = float(np.mean(valid_np != valid_g))
        both_v = valid_np & valid_g
        dd = _nanaware_maxdiff(depth_np, depth_g, both_v)
        dx = _nanaware_maxdiff(xs_np, xs_g, both_v)
        dy = _nanaware_maxdiff(ys_np, ys_g, both_v)
        dm = float(np.mean(mv_np != mv_g))
        both_m = mv_np & mv_g
        dg = _nanaware_maxdiff(mag_np, mag_g, both_m)

        skeys = ("min_m", "nearest_valid_surface_m", "median_m",
                 "percent_valid")
        sd = 0.0
        snm = 0
        for a, b in zip(cells_np, cells_g):
            if a["name"] != b["name"]:
                snm += 1
            for k in skeys:
                va, vb = float(a[k]), float(b[k])
                if np.isnan(va) and np.isnan(vb):
                    continue
                if np.isnan(va) or np.isnan(vb):
                    sd = float("inf")
                else:
                    sd = max(sd, abs(va - vb))
            if a["valid_count"] != b["valid_count"]:
                sd = float("inf")

        okeys = ("median_m", "valid_fraction")
        od = 0.0
        olm = 0
        for a, b in zip(occ_np, occ_g):
            if a["label"] != b["label"]:
                olm += 1
            for k in okeys:
                va, vb = float(a[k]), float(b[k])
                if np.isnan(va) and np.isnan(vb):
                    continue
                if np.isnan(va) or np.isnan(vb):
                    od = float("inf")
                else:
                    od = max(od, abs(va - vb))

        cd = abs(int(cloud_np["count"]) - int(cloud_g["count"]))
        if cd == 0 and int(cloud_np["count"]):
            pg = cloud_g["points"].float().cpu().numpy()
            pd = float(np.max(np.abs(cloud_np["points"] - pg)))
            uvm = int(np.sum(cloud_np["us"] != cloud_g["us"].cpu().numpy())
                      + np.sum(cloud_np["vs"] != cloud_g["vs"].cpu().numpy()))
        else:
            pd = 0.0 if cd == 0 else float("inf")
            uvm = 0 if cd == 0 else -1

        worst["depth_max_abs_diff_m"] = max(worst["depth_max_abs_diff_m"], dd)
        worst["valid_mismatch_frac"] = max(worst["valid_mismatch_frac"], vm)
        worst["xs_max_abs_diff_m"] = max(worst["xs_max_abs_diff_m"], dx)
        worst["ys_max_abs_diff_m"] = max(worst["ys_max_abs_diff_m"], dy)
        worst["disc_valid_mismatch_frac"] = max(
            worst["disc_valid_mismatch_frac"], dm)
        worst["disc_mag_max_abs_diff"] = max(
            worst["disc_mag_max_abs_diff"], dg)
        worst["spatial_max_abs_diff"] = max(worst["spatial_max_abs_diff"], sd)
        worst["occupancy_max_abs_diff"] = max(
            worst["occupancy_max_abs_diff"], od)
        worst["cloud_points_max_abs_diff"] = max(
            worst["cloud_points_max_abs_diff"], pd)
        worst_counts["spatial_name_mismatch"] += snm
        worst_counts["occupancy_label_mismatch"] += olm
        worst_counts["cloud_count_diff"] += cd
        worst_counts["cloud_uv_mismatch"] += uvm
        print(f"  [{i}] {name}: depth_diff={dd:.3e}m valid_mm={vm:.3e} "
              f"xs={dx:.3e} ys={dy:.3e} disc_mm={dm:.3e} disc_mag={dg:.3e} "
              f"spatial={sd:.3e} occ={od:.3e} cloud_n={cd} cloud_pts={pd:.3e}")
    print(f"verify-gpugeometry WORST over {n} scenes: " +
          " ".join(f"{k}={v:.3e}" for k, v in worst.items()) +
          f" name_mm={worst_counts['spatial_name_mismatch']} "
          f"label_mm={worst_counts['occupancy_label_mismatch']} "
          f"cloud_count_diff={worst_counts['cloud_count_diff']} "
          f"cloud_uv_mm={worst_counts['cloud_uv_mismatch']} "
          f"(numbers only, no threshold)")
    return {"worst": worst, "mismatch_counts": worst_counts}


def apply_channels_last(net: torch.nn.Module) -> torch.nn.Module:
    """Stride-only conversion: 4D params/buffers -> channels_last, 5D ->
    channels_last_3d. Values are untouched. (Plain
    ``net.to(memory_format=torch.channels_last)`` raises on this model's 3D
    aggregation weights, hence the per-tensor conversion.)"""
    with torch.no_grad():
        for t in list(net.parameters()) + list(net.buffers()):
            if t.dim() == 4 and not t.is_contiguous(
                    memory_format=torch.channels_last):
                t.data = t.data.to(memory_format=torch.channels_last)
            elif t.dim() == 5 and not t.is_contiguous(
                    memory_format=torch.channels_last_3d):
                t.data = t.data.to(memory_format=torch.channels_last_3d)
    return net


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="cpu | cuda (default: cuda if available else cpu)")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--with-gt", action="store_true",
                    help="also read GT disparity (accuracy benchmarking only; "
                         "live path default is no GT)")
    ap.add_argument("--decode", default="par", choices=("seq", "par"),
                    help="seq = sequential cv2 | par = concurrent (default par)")
    ap.add_argument("--norm", default="gpu", choices=("host", "gpu"),
                    help="host = numpy normalize | gpu = torch normalize (default gpu)")
    ap.add_argument("--verify-decode", action="store_true")
    ap.add_argument("--verify-gpunorm", action="store_true")
    ap.add_argument("--gpu-geometry", action="store_true",
                    help="opt-in: torch-CUDA ports of metric_depth, "
                         "cloud/spatial/occupancy and discontinuity "
                         "(frozen numpy stays the default)")
    ap.add_argument("--verify-gpugeometry", action="store_true",
                    help="numbers-only equivalence of the GPU ports vs the "
                         "frozen numpy on N scenes (no threshold)")
    ap.add_argument("--cudnn-bench", action="store_true", default=False,
                    help="opt-in inference: torch.backends.cudnn.benchmark=True "
                         "(default OFF = deterministic: benchmark=False)")
    ap.add_argument("--channels-last", action="store_true",
                    help="opt-in inference: channels_last model + inputs")
    ap.add_argument("--compile", action="store_true",
                    help="opt-in inference: torch.compile(model) with clean "
                         "fallback if compilation fails")
    ap.add_argument("--profile", action="store_true",
                    help="step-5 probe: cProfile the host-numpy stages on one scene")
    ap.add_argument("--checkpoint", default=None,
                    help="opt-in checkpoint override: repo-relative or absolute "
                         "path to a .pth file. Absent = the frozen checkpoint "
                         "(default behaviour unchanged). The file's measured "
                         "sha256 is asserted at load and recorded in the JSON.")
    return ap.parse_args()


def run_profile(net: torch.nn.Module, ds: Kitti2015Stereo, dev: str) -> None:
    """Step 5 (measurement only): cProfile one scene's host-numpy stages."""
    import cProfile
    import io
    import pstats

    name = ds.names[0]
    root = ds.root
    left = pad_and_crop(read_image(root / "training" / "image_2" / name))
    right = pad_and_crop(read_image(root / "training" / "image_3" / name))
    tl = normalize_gpu(left, dev) if dev == "cuda" else \
        torch.from_numpy(normalize(left)).to(dev)
    tr = normalize_gpu(right, dev) if dev == "cuda" else \
        torch.from_numpy(normalize(right)).to(dev)
    with torch.no_grad():
        out = net(tl, tr)
    disparity = out[0, 0].detach().float().cpu().numpy().astype(np.float64)
    cal_path = ds.calibration_path(name)
    calib = parse_kitti_cam_to_cam(cal_path)
    fy = read_fy_from_calib(cal_path)

    def host_stages() -> None:
        depth, valid = disparity_to_depth(calib, disparity)
        xs, ys, zs = reproject_xyz(calib, depth, fy)
        phys = physical_mask(depth, valid, PHYSICAL_MAX_M)
        spatial_cells(depth, valid, 1, 3, physical_valid=phys)
        occupancy_grid(depth, phys, 4, 6, near_threshold_m=NEAR_THRESHOLD_M)
        depth_to_pointcloud(xs, ys, zs, phys, stride=CLOUD_STRIDE)
        depth_discontinuity(depth, valid)

    host_stages()  # warm the caches once, outside the profile
    pr = cProfile.Profile()
    pr.enable()
    host_stages()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(f"cProfile of metric_depth + cloud/spatial/occupancy + discontinuity "
          f"on scene {name}:")
    print(s.getvalue())


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda"):
        print(f"unknown --device {device!r}", file=sys.stderr)
        sys.exit(2)
    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device available for --device cuda", file=sys.stderr)
        sys.exit(2)

    kitti_root = REPO / "data" / "kitti2015"
    ckpt, ckpt_label = resolve_checkpoint(args.checkpoint)
    if not ckpt.exists():
        print(f"checkpoint missing: {ckpt}", file=sys.stderr)
        sys.exit(2)
    ckpt_sha = AD.sha256_file(ckpt)
    try:
        ds = Kitti2015Stereo(kitti_root, split="hailo_val")
    except FileNotFoundError as e:
        print(f"KITTI data missing: {e}", file=sys.stderr)
        sys.exit(2)
    n = min(args.scenes, len(ds))
    names = list(ds.names[:n])
    concurrent = args.decode == "par"
    use_gpu_norm = args.norm == "gpu"

    if args.verify_decode:
        verify_decode(ds, n)
        if not args.verify_gpunorm and args.profile is False:
            # decode-only check requested; still continue to timed run
            pass

    net, dev = load_net(device, ckpt, ckpt_sha,
                      override=args.checkpoint is not None)
    net.eval()
    is_cuda = dev == "cuda"

    # --- opt-in inference attacks (round B; all default off) ---
    use_cl = bool(args.channels_last)
    compile_requested = bool(args.compile)
    compile_status = "not_requested"
    compile_time_s = None
    if args.cudnn_bench and is_cuda:
        torch.backends.cudnn.benchmark = True
    elif is_cuda:
        # Deterministic default: cudnn autotune stays OFF unless --cudnn-bench
        # opts in (benchmark=True is measured but nondeterministic).
        torch.backends.cudnn.benchmark = False
    if args.cudnn_bench and not is_cuda:
        print("--cudnn-bench needs cuda; ignored on cpu", file=sys.stderr)
    if use_cl and is_cuda:
        net = apply_channels_last(net)
    if use_cl and not is_cuda:
        print("--channels-last needs cuda; ignored on cpu", file=sys.stderr)
        use_cl = False
    if args.gpu_geometry and not is_cuda:
        print("--gpu-geometry needs --device cuda", file=sys.stderr)
        sys.exit(2)
    gpu_geom = bool(args.gpu_geometry) and is_cuda
    if compile_requested:
        try:
            net = torch.compile(net)  # mode="default"; lazy, compiles on 1st run
            compile_status = "compiled"
        except Exception as e:  # e.g. no Triton on Windows: clean fallback
            print(f"torch.compile wrap failed ({type(e).__name__}: {e}); "
                  f"continuing uncompiled", file=sys.stderr)
            compile_status = f"fallback_wrap_{type(e).__name__}"

    if args.verify_gpunorm:
        verify_gpunorm(net, ds, dev, n)

    geom_equiv = None
    if args.verify_gpugeometry:
        geom_equiv = verify_gpugeometry(net, ds, dev, n)

    if args.profile:
        run_profile(net, ds, dev)
        return

    def run_pipeline(idx: int) -> tuple[dict, np.ndarray]:
        t: dict = {}
        if is_cuda:
            torch.cuda.synchronize()
        s = time.perf_counter()
        name = names[idx]
        root = ds.root
        left_u8, right_u8 = decode_pair(root / "training" / "image_2" / name,
                                       root / "training" / "image_3" / name,
                                       concurrent=concurrent)
        gt = None
        if args.with_gt:
            gt = read_disparity_png(ds.gt_dir / name, scale=ds.disparity_scale)
        left_u8 = pad_and_crop(left_u8)
        right_u8 = pad_and_crop(right_u8)
        if gt is not None:
            gt = pad_and_crop(gt)
        if use_gpu_norm and is_cuda:
            torch.cuda.synchronize()  # decode done; upload+norm timed on GPU below
        if use_gpu_norm and is_cuda:
            tl = normalize_gpu(left_u8, dev)
            tr = normalize_gpu(right_u8, dev)
        else:
            tl_np = normalize(left_u8)
            tr_np = normalize(right_u8)
            tl = torch.from_numpy(tl_np).to(dev)
            tr = torch.from_numpy(tr_np).to(dev)
        if use_cl and is_cuda:
            tl = tl.to(memory_format=torch.channels_last)
            tr = tr.to(memory_format=torch.channels_last)
        cal_path = ds.calibration_path(name)
        calib = parse_kitti_cam_to_cam(cal_path)
        fy = read_fy_from_calib(cal_path)
        if is_cuda:
            torch.cuda.synchronize()
        t["load_preprocess"] = (time.perf_counter() - s) * 1e3

        if gpu_geom:
            # GPU-geometry path: disparity stays on the device (the old D2H
            # inside inference is eliminated, not moved).
            if is_cuda:
                torch.cuda.synchronize()
            s = time.perf_counter()
            with torch.no_grad():
                out = net(tl, tr)
            disp_t = out[0, 0].detach()
            if is_cuda:
                torch.cuda.synchronize()
            t["inference"] = (time.perf_counter() - s) * 1e3

            s = time.perf_counter()
            depth_t, valid_t = disparity_to_depth_gpu(disp_t, calib.fB)
            xs_t, ys_t, zs_t = reproject_xyz_gpu(depth_t, calib.focal_px, fy,
                                                 calib.cx, calib.cy)
            if is_cuda:
                torch.cuda.synchronize()
            t["metric_depth"] = (time.perf_counter() - s) * 1e3

            s = time.perf_counter()
            phys_t = physical_mask_gpu(depth_t, valid_t, PHYSICAL_MAX_M)
            cells = spatial_cells_gpu(depth_t, valid_t, 1, 3, phys_t)
            # occupancy: the torch port is CORRECT (verify-gpugeometry:
            # label_mm=0, max diff ~1e-16) but SLOWER than frozen numpy
            # (15.0 ms vs 6.4 ms medians: 24 small per-cell sorts; numpy
            # sorts 19k float64 in 0.24 ms, no copies). Keep frozen numpy
            # here: one D2H of depth+valid, phys recomputed on host.
            depth_np = depth_t.cpu().numpy()
            valid_np = valid_t.cpu().numpy()
            phys_np = physical_mask(depth_np, valid_np, PHYSICAL_MAX_M)
            occ = occupancy_grid(depth_np, phys_np, 4, 6,
                                 near_threshold_m=NEAR_THRESHOLD_M)
            cloud = pointcloud_gpu(xs_t, ys_t, zs_t, phys_t,
                                   stride=CLOUD_STRIDE)
            if is_cuda:
                torch.cuda.synchronize()
            t["cloud_spatial_occupancy"] = (time.perf_counter() - s) * 1e3

            s = time.perf_counter()
            mag_t, mag_valid_t = discontinuity_gpu(depth_t, valid_t)
            if is_cuda:
                torch.cuda.synchronize()
            t["discontinuity"] = (time.perf_counter() - s) * 1e3

            assert disp_t.shape == (368, 1232)
            assert len(cells) and len(occ) and cloud["count"] >= 0 \
                and mag_t.shape == depth_t.shape
            return t, None, disp_t

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

        assert disparity.shape == (368, 1232)
        assert len(cells) and len(occ) and cloud["count"] >= 0 \
            and mag.shape == depth.shape
        return t, disparity, None

    if compile_requested and compile_status == "compiled":
        # First call triggers compilation (lazy): time it separately from the
        # steady state, with a clean fallback to a fresh uncompiled net.
        try:
            _t0 = time.perf_counter()
            run_pipeline(0)
            compile_time_s = time.perf_counter() - _t0
        except Exception as e:
            print(f"torch.compile first run failed ({type(e).__name__}: {e}); "
                  f"falling back to uncompiled net", file=sys.stderr)
            net, _ = load_net(device, ckpt, ckpt_sha,
                              override=args.checkpoint is not None)
            net.eval()
            if use_cl and is_cuda:
                net = apply_channels_last(net)
            compile_status = f"fallback_run_{type(e).__name__}"
            compile_time_s = None

    for _ in range(max(0, args.warmup)):
        run_pipeline(0)

    per_stage: dict[str, list[float]] = {k: [] for k in STAGES}
    totals: list[float] = []
    disp0 = None
    for i in range(n):
        t, disp, disp_gpu = run_pipeline(i)
        if i == 0:
            disp0 = disp if disp is not None else disp_gpu
        for k in STAGES:
            per_stage[k].append(t[k])
        totals.append(sum(t[k] for k in STAGES))
    if isinstance(disp0, torch.Tensor):
        # gpu-geometry path kept disparity on device; materialise scene 0
        # once, untimed, for the scene0 diagnostic only.
        if is_cuda:
            torch.cuda.synchronize()
        disp0 = disp0.detach().float().cpu().numpy().astype(np.float64)

    stats = {}
    for k in STAGES:
        v = np.asarray(per_stage[k], dtype=np.float64)
        stats[k] = {"median_ms": float(np.median(v)), "min_ms": float(v.min()),
                    "max_ms": float(v.max()), "samples": per_stage[k]}
    tot = np.asarray(totals, dtype=np.float64)
    total_median = float(np.median(tot))
    fps = 1000.0 / total_median if total_median > 0 else float("nan")

    max_abs_diff = None
    if n > 0:
        ref_net, _ = AD.load_frozen_net("cpu")
        ref_net.eval()
        with torch.no_grad():
            ref_disp = AD.infer_disparity_scene(ref_net, "cpu", names[0])
        max_abs_diff = float(np.max(np.abs(disp0 - ref_disp)))

    device_name = torch.cuda.get_device_name(0) if is_cuda else "cpu"
    any_opt = bool(gpu_geom or args.cudnn_bench or use_cl or compile_requested)
    out_path = Path(args.out) if args.out else (
        REPO / "stage_c_deploy" / "runtime" / "out" /
        f"runtime_{dev}_fp32_{'opt_' if any_opt else ''}s{n}_w{args.warmup}.json")
    payload = {
        "torch_version": torch.__version__, "device": dev, "device_name": device_name,
        "dtype": "fp32", "scenes": n, "warmup": args.warmup,
        "scene_names": names, "checkpoint": ckpt_label,
        "checkpoint_sha256": ckpt_sha,
        "runtime_path": {"with_gt": bool(args.with_gt), "decode": args.decode,
                          "norm": args.norm, "gpu_geometry": bool(gpu_geom),
                          "cudnn_bench": bool(args.cudnn_bench),
                          "channels_last": bool(use_cl),
                          "compile_requested": bool(compile_requested),
                          "compile_status": compile_status,
                          "compile_time_s": compile_time_s},
        "stages_ms": stats, "total_median_ms": total_median,
        "total_min_ms": float(tot.min()), "total_max_ms": float(tot.max()),
        "fps": fps, "scene0_max_abs_disparity_diff_vs_fp32cpu": max_abs_diff,
    }
    if geom_equiv is not None:
        payload["gpu_geometry_equivalence"] = geom_equiv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    w = 26
    print(f"runtime path: device={dev} ({device_name}) dtype=fp32 scenes={n} "
          f"warmup={args.warmup} torch={torch.__version__} "
          f"with_gt={args.with_gt} decode={args.decode} norm={args.norm} "
          f"gpu_geometry={gpu_geom} cudnn_bench={bool(args.cudnn_bench)} "
          f"channels_last={use_cl} compile={compile_requested}/{compile_status}")
    if compile_time_s is not None:
        print(f"torch.compile first-run (compile+run) time: {compile_time_s:.2f} s "
              f"(excluded from steady-state medians)")
    print(f"checkpoint: {ckpt_label} sha256: {ckpt_sha}")
    print(f"{'stage':<{w}} {'median_ms':>10} {'min_ms':>10} {'max_ms':>10}")
    for k in STAGES:
        s_ = stats[k]
        print(f"{k:<{w}} {s_['median_ms']:>10.2f} {s_['min_ms']:>10.2f} "
              f"{s_['max_ms']:>10.2f}")
    print(f"{'total_end_to_end':<{w}} {total_median:>10.2f} "
          f"{float(tot.min()):>10.2f} {float(tot.max()):>10.2f}")
    print(f"FPS (from total median): {fps:.2f}")
    if max_abs_diff is not None:
        print(f"scene0 ({names[0]}) max abs disparity diff vs fp32/cpu demo path: "
              f"{max_abs_diff:.6f} px  (diagnostic, no threshold)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
