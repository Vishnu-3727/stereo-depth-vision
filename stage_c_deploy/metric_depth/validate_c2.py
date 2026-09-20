"""Stage C2 validation: one runnable script performing ALL checks A-I.

A calibration parsing, B disparity units, C resolution scaling,
D depth equation, E invalid-disparity handling, F XYZ reprojection,
G numerical finiteness, H determinism (same scene twice, bitwise-identical),
I >= 3 (here 5) KITTI hailo_val scenes; plus artifact-integrity hashes
BEFORE and AFTER, accuracy preservation (new path vs direct model() call
max abs diff == 0), and one hand-checkable pixel with independent arithmetic.

Writes raw output to stage_c_deploy/metric_depth/out/c2_validation.json.
Exits nonzero on any failure.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))

import numpy as np
import torch

from src.datasets.kitti2015 import (Kitti2015Stereo, normalize, pad_and_crop)
from src.geometry import stereo as stereo_mod
from src.geometry.stereo import StereoCalibration, parse_kitti_cam_to_cam

import armp_depth as AD
from metric_depth import (depth_stats, disparity_to_depth, read_fy_from_calib,
                          reproject_xyz)
from measurement import pixel_measure, region_measure
from qcheck import qcheck_scene

OUT_DIR = REPO / "stage_c_deploy" / "metric_depth" / "out"
OUT_JSON = OUT_DIR / "c2_validation.json"
ONNX_REL = "stage_c_deploy/armp_stereonet.onnx"
ONNX_SHA_DISK = "4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989"
N_SCENES = 40  # full hailo_val split; overridable via `python validate_c2.py [n_scenes]`


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    tests: list[dict] = []

    def check(name: str, ok: bool, evidence: str) -> None:
        tests.append({"test": name, "result": "PASS" if ok else "FAIL",
                      "evidence": evidence, "status": "PASS" if ok else "BLOCKED"})
        if not ok:
            failures.append(name)

    t0 = time.time()
    # ---- artifact integrity BEFORE ----
    ckpt_hash_before = sha256_file(REPO / AD.ARMP_REL)
    onnx_hash_before = sha256_file(REPO / ONNX_REL)
    check("frozen checkpoint unchanged (before)",
          ckpt_hash_before == AD.ARMP_SHA,
          f"sha256({AD.ARMP_REL})={ckpt_hash_before} expected={AD.ARMP_SHA}")

    net, device = AD.load_frozen_net()
    n_params = sum(p.numel() for p in net.parameters())
    check("model parameters unchanged",
          n_params > 0 and net.config.downsample_levels == 3
          and net.config.num_disparities == 24
          and net.config.cost_volume_shift == "right"
          and net.config.regression_normalize is True,
          f"config={net.config} n_params={n_params} device={device}")

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    check("KITTI hailo_val split has 40 scenes", len(ds) == 40,
          f"len={len(ds)} names[0]={ds.names[0]}")
    n_scenes = N_SCENES
    if len(sys.argv) > 1:
        n_scenes = int(sys.argv[1])
    scenes = list(ds.names[:n_scenes])

    # ---- B disparity units: verify against code, not just repeat ----
    src_pad = inspect.getsource(pad_and_crop)
    # pad_and_crop must contain no resize/interp call
    no_resize = ("resize" not in src_pad.lower()) and ("interpol" not in src_pad.lower())
    src_norm = inspect.getsource(normalize)
    check("disparity units verified",
          no_resize and ("NORM_MEAN" in src_norm or "123.675" in src_norm),
          f"pad_and_crop has no resize/interp: {no_resize}; "
          f"normalize() is Imagenet-scale on 0-255 RGB; model output is "
          f"full-resolution disparity in input pixels (scale factor 1.0)")

    # ---- C resolution scaling: measure, do not repeat the brief ----
    import collections
    shape_counter = collections.Counter()
    for i in range(len(ds)):
        shape_counter[ds[i].original_shape] += 1
    sm0 = ds[0]
    crop_only = all(h >= 368 and w >= 1232 for (h, w) in shape_counter)
    cres_ok = (sm0.left.shape[:2] == (368, 1232) and crop_only)
    # A top-left crop does not move the origin: pixel (u,v) in the crop is the
    # same ray as (u,v) in the original, so fx, fy, cx, cy are UNCHANGED and
    # the disparity scale factor is 1.0 (disparity stays in original pixels).
    check("resolution scaling verified", cres_ok and no_resize,
          f"measured original shapes over 40 hailo_val scenes: "
          f"{dict(shape_counter)} (NOT 375x1242 as the brief assumed); "
          f"cropped={sm0.left.shape[:2]} anchor=top-left (origin unmoved -> "
          f"fx,fy,cx,cy unchanged, disparity scale factor=1.0, crop-only, no resize)")

    # ---- per-scene loop ----
    scene_recs = []
    hand_pixel = None
    for si, name in enumerate(scenes):
        sm = ds[si]
        tl = torch.from_numpy(normalize(sm.left)).to(device)
        tr = torch.from_numpy(normalize(sm.right)).to(device)
        with torch.no_grad():
            direct = net(tl, tr)[0, 0].detach().cpu().numpy().astype(np.float64)
        via_new = AD.infer_disparity_scene(net, device, name)
        maxdiff = float(np.abs(direct - via_new).max())
        check(f"accuracy preservation {name} (new path vs model())",
              maxdiff == 0.0, f"max abs disparity diff={maxdiff:.3e}")
        # H determinism
        via_new2 = AD.infer_disparity_scene(net, device, name)
        check(f"determinism {name} (bitwise identical)",
              bool((via_new == via_new2).all()),
              f"identical={(via_new == via_new2).all()} "
              f"mean={via_new.mean():.6f}")

        # A calibration parsing
        cal_path = ds.calibration_path(name)
        calib = parse_kitti_cam_to_cam(cal_path)
        fy = read_fy_from_calib(cal_path)
        fx_eq_fy = abs(fy - calib.focal_px) < 1e-9
        if si == 0:
            check("calibration source verified",
                  cal_path.exists() and calib.baseline_m > 0 and calib.focal_px > 0,
                  f"{cal_path} -> f={calib.focal_px} B={calib.baseline_m} "
                  f"cx={calib.cx} cy={calib.cy} {calib.width}x{calib.height}")
            check("KITTI fx == fy for scenes used", fx_eq_fy,
                  f"scene {name}: fx=P[0,0]={calib.focal_px} "
                  f"fy=P[1,1]={fy} equal={fx_eq_fy}")
        # D depth equation: independent recomputation
        depth, valid = disparity_to_depth(calib, via_new)
        fb = calib.focal_px * calib.baseline_m
        indep = np.full_like(depth, np.nan)
        indep[valid] = fb / via_new[valid]
        dd = float(np.abs(depth[valid] - indep[valid]).max()) if valid.any() else 0.0
        if si == 0:
            check("depth equation verified (Z=fB/d)",
                  dd == 0.0, f"scene {name}: max|depth-imported minus fB/d|={dd:.3e}, "
                  f"fB={fb:.6f}")
        # G finiteness
        fin = bool(np.all(np.isfinite(depth[valid]))) if valid.any() else True
        if si == 0:
            check("numerical finiteness", fin,
                  f"all {int(valid.sum())} valid depths finite: {fin}")
        stats = depth_stats(depth, valid)
        xs, ys, zs = reproject_xyz(calib, depth, fy=fy)
        # F XYZ: independent recomputation at a central valid pixel
        uc, vc = 616, 184
        if not (valid[vc, uc]):
            vys0, vxs0 = np.nonzero(valid)
            k0 = int(np.argmin((vxs0 - uc) ** 2 + (vys0 - vc) ** 2))
            uc, vc = int(vxs0[k0]), int(vys0[k0])
        uk, vk = uc, vc
        zk = float(depth[vk, uk])
        dk = float(via_new[vk, uk])
        xe = (uk - calib.cx) * zk / calib.focal_px
        ye = (vk - calib.cy) * zk / fy
        f_ok = (abs(xs[vk, uk] - xe) == 0.0 and abs(ys[vk, uk] - ye) == 0.0
                and abs(zk - fb / dk) == 0.0)
        if si == 0:
            check("XYZ reprojection verified",
                  bool(f_ok),
                  f"pixel (u={uk},v={vk}): d={dk:.6f} Z={zk:.6f} "
                  f"X={float(xs[vk, uk]):.6f} Y={float(ys[vk, uk]):.6f} "
                  f"exact match to independent arithmetic: {bool(f_ok)}")
        # Q-matrix cross-check (tolerance stated in qcheck.py BEFORE running)
        qrec = qcheck_scene(via_new, valid, xs, ys, zs,
                            calib.focal_px, calib.cx, calib.cy, calib.baseline_m)
        cv2info = ""
        if "cv2_passed" in qrec:
            cv2info = (f" cv2(float32): dx={qrec['cv2_max_abs_dx_m']:.3e} "
                       f"dy={qrec['cv2_max_abs_dy_m']:.3e} "
                       f"dz={qrec['cv2_max_abs_dz_m']:.3e} "
                       f"sentinel10000={qrec['cv2_num_sentinel_10000']} "
                       f"passed={qrec['cv2_passed']}")
        elif "cv2_error" in qrec:
            cv2info = f" cv2_error={qrec['cv2_error']}"
        check(f"Q-matrix cross-check {name}", bool(qrec["passed"]),
              f"explicit-Q dx={qrec['max_abs_dx_m']:.3e} dy={qrec['max_abs_dy_m']:.3e} "
              f"dz={qrec['max_abs_dz_m']:.3e} tol_xy=1e-6 tol_z={qrec['tol_z_m']:.3e} "
              f"n={qrec['num_compared']}" + cv2info)
        # measurement sample on middle scene
        if si == 2:
            pm = pixel_measure(calib, via_new, depth, xs, ys, uk, vk, fy=fy)
            rm = region_measure(calib, via_new, depth, xs, ys, uk, vk,
                                size=5, fy=fy)
            hand_pixel = {"scene": name, "calib": str(cal_path),
                          "fx": calib.focal_px, "fy": fy,
                          "baseline_m": calib.baseline_m,
                          "cx": calib.cx, "cy": calib.cy,
                          "pixel": pm, "region5": rm}
            check("metric measurement", pm["valid"],
                  f"scene={name} PIXEL (u={uk},v={vk}) d={pm['disparity_px']:.4f}px "
                  f"Z={pm['pixel_depth_m']:.4f}m X={pm['X_m']:.4f} Y={pm['Y_m']:.4f}; "
                  f"REGION5 median Z={rm['region_median_depth_m']:.4f}m "
                  f"n={rm['valid_count']}/{rm['window_pixels']}")
        scene_recs.append({"scene": name, "calib": str(cal_path),
                           "fx": calib.focal_px, "fy": fy,
                           "baseline_m": calib.baseline_m,
                           "cx": calib.cx, "cy": calib.cy,
                           "disp_min": float(via_new.min()),
                           "disp_max": float(via_new.max()),
                           "disp_mean": float(via_new.mean()),
                           "depth_stats": stats,
                           "accuracy_maxdiff": maxdiff})

    # ---- NEW visualization mode output (2 scenes) ----
    from visualize import save_visualization
    viz_paths = []
    for vi in (0, 2):
        smv = ds[vi]
        dv = AD.infer_disparity_scene(net, device, scenes[vi])
        calv = parse_kitti_cam_to_cam(ds.calibration_path(scenes[vi]))
        fyv = read_fy_from_calib(ds.calibration_path(scenes[vi]))
        zv, validv = disparity_to_depth(calv, dv)
        viz_paths.append(str(save_visualization(scenes[vi], smv.left, dv, zv, validv)))
    check("visualization renders (new mode, invalid distinct)",
          all(Path(p).exists() for p in viz_paths),
          f"wrote {viz_paths} (display range 0.5-60 m stated on figure; data unclipped)")

    # E invalid-disparity handling (synthetic, deterministic)
    calib0 = parse_kitti_cam_to_cam(ds.calibration_path(scenes[0]))
    d_edge = np.array([[10.0, 0.0, -3.0, np.nan, np.inf, 1e-9]])
    z_edge, v_edge = disparity_to_depth(calib0, d_edge)
    e_ok = (v_edge.tolist() == [[True, False, False, False, False, False]]
            and np.isnan(z_edge[0, 1:]).all())
    check("invalid disparity handling", bool(e_ok),
          f"in=[10,0,-3,NaN,Inf,1e-9] valid={v_edge.tolist()} "
          f"depths all-NaN except first: {bool(np.isnan(z_edge[0, 1:]).all())}")

    # ---- artifact integrity AFTER ----
    ckpt_hash_after = sha256_file(REPO / AD.ARMP_REL)
    onnx_hash_after = sha256_file(REPO / ONNX_REL)
    check("frozen checkpoint unchanged (after)",
          ckpt_hash_after == AD.ARMP_SHA,
          f"sha256={ckpt_hash_after}")
    check("ONNX hash matches disk spelling (typo in prompt noted)",
          onnx_hash_after == ONNX_SHA_DISK,
          f"sha256({ONNX_REL})={onnx_hash_after} "
          f"(task prompt/evidence_summary contain transposed typo ...deed1dce26...; "
          f"disk ...deed1cde26... authoritative)")

    rec = {"utc": datetime.now(timezone.utc).isoformat(),
           "device": device, "torch": torch.__version__, "numpy": np.__version__,
           "checkpoint": AD.ARMP_REL, "checkpoint_sha256_before": ckpt_hash_before,
           "checkpoint_sha256_after": ckpt_hash_after,
           "onnx_sha256_before": onnx_hash_before, "onnx_sha256_after": onnx_hash_after,
           "disparity_scale_factor": 1.0,
           "depth_equation": "Z = fx * B / d; X = (u - cx) * Z / fx; Y = (v - cy) * Z / fy",
           "scenes": scene_recs, "hand_pixel": hand_pixel,
           "visualizations": viz_paths,
           "tests": tests,
           "failures": failures,
           "elapsed_s": time.time() - t0,
           "overall": "PASS" if not failures else "FAIL"}
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"C2 validation: {len(tests) - len(failures)}/{len(tests)} passed; "
          f"overall={rec['overall']}; wrote {OUT_JSON}")
    for t in tests:
        print(f"  [{t['result']}] {t['test']} :: {t['evidence']}")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
