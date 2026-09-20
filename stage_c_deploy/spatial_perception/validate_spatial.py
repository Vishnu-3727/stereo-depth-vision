"""Stage C2.1 validation: deterministic CPU post-processing on frozen C2 depth.

Same shape as C2's validate_c2.py: named checks, a JSON record at
stage_c_deploy/spatial_perception/out/c2_1_validation.json, nonzero exit on
failure, default = all 40 hailo_val scenes with an optional [n_scenes] argv
override.

Checks: frozen checkpoint + ONNX hashes before and after; C2 depth unchanged
(C2.1 path vs independent fB/d recomputation, max abs diff == 0, bitwise);
XYZ cross-checked against qcheck.py's Q machinery; point-cloud coordinate
correctness (point i -> (u,v) reproduces X,Y,Z exactly); region measurement
against an INDEPENDENT hand calculation (sort+index, not np.median); invalid
handling; nearest-surface values from real data (min exact membership, p5
definitional check); synthetic spatial-map case; physical-mask separation;
determinism (same scene twice, bitwise-identical, all scenes).

Timing (DEVELOPMENT-MACHINE MEASUREMENT, never Hailo performance) is recorded
SEPARATELY: ARM-P inference, depth conversion, XYZ, spatial analysis,
visualization. Memory via tracemalloc + array nbytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "spatial_perception"))

import numpy as np
import torch

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.geometry.stereo import parse_kitti_cam_to_cam

import armp_depth as AD
from metric_depth import disparity_to_depth, read_fy_from_calib, reproject_xyz
from measurement import pixel_measure, region_measure
from qcheck import qcheck_scene

from pointcloud import depth_to_pointcloud, physical_mask
from spatial import spatial_cells
from discontinuity import depth_discontinuity, disparity_discontinuity
from occupancy import occupancy_grid
from visualize_spatial import save_spatial_visualization

OUT_DIR = REPO / "stage_c_deploy" / "spatial_perception" / "out"
OUT_JSON = OUT_DIR / "c2_1_validation.json"
ONNX_REL = "stage_c_deploy/armp_stereonet.onnx"
ONNX_SHA_DISK = "4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989"
N_SCENES = 40  # full hailo_val split; overridable via `python validate_spatial.py [n_scenes]`
PHYSICAL_CAP_M = 60.0  # reported SEPARATE mask only; never applied by default


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hand_median(vals: np.ndarray) -> float:
    """INDEPENDENT median: sort + middle index (no np.median / np.percentile)."""
    s = np.sort(np.asarray(vals, dtype=np.float64))
    n = s.size
    assert n > 0
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


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
    check("ONNX hash matches disk spelling (before)",
          onnx_hash_before == ONNX_SHA_DISK,
          f"sha256({ONNX_REL})={onnx_hash_before}")

    net, device = AD.load_frozen_net()
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    check("KITTI hailo_val split has 40 scenes", len(ds) == 40,
          f"len={len(ds)} names[0]={ds.names[0]} names[-1]={ds.names[-1]}")
    n_scenes = N_SCENES
    if len(sys.argv) > 1:
        n_scenes = int(sys.argv[1])
    scenes = list(ds.names[:n_scenes])

    # ---- synthetic spatial-map case (obvious expected answer) ----
    synth = np.full((30, 90), 50.0)
    synth[:, 30:60] = 10.0
    synth_cells = spatial_cells(synth, np.ones_like(synth, bool))
    synth_ok = (synth_cells[1]["nearest_valid_surface_m"] == 10.0
                and synth_cells[1]["min_m"] == 10.0
                and synth_cells[1]["median_m"] == 10.0
                and synth_cells[0]["nearest_valid_surface_m"] == 50.0
                and synth_cells[2]["nearest_valid_surface_m"] == 50.0)
    check("synthetic spatial-map case (near centre band)",
          bool(synth_ok),
          f"expected CENTER nearest/min/median=10.0, LEFT=RIGHT=50.0; got "
          f"CENTER={synth_cells[1]['nearest_valid_surface_m']}/"
          f"{synth_cells[1]['min_m']}/{synth_cells[1]['median_m']}, "
          f"LEFT={synth_cells[0]['nearest_valid_surface_m']}, "
          f"RIGHT={synth_cells[2]['nearest_valid_surface_m']}")

    # ---- invalid-depth handling (synthetic, deterministic) ----
    calib0 = parse_kitti_cam_to_cam(ds.calibration_path(scenes[0]))
    d_edge = np.array([[10.0, 0.0, -3.0, np.nan, np.inf, 1e-9, 0.03]])
    z_edge, v_edge = disparity_to_depth(calib0, d_edge)
    e_ok = (v_edge.tolist() == [[True, False, False, False, False, False, True]]
            and bool(np.isnan(z_edge[0, 1:6]).all())
            and np.isfinite(z_edge[0, 0]) and np.isfinite(z_edge[0, 6]))
    check("invalid-depth handling (C2 rules preserved)",
          bool(e_ok),
          f"in=[10,0,-3,NaN,Inf,1e-9,0.03] valid={v_edge.tolist()}; "
          f"0.03px stays C2-valid with Z={float(z_edge[0, 6]):.1f}m "
          f"(geometrically valid, physically unreliable -> separate mask)")

    # ---- physical mask separation (synthetic) ----
    zb = z_edge.copy()
    pm = physical_mask(z_edge, v_edge, PHYSICAL_CAP_M)
    p_ok = ((pm <= v_edge).all() and not pm[0, 6] and v_edge[0, 6]
            and np.array_equal(z_edge, zb, equal_nan=True))
    check("physical-range filter is a separate mask (never modifies depth)",
          bool(p_ok),
          f"cap={PHYSICAL_CAP_M}m: 0.03px pixel excluded from mask but kept "
          f"C2-valid; stored depth array unchanged: "
          f"{bool(np.array_equal(z_edge, zb, equal_nan=True))}")

    # ---- per-scene loop ----
    scene_recs = []
    sample = None
    acc = {"infer": 0.0, "depth": 0.0, "xyz": 0.0, "spatial": 0.0}
    peak_spatial_bytes = 0
    for si, name in enumerate(scenes):
        t = time.perf_counter()
        disp = AD.infer_disparity_scene(net, device, name)
        acc["infer"] += time.perf_counter() - t
        # determinism: same scene twice, bitwise-identical
        disp2 = AD.infer_disparity_scene(net, device, name)
        check(f"determinism {name} (inference bitwise-identical)",
              bool((disp == disp2).all()),
              f"identical={(disp == disp2).all()} mean={disp.mean():.6f}")

        cal_path = ds.calibration_path(name)
        calib = parse_kitti_cam_to_cam(cal_path)
        fy = read_fy_from_calib(cal_path)

        t = time.perf_counter()
        depth, valid = disparity_to_depth(calib, disp)
        acc["depth"] += time.perf_counter() - t

        # C2 depth unchanged: C2.1-imported path vs independent fB/d
        fb = calib.focal_px * calib.baseline_m
        indep = np.full_like(depth, np.nan)
        ok_pix = np.isfinite(disp) & (disp > 1e-3)
        indep[ok_pix] = fb / disp[ok_pix]
        same_nan = bool((np.isnan(depth) == np.isnan(indep)).all())
        vv = valid & ok_pix & np.isfinite(depth) & np.isfinite(indep)
        maxdiff = float(np.abs(depth[vv] - indep[vv]).max()) if vv.any() else 0.0
        check(f"C2 depth unchanged {name} (max abs diff == 0)",
              same_nan and maxdiff == 0.0,
              f"max|C2.1-depth minus fB/d|={maxdiff:.3e} NaN-pattern-identical={same_nan}")

        t = time.perf_counter()
        xs, ys, zs = reproject_xyz(calib, depth, fy=fy)
        acc["xyz"] += time.perf_counter() - t

        # XYZ cross-checked against the existing Q machinery
        qrec = qcheck_scene(disp, valid, xs, ys, zs,
                            calib.focal_px, calib.cx, calib.cy, calib.baseline_m)
        check(f"Q-matrix cross-check {name}", bool(qrec["passed"]),
              f"explicit-Q dx={qrec['max_abs_dx_m']:.3e} "
              f"dy={qrec['max_abs_dy_m']:.3e} dz={qrec['max_abs_dz_m']:.3e} "
              f"tol_z={qrec['tol_z_m']:.3e} n={qrec['num_compared']}")

        t = time.perf_counter()
        tracemalloc.start()
        pc = depth_to_pointcloud(xs, ys, zs, valid, stride=1)
        cells = spatial_cells(depth, valid)
        dmag, dmask = depth_discontinuity(depth, valid)
        smag, smask = disparity_discontinuity(disp, valid)
        occ = occupancy_grid(depth, valid)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        acc["spatial"] += time.perf_counter() - t
        peak_spatial_bytes = max(peak_spatial_bytes, int(peak))

        # spatial determinism: recompute, bitwise-identical
        pc_b = depth_to_pointcloud(xs, ys, zs, valid, stride=1)
        cells_b = spatial_cells(depth, valid)
        det_ok = (bool((pc["points"] == pc_b["points"]).all())
                  and bool((pc["us"] == pc_b["us"]).all())
                  and all(a["nearest_valid_surface_m"] == b["nearest_valid_surface_m"]
                          and a["min_m"] == b["min_m"] and a["median_m"] == b["median_m"]
                          for a, b in zip(cells, cells_b)))
        check(f"determinism {name} (spatial bitwise-identical)", bool(det_ok),
              f"points-identical={(pc['points'] == pc_b['points']).all()} "
              f"cells-identical={det_ok}")

        # point-cloud coordinate correctness: sample points map back exactly
        n = pc["count"]
        idx = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n >= 5 else list(range(n))
        pc_ok = True
        for i in idx:
            u, v = int(pc["us"][i]), int(pc["vs"][i])
            if not (valid[v, u] and pc["points"][i, 0] == np.float32(xs[v, u])
                    and pc["points"][i, 1] == np.float32(ys[v, u])
                    and pc["points"][i, 2] == np.float32(zs[v, u])):
                pc_ok = False
                break
        check(f"point-cloud coordinates {name}", bool(pc_ok),
              f"count={n} nbytes={pc['nbytes']}; "
              f"{len(idx)} sampled points reproduce X,Y,Z exactly at their (u,v)")

        # nearest-surface values come from real data (per cell)
        ns_ok = True
        ns_ev = []
        for cell in cells:
            v0, v1, u0, u1 = cell["bounds"]
            vals = depth[v0:v1, u0:u1][valid[v0:v1, u0:u1]]
            if vals.size == 0:
                ok_c = np.isnan(cell["nearest_valid_surface_m"]) and np.isnan(cell["min_m"])
            else:
                mn, p5 = cell["min_m"], cell["nearest_valid_surface_m"]
                frac_le = float((vals <= p5 * (1 + 1e-12) + 1e-12).mean())
                frac_ge = float((vals >= p5 - 1e-9).mean())
                ok_c = (bool((vals == mn).any())
                        and float(vals.min()) <= p5 <= float(vals.max())
                        and frac_le >= 0.05 - 1e-9 and frac_ge >= 0.94 - 1e-9)
            ns_ok = ns_ok and ok_c
            ns_ev.append(f"{cell['name']}:min_in_data+5th-def")
        check(f"nearest-surface from real data {name}", bool(ns_ok),
              "min exact member of valid depths; 5th pct within [min,max] with "
              f">=5% of samples at/below it: {'; '.join(ns_ev)}")

        # physical-mcap accounting on real data (separate mask, depth untouched)
        pm_real = physical_mask(depth, valid, PHYSICAL_CAP_M)
        n_excl = int(valid.sum() - pm_real.sum())
        pm_ok = bool((pm_real <= valid).all())
        if si == 0:
            check("physical mask is a C2-mask subset on real data", pm_ok,
                  f"scene {name}: C2-valid={int(valid.sum())} "
                  f"excluded-by-{PHYSICAL_CAP_M} m-cap={n_excl}; stored depth unmodified")

        # region measurement vs INDEPENDENT hand calculation (middle scene)
        uc, vc = 616, 184
        if not valid[vc, uc]:
            vys0, vxs0 = np.nonzero(valid)
            k0 = int(np.argmin((vxs0 - uc) ** 2 + (vys0 - vc) ** 2))
            uc, vc = int(vxs0[k0]), int(vys0[k0])
        if si == 2:
            rm = region_measure(calib, disp, depth, xs, ys, uc, vc, size=5, fy=fy)
            pm_ = pixel_measure(calib, disp, depth, xs, ys, uc, vc, fy=fy)
            r = 2
            h, w = disp.shape
            u0, u1 = max(uc - r, 0), min(uc + r + 1, w)
            v0, v1 = max(vc - r, 0), min(vc + r + 1, h)
            zw = depth[v0:v1, u0:u1]
            m = np.isfinite(zw)
            hz = hand_median(zw[m])
            hx = hand_median(xs[v0:v1, u0:u1][m])
            hy = hand_median(ys[v0:v1, u0:u1][m])
            hd = hand_median(disp[v0:v1, u0:u1][m])
            xe = (uc - calib.cx) * pm_["pixel_depth_m"] / calib.focal_px
            ye = (vc - calib.cy) * pm_["pixel_depth_m"] / fy
            hand_ok = (rm["region_median_depth_m"] == hz
                       and rm["region_median_X_m"] == hx
                       and rm["region_median_Y_m"] == hy
                       and rm["region_median_disparity_px"] == hd
                       and pm_["X_m"] == xe and pm_["Y_m"] == ye)
            check("region measurement vs independent hand calculation",
                  bool(hand_ok),
                  f"scene={name} (u={uc},v={vc}): hand-median "
                  f"Z={hz:.6f} X={hx:.6f} Y={hy:.6f} d={hd:.6f} vs module "
                  f"Z={rm['region_median_depth_m']:.6f} X={rm['region_median_X_m']:.6f} "
                  f"Y={rm['region_median_Y_m']:.6f} d={rm['region_median_disparity_px']:.6f}; "
                  f"pixel X/Y exact: {pm_['X_m'] == xe and pm_['Y_m'] == ye}")
            center_cell = next(c for c in cells if c["name"] == "CENTER")
            sample = {"scene": name, "u": uc, "v": vc,
                      "fx": calib.focal_px, "fy": fy,
                      "baseline_m": calib.baseline_m, "cx": calib.cx, "cy": calib.cy,
                      "pixel": pm_, "region5": rm,
                      "center_nearest_valid_surface_m":
                          center_cell["nearest_valid_surface_m"],
                      "center_min_m": center_cell["min_m"],
                      "center_median_m": center_cell["median_m"]}

        scene_recs.append({
            "scene": name, "fx": calib.focal_px, "fy": fy,
            "baseline_m": calib.baseline_m, "cx": calib.cx, "cy": calib.cy,
            "disp_min": float(disp.min()), "disp_max": float(disp.max()),
            "disp_mean": float(disp.mean()),
            "num_valid": int(valid.sum()),
            "percent_valid": 100.0 * float(valid.sum()) / valid.size,
            "point_count": pc["count"], "point_nbytes": int(pc["nbytes"]),
            "cells": [{k: c[k] for k in ("name", "valid_count", "percent_valid",
                                         "min_m", "nearest_valid_surface_m",
                                         "median_m")} for c in cells],
            "discontinuity_valid": int(dmask.sum()),
            "occupancy": [g["label"] for g in occ],
            "c2_maxdiff": maxdiff,
            "q_dx": qrec["max_abs_dx_m"], "q_dy": qrec["max_abs_dy_m"],
            "q_dz": qrec["max_abs_dz_m"],
            "excluded_by_60m_cap": n_excl})

    # ---- visualization (2 scenes, NEW mode) ----
    t = time.perf_counter()
    viz_paths = []
    viz_idx = [0, 2] if len(scenes) >= 3 else list(range(len(scenes)))
    for vi in viz_idx:
        smv = ds[vi]
        dv = AD.infer_disparity_scene(net, device, scenes[vi])
        calv = parse_kitti_cam_to_cam(ds.calibration_path(scenes[vi]))
        fyv = read_fy_from_calib(ds.calibration_path(scenes[vi]))
        zv, validv = disparity_to_depth(calv, dv)
        xv, yv, _ = reproject_xyz(calv, zv, fy=fyv)
        cellsv = spatial_cells(zv, validv)
        sel = next(c for c in cellsv if c["name"] == "CENTER")
        rmv = region_measure(calv, dv, zv, xv, yv, 616, 184, size=5, fy=fyv)
        pmv = pixel_measure(calv, dv, zv, xv, yv, 616, 184, fy=fyv)
        text = (f"SELECTED REGION CENTER: median={sel['median_m']:.2f} m "
                f"nearest={sel['nearest_valid_surface_m']:.2f} m "
                f"min={sel['min_m']:.2f} m | pixel(616,184): "
                f"Z={pmv['pixel_depth_m']:.2f} X={pmv['X_m']:.2f} "
                f"Y={pmv['Y_m']:.2f} region5-median={rmv['region_median_depth_m']:.2f}")
        viz_paths.append(str(save_spatial_visualization(
            scenes[vi], smv.left, dv, zv, validv, cellsv, "CENTER", text)))
    viz_time = time.perf_counter() - t
    check("visualization renders (new mode, 2 scenes)",
          all(Path(p).exists() for p in viz_paths),
          f"wrote {viz_paths} (display 0.5-60 m / 0-176 px; data unclipped, no smoothing)")

    # ---- artifact integrity AFTER ----
    ckpt_hash_after = sha256_file(REPO / AD.ARMP_REL)
    onnx_hash_after = sha256_file(REPO / ONNX_REL)
    check("frozen checkpoint unchanged (after)",
          ckpt_hash_after == AD.ARMP_SHA, f"sha256={ckpt_hash_after}")
    check("ONNX hash matches disk spelling (after)",
          onnx_hash_after == ONNX_SHA_DISK,
          f"sha256({ONNX_REL})={onnx_hash_after} "
          f"(prompt typo ...deed1dce26... noted; disk ...deed1cde26... authoritative)")

    # ---- resource accounting (DEVELOPMENT-MACHINE MEASUREMENT) ----
    n = len(scenes)
    per = {k: v / n for k, v in acc.items()}
    per["viz_per_fig"] = viz_time / len(viz_paths)
    # full-resolution buffers held at once: depth, valid, X, Y, Z (+disp input)
    h0, w0 = 368, 1232
    fullres_bytes = (5 * h0 * w0 * 8) + (h0 * w0) + (h0 * w0 * 8)  # X,Y,Z,depth,indep + valid + disp
    mem_info = {"tracemalloc_peak_spatial_bytes": peak_spatial_bytes,
                "fullres_buffers_bytes": int(fullres_bytes),
                "mean_point_nbytes": float(np.mean([s["point_nbytes"] for s in scene_recs])),
                "mean_point_count": float(np.mean([s["point_count"] for s in scene_recs]))}

    rec = {"utc": datetime.now(timezone.utc).isoformat(),
           "device": device, "torch": torch.__version__, "numpy": np.__version__,
           "checkpoint": AD.ARMP_REL, "checkpoint_sha256_before": ckpt_hash_before,
           "checkpoint_sha256_after": ckpt_hash_after,
           "onnx_sha256_before": onnx_hash_before, "onnx_sha256_after": onnx_hash_after,
           "physical_cap_m": PHYSICAL_CAP_M,
           "physical_note": "separate mask only; never applied by default; stored depth never modified",
           "coordinate_convention": "metres, camera optical centre at origin, "
                                    "+X right, +Y down, +Z forward, right-handed (reused from C2)",
           "timing_s": {"per_frame_mean": per, "totals": acc,
                        "viz_total_s": viz_time},
           "timing_note": "DEVELOPMENT-MACHINE MEASUREMENT, never Hailo performance",
           "memory": mem_info,
           "scenes": scene_recs, "sample": sample,
           "visualizations": viz_paths,
           "tests": tests, "failures": failures,
           "elapsed_s": time.time() - t0,
           "overall": "PASS" if not failures else "FAIL"}
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"C2.1 validation: {len(tests) - len(failures)}/{len(tests)} passed; "
          f"overall={rec['overall']}; wrote {OUT_JSON}")
    for t_ in tests:
        print(f"  [{t_['result']}] {t_['test']} :: {t_['evidence']}")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
