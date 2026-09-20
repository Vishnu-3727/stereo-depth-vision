"""Stage C2.1.1: depth-discontinuity validation gate (evidence only).

The discontinuity OPERATOR (stage_c_deploy/spatial_perception/discontinuity.py)
is FROZEN for this task -- this validator never modifies it, never re-derives
expected answers from it. Every synthetic expectation below is a hand-derived
literal built from the documented forward-difference convention:

    du[v,u] = |F[v,u+1] - F[v,u]|,  dv[v,u] = |F[v+1,u] - F[v,u]|
    magnitude = hypot(du, dv), defined ONLY where both hold,
    i.e. mask True exactly on [:-1, :-1] for fully-valid input
    (last row / last column always invalid, by design).

Named checks, prints each, writes
stage_c_deploy/spatial_perception/out/c2_1_1_discontinuity_validation.json,
exits nonzero on any failure.
"""

from __future__ import annotations

import hashlib
import json
import math
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

from discontinuity import depth_discontinuity, disparity_discontinuity

OUT_JSON = (REPO / "stage_c_deploy" / "spatial_perception" / "out"
            / "c2_1_1_discontinuity_validation.json")
DISC_REL = "stage_c_deploy/spatial_perception/discontinuity.py"
ONNX_REL = "stage_c_deploy/armp_stereonet.onnx"
ONNX_SHA_DISK = "4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989"
CKPT_SHA = "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"
# Transposed typo carried (again) by the task prompt; disk is authoritative.
ONNX_SHA_PROMPT_TYPO = ONNX_SHA_DISK.replace("deed1cde26", "deed1dce26")
REAL_SCENE = "000160_10.png"
MAG_DESCRIBE_THRESHOLD_M_PER_PX = 1.0
TIMING_REPS = 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    tests: list[dict] = []

    def check(name: str, ok: bool, evidence: str) -> None:
        tests.append({"test": name, "result": "PASS" if ok else "FAIL",
                      "evidence": evidence,
                      "status": "PASS" if ok else "BLOCKED"})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} :: {evidence}")
        if not ok:
            failures.append(name)

    t0 = time.time()
    disc_hash_before = sha256_file(REPO / DISC_REL)
    ckpt_hash_before = sha256_file(
        REPO / "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth")
    onnx_hash_before = sha256_file(REPO / ONNX_REL)
    check("frozen discontinuity.py present (before)",
          (REPO / DISC_REL).exists(),
          f"sha256({DISC_REL})={disc_hash_before}")
    check("frozen checkpoint unchanged (before)",
          ckpt_hash_before == CKPT_SHA,
          f"sha256={ckpt_hash_before} expected={CKPT_SHA}")
    check("ONNX hash matches disk spelling (before)",
          onnx_hash_before == ONNX_SHA_DISK,
          f"sha256({ONNX_REL})={onnx_hash_before} "
          f"(prompt typo {ONNX_SHA_PROMPT_TYPO} noted; disk authoritative)")

    # ---- 1. vertical boundary: A=10 left of k=5, B=30 right ----
    A, B, K = 10.0, 30.0, 5
    STEP = 20.0  # |B - A|, hand-derived literal
    fv = np.full((6, 10), A)
    fv[:, K:] = B
    mag, mv = depth_discontinuity(fv, np.ones_like(fv, bool))
    # Hand-derived: step between col 4 and col 5 -> du[.,4] = 20;
    # dv = 0 everywhere; magnitude 20 at col 4, 0 at other valid cols.
    exp_mask = np.zeros_like(fv, bool)
    exp_mask[:-1, :-1] = True
    ok_mask = bool((mv == exp_mask).all())
    ok_step = bool((mag[:-1, K - 1] == STEP).all())
    left_zero = bool((mag[:-1, :K - 1] == 0.0).all())
    right_zero = bool((mag[:-1, K:-1] == 0.0).all())
    check("synthetic vertical boundary (step at col k, magnitude at k-1)",
          ok_mask and ok_step and left_zero and right_zero,
          f"A={A} B={B} k={K}: expected mag=={STEP} at col {K - 1} "
          f"(got {mag[0, K - 1]}), 0.0 at all other valid cols "
          f"(left_zero={left_zero} right_zero={right_zero}); "
          f"mask==[:-1,:-1] exactly: {ok_mask}")

    # ---- 2. horizontal boundary: A=10 above k=5, B=30 below ----
    fh = np.full((8, 6), A)
    fh[K:, :] = B
    magh, mvh = depth_discontinuity(fh, np.ones_like(fh, bool))
    exp_mask_h = np.zeros_like(fh, bool)
    exp_mask_h[:-1, :-1] = True
    ok_h = (bool((mvh == exp_mask_h).all())
            and bool((magh[K - 1, :-1] == STEP).all())
            and bool((magh[:K - 1, :-1] == 0.0).all())
            and bool((magh[K:-1, :-1] == 0.0).all()))
    check("synthetic horizontal boundary (step at row k, magnitude at k-1)",
          ok_h,
          f"A={A} B={B} k={K}: expected mag=={STEP} at row {K - 1} "
          f"(got {magh[K - 1, 0]}), 0.0 elsewhere valid; "
          f"mask==[:-1,:-1] exactly: {bool((mvh == exp_mask_h).all())}")

    # ---- 3. constant depth ----
    fc = np.full((5, 7), 12.5)
    magc, mvc = depth_discontinuity(fc, np.ones_like(fc, bool))
    exp_mask_c = np.zeros_like(fc, bool)
    exp_mask_c[:-1, :-1] = True
    ok_c = (bool((mvc == exp_mask_c).all())
            and bool((magc[mvc] == 0.0).all()))
    check("synthetic constant depth (magnitude 0 where valid)",
          ok_c,
          f"expected 0.0 everywhere mask True, mask True exactly on [:-1,:-1]; "
          f"got mag[0,0]={magc[0, 0]}, valid_count={int(mvc.sum())} "
          f"(expected {(fc.shape[0] - 1) * (fc.shape[1] - 1)})")

    # ---- 4. multiple regions: TL=10 TR=14 / BL=19 BR=23 ----
    # Vertical step 4 (cols 3|4), horizontal step 9 (rows 2|3).
    # Corner pixel (2,3): du=|14-10|=4, dv=|19-10|=9 -> hypot(4,9)=sqrt(97).
    fm = np.full((6, 8), 10.0)
    fm[:, 4:] = 14.0
    fm[3:, :4] = 19.0
    fm[3:, 4:] = 23.0
    magm, mvm = depth_discontinuity(fm, np.ones_like(fm, bool))
    exp_corner = math.hypot(4.0, 9.0)  # sqrt(97) ~= 9.848857801796104
    corner_ok = abs(float(magm[2, 3]) - exp_corner) <= 1e-12
    edge_u_ok = bool(magm[0, 3] == 4.0)
    edge_v_ok = bool(magm[2, 0] == 9.0)
    flat_ok = bool(magm[0, 0] == 0.0)
    # At (2,4): du=|14-14|=0; dv=|23-14|=9 -> hypot(0,9)=9.0 exactly.
    br9_ok = bool(magm[2, 4] == 9.0)
    exp_mask_m = np.zeros_like(fm, bool)
    exp_mask_m[:-1, :-1] = True
    mask_m_ok = bool((mvm == exp_mask_m).all())
    ok_m = corner_ok and edge_u_ok and edge_v_ok and flat_ok and br9_ok and mask_m_ok
    check("synthetic multiple regions (corner hypot(step_u, step_v))",
          ok_m,
          f"expected mag[2,3]==hypot(4,9)={exp_corner:.15f} (got {magm[2, 3]:.15f}, "
          f"|diff|={abs(float(magm[2, 3]) - exp_corner):.3e} <= 1e-12); "
          f"mag[0,3]==4.0 (got {magm[0, 3]}); mag[2,0]==9.0 (got {magm[2, 0]}); "
          f"mag[0,0]==0.0 (got {magm[0, 0]}); mag[2,4]==9.0 (got {magm[2, 4]}); "
          f"mask==[:-1,:-1]: {mask_m_ok}")

    # ---- 5. invalid-pixel poisoning: single invalid pixel at (1,2) ----
    # Hand-derived from the validity rule: du needs both horizontal neighbours,
    # dv needs both vertical neighbours; magnitude needs both.
    # du_ok false at (1,1),(1,2); dv_ok false at (0,2),(1,2);
    # both (=mag mask on [:-1,:-1]) false exactly at (0,2),(1,1),(1,2).
    fp = np.full((4, 5), 7.0)
    vp = np.ones_like(fp, bool)
    vp[1, 2] = False
    fp[1, 2] = np.nan
    _, mvp = depth_discontinuity(fp, vp)
    exp_mask_p = np.zeros_like(fp, bool)
    exp_mask_p[:-1, :-1] = True
    exp_mask_p[0, 2] = False
    exp_mask_p[1, 1] = False
    exp_mask_p[1, 2] = False
    poison_ok = bool((mvp == exp_mask_p).all())
    spot_ok = (not mvp[1, 2] and not mvp[1, 1] and not mvp[0, 2]
               and mvp[0, 0] and mvp[2, 3])
    check("synthetic invalid-pixel poisoning (exact false indices)",
          poison_ok and spot_ok,
          f"invalid pixel (1,2) -> expected mask False exactly at "
          f"(0,2),(1,1),(1,2) plus last row/col; full-mask equality: {poison_ok}; "
          f"spot (000 vs 111/102 False, 000/233 True): {spot_ok}; "
          f"got mask=\n{mvp.astype(int)}")

    # ---- 6. disparity twin: same operator + C2-style mask ----
    magd, mvd = disparity_discontinuity(fv, None)  # all finite -> all valid
    twin_ok = (bool((mvd == exp_mask).all())
               and bool((magd[:-1, K - 1] == STEP).all())
               and bool((magd[:-1, :K - 1] == 0.0).all()))
    dd = np.full((4, 5), 20.0)
    dd[0, 0] = 0.0
    _, mvd2 = disparity_discontinuity(dd, valid=dd > 1e-3)
    # Hand-derived: valid[0,0]=False poisons du_ok[0,0] and dv_ok[0,0],
    # so mask[0,0] is False; (1,1) is untouched -> True; last row/col False.
    c2mask_ok = (not mvd2[0, 0]) and mvd2[1, 1] and not mvd2[-1, :].any() \
        and not mvd2[:, -1].any()
    check("disparity twin (same operator, incl. C2-style d>1e-3 mask)",
          twin_ok and c2mask_ok,
          f"step case identical to depth twin: {twin_ok} "
          f"(mag[0,4]={magd[0, K - 1]}); C2-mask d[0,0]=0 excluded -> "
          f"mask[0,0]=False ({not mvd2[0, 0]}), mask[1,1]=True ({mvd2[1, 1]}), "
          f"borders False: {bool(not mvd2[-1, :].any() and not mvd2[:, -1].any())}")

    # ---- 7. real KITTI sanity (SANITY, never accuracy) ----
    import armp_depth as AD
    from src.datasets.kitti2015 import Kitti2015Stereo
    from src.geometry.stereo import parse_kitti_cam_to_cam
    from metric_depth import disparity_to_depth

    net, device = AD.load_frozen_net()
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert REAL_SCENE in list(ds.names), f"{REAL_SCENE} not in hailo_val"
    disp = AD.infer_disparity_scene(net, device, REAL_SCENE)
    calib = parse_kitti_cam_to_cam(ds.calibration_path(REAL_SCENE))
    depth, valid = disparity_to_depth(calib, disp)

    # C2 depth unchanged on this scene (independent fB/d recomputation).
    fb = calib.focal_px * calib.baseline_m
    indep = np.full_like(depth, np.nan)
    ok_pix = np.isfinite(disp) & (disp > 1e-3)
    indep[ok_pix] = fb / disp[ok_pix]
    same_nan = bool((np.isnan(depth) == np.isnan(indep)).all())
    vv = valid & ok_pix & np.isfinite(depth) & np.isfinite(indep)
    maxdiff = float(np.abs(depth[vv] - indep[vv]).max()) if vv.any() else 0.0
    check("C2 output unchanged on sanity scene (max|depth - fB/d| == 0)",
          same_nan and maxdiff == 0.0,
          f"scene {REAL_SCENE}: max|C2-depth minus fB/d|={maxdiff:.3e} "
          f"NaN-pattern-identical={same_nan}")

    depth_before = depth.copy()
    disp_before = disp.copy()
    mag_r, mask_r = depth_discontinuity(depth, valid)
    smag_r, smask_r = disparity_discontinuity(disp, valid)
    inputs_untouched = (bool(np.array_equal(depth, depth_before, equal_nan=True))
                        and bool(np.array_equal(disp, disp_before, equal_nan=True)))
    check("real KITTI sanity: inputs not modified (bitwise-identical)",
          inputs_untouched,
          f"depth and disparity arrays bitwise-identical after call: "
          f"{inputs_untouched}")
    shape_ok = mag_r.shape == depth.shape and mask_r.shape == depth.shape
    finite_ok = bool(np.isfinite(mag_r[mask_r]).all())
    check("real KITTI sanity: shape == input, finite where mask True",
          shape_ok and finite_ok,
          f"scene {REAL_SCENE}: in={depth.shape} out={mag_r.shape} "
          f"shape_ok={shape_ok}; finite-where-valid={finite_ok} "
          f"(valid={int(mask_r.sum())}/{mask_r.size})")
    mag_r2, mask_r2 = depth_discontinuity(depth, valid)
    det_ok = (bool(np.array_equal(mag_r, mag_r2, equal_nan=True))
              and bool((mask_r == mask_r2).all()))
    check("real KITTI sanity: deterministic across repeats (bitwise-identical)",
          det_ok,
          f"two calls bitwise-identical (equal_nan): mag={np.array_equal(mag_r, mag_r2, equal_nan=True)} "
          f"mask={(mask_r == mask_r2).all()}")
    n_valid = int(mask_r.sum())
    n_above = int((mag_r[mask_r] > MAG_DESCRIBE_THRESHOLD_M_PER_PX).sum())
    frac = (n_above / n_valid) if n_valid else float("nan")
    nonempty = n_valid > 0 and n_above > 0
    check("real KITTI sanity: map non-empty where real transitions exist "
          "(descriptive, not a threshold)",
          nonempty,
          f"DESCRIPTIVE (not correctness): fraction of valid pixels with "
          f"mag>{MAG_DESCRIBE_THRESHOLD_M_PER_PX} m/px = {n_above}/{n_valid} = {frac:.6f}; "
          f"mag[valid] min={float(mag_r[mask_r].min()):.6f} "
          f"max={float(mag_r[mask_r].max()):.6f} mean={float(mag_r[mask_r].mean()):.6f}")

    # ---- 8. resource: discontinuity operation ALONE ----
    t = time.perf_counter()
    for _ in range(TIMING_REPS):
        depth_discontinuity(depth, valid)
    disc_s = (time.perf_counter() - t) / TIMING_REPS
    tracemalloc.start()
    depth_discontinuity(depth, valid)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out_bytes = int(mag_r.nbytes + mask_r.nbytes)  # float64 HxW + bool HxW
    check("resource: discontinuity-only runtime and memory measured",
          disc_s > 0 and out_bytes > 0,
          f"DEVELOPMENT-MACHINE MEASUREMENT: {disc_s * 1000:.3f} ms/frame "
          f"over {TIMING_REPS} reps (scene {REAL_SCENE}, {depth.shape}); "
          f"tracemalloc peak for one call={peak} bytes; "
          f"output buffers={out_bytes} bytes "
          f"(mag {mag_r.nbytes} + mask {mask_r.nbytes})")

    # ---- artifact integrity AFTER; prior artifacts preserved ----
    disc_hash_after = sha256_file(REPO / DISC_REL)
    ckpt_hash_after = sha256_file(
        REPO / "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth")
    onnx_hash_after = sha256_file(REPO / ONNX_REL)
    check("frozen discontinuity.py unchanged (after == before)",
          disc_hash_after == disc_hash_before,
          f"before={disc_hash_before} after={disc_hash_after}")
    check("frozen checkpoint unchanged (after)",
          ckpt_hash_after == CKPT_SHA, f"sha256={ckpt_hash_after}")
    check("ONNX hash matches disk spelling (after)",
          onnx_hash_after == ONNX_SHA_DISK,
          f"sha256({ONNX_REL})={onnx_hash_after}")
    c2_1_json = REPO / "stage_c_deploy" / "spatial_perception" / "out" / "c2_1_validation.json"
    pngs = sorted((REPO / "stage_c_deploy" / "spatial_perception" / "out").glob("*_c2_1_spatial.png"))
    check("prior C2/C2.1 artifacts preserved (not overwritten)",
          c2_1_json.exists() and len(pngs) >= 2,
          f"c2_1_validation.json exists={c2_1_json.exists()}; "
          f"c2_1 PNGs={[p.name for p in pngs]}")

    rec = {"utc": datetime.now(timezone.utc).isoformat(),
           "task": "C2.1.1 depth-discontinuity validation gate (evidence only)",
           "algorithm_frozen": True,
           "discontinuity_sha256_before": disc_hash_before,
           "discontinuity_sha256_after": disc_hash_after,
           "checkpoint_sha256_before": ckpt_hash_before,
           "checkpoint_sha256_after": ckpt_hash_after,
           "onnx_sha256_before": onnx_hash_before,
           "onnx_sha256_after": onnx_hash_after,
           "onnx_prompt_typo": ONNX_SHA_PROMPT_TYPO,
           "onnx_note": "prompt carries transposed typo ...deed1dce26...; "
                        "disk ...deed1cde26... authoritative; fix nothing",
           "real_scene": REAL_SCENE,
           "real_scene_stats": {
               "shape": list(depth.shape),
               "num_valid_depth": int(valid.sum()),
               "num_valid_mag": n_valid,
               "num_above_threshold": n_above,
               "fraction_above_threshold": frac,
               "threshold_m_per_px": MAG_DESCRIBE_THRESHOLD_M_PER_PX,
               "mag_valid_min": float(mag_r[mask_r].min()),
               "mag_valid_max": float(mag_r[mask_r].max()),
               "mag_valid_mean": float(mag_r[mask_r].mean()),
               "c2_maxdiff": maxdiff},
           "resource": {
               "discontinuity_only_s_per_frame": disc_s,
               "discontinuity_tracemalloc_peak_bytes": int(peak),
               "discontinuity_output_bytes": out_bytes,
               "reps": TIMING_REPS,
               "note": "DEVELOPMENT-MACHINE MEASUREMENT, never Hailo performance; "
                       "discontinuity op alone, excl. inference and rest of spatial layer"},
           "expected_corner_hypot_4_9": exp_corner,
           "tests": tests, "failures": failures,
           "elapsed_s": time.time() - t0,
           "overall": "PASS" if not failures else "FAIL"}
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"C2.1.1 discontinuity validation: "
          f"{len(tests) - len(failures)}/{len(tests)} passed; "
          f"overall={rec['overall']}; wrote {OUT_JSON}")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
