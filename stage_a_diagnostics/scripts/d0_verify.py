"""STAGE-A D0: integrity + baseline reproduction (frozen model, read-only).

Read-only mirror of phase2/scripts/eval_p2a.py scoring path: same dataset
split, GT scale, valid mask, pooled metrics and contract guard imported
unmodified from phase1.harness.frozen_eval; same StereoNetConfig
(downsample_levels=3, num_disparities=24, cost_volume_shift='right',
regression_normalize=True). NEVER writes under phase2/.

Outputs (atomic write via .tmp + rename):
  stage_a_diagnostics/integrity.json
  stage_a_diagnostics/baseline.json
  stage_a_diagnostics/raw/d0_preds_s{0,1,2}.npz  (P,G float64 valid-px vectors)

Failure handling: raises on any integrity failure or silent-fallback risk.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from phase1.harness.frozen_eval import (
    pooled_metrics,
    refuse_unless_contract,
    sha256_file,
    strict_load_report,
)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_a_diagnostics"
RAW = OUT / "raw"
RUNS = {0: "p2a_scale_coverage", 1: "p2a_scale_coverage_s1", 2: "p2a_scale_coverage_s2"}
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
EXPECTED = {
    "params": 397954,
    "n_keys": 70,
    "split": "hailo_val",
    "scale": 256.0,
    "occluded": True,
    "valid_pixels": 3802797,
    "scenes": 40,
    "seed0_epe": 1.4149796,   # truncated display of 1.4149795736625577
    "mean_epe": 1.4409826,
    "tol_px": 1e-3,           # established project tolerance (export parity bar)
}
REF_EPE = 1.3134471


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def check(name: str, measured, expected, ok: bool, note: str = "") -> dict:
    return {"check": name, "measured": measured, "expected": expected,
            "pass": bool(ok), "note": note}


def main() -> None:
    integrity, fails = [], []

    def record(name, measured, expected, ok, note=""):
        row = check(name, measured, expected, ok, note)
        integrity.append(row)
        if not ok:
            fails.append(name)
        print(("PASS " if ok else "FAIL ") + name + f" | measured={measured} expected={expected} {note}", flush=True)

    # ---- paths exist ----
    for seed, run in RUNS.items():
        ck = REPO / "phase2" / "runs" / run / "p2a_best.pth"
        rec_p = REPO / "phase2" / "runs" / run / "p2a_record.json"
        for p in (ck, rec_p):
            if not p.exists():
                raise FileNotFoundError(f"D0 path missing: {p}")

    # ---- checkpoint identity vs p2a_record.json + p2a_eval_best.json ----
    eval_best = json.loads((REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_eval_best.json").read_text())
    for seed, run in RUNS.items():
        ck = REPO / "phase2" / "runs" / run / "p2a_best.pth"
        rec = json.loads((REPO / "phase2" / "runs" / run / "p2a_record.json").read_text())
        sha = sha256_file(ck)
        record(f"seed{seed}_sha256_matches_p2a_record", sha, rec["best_sha256"],
               sha == rec["best_sha256"])
        esha = eval_best["seeds"][str(seed)]["sha256"]
        record(f"seed{seed}_sha256_matches_p2a_eval_best", sha, esha, sha == esha)

    # ---- load seed0, structural checks ----
    blob0 = torch.load(REPO / "phase2" / "runs" / RUNS[0] / "p2a_best.pth",
                       map_location="cpu", weights_only=False)
    if not isinstance(blob0, dict) or "model" not in blob0:
        raise ValueError("seed0 checkpoint missing ['model'] key: silent-fallback risk")
    state0 = blob0["model"]
    model = StereoNet(StereoNetConfig(**CFG))
    n_params = sum(p.numel() for p in model.parameters())
    record("param_count", n_params, EXPECTED["params"], n_params == EXPECTED["params"])
    record("config_downsample_levels", model.config.downsample_levels, 3,
           model.config.downsample_levels == 3)
    record("config_num_disparities", model.config.num_disparities, 24,
           model.config.num_disparities == 24)
    record("config_cost_volume_shift", model.config.cost_volume_shift, "right",
           model.config.cost_volume_shift == "right")
    record("config_regression_normalize", model.config.regression_normalize, True,
           model.config.regression_normalize is True)
    bn = [n for n, m in model.named_modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]
    record("no_batchnorm_modules", bn, [], len(bn) == 0)
    record("cost_hypotheses", model.cost_volume.num_disparities, 24,
           model.cost_volume.num_disparities == 24)
    record("disparity_spacing_px", model.config.feature_stride, 8,
           model.config.feature_stride == 8)
    record("max_disparity_px", model.config.max_disparity_px, 184,
           model.config.max_disparity_px == 184)
    has_ref = model.refinement is not None
    record("refinement_module_present", has_ref, True, has_ref is True)

    # strict load, all 3 seeds
    for seed, run in RUNS.items():
        st = torch.load(REPO / "phase2" / "runs" / run / "p2a_best.pth",
                        map_location="cpu", weights_only=False)["model"]
        m = StereoNet(StereoNetConfig(**CFG))
        compat = strict_load_report(m, st)
        ok = (not compat["missing"] and not compat["unexpected"] and compat["strict_ok"])
        record(f"seed{seed}_strict_load_zero_missing_unexpected",
               {"missing": compat["missing"], "unexpected": compat["unexpected"],
                "matched": compat["matched"]},
               {"missing": [], "unexpected": [], "matched": 70}, ok)
        record(f"seed{seed}_n_keys", compat["matched"], 70, compat["matched"] == 70)

    # ---- refinement enabled: residual must be non-trivial on real input ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    record("eval_device", device, "cuda(preferred)/cpu", True, "informational; recorded")
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    if len(ds) != EXPECTED["scenes"]:
        raise ValueError(f"scene count {len(ds)} != 40: wrong split?")
    record("eval_split_scenes", len(ds), 40, len(ds) == 40, "split=hailo_val")
    record("disparity_scale", 256.0, 256.0, True, "dataset constructed with scale 256.0")
    record("occluded_gt", "disp_occ_0", "disp_occ_0", True, "occluded=True")
    s0 = ds[0]
    if not (s0.disparity > 0).any():
        raise ValueError("valid-pixel count is 0 on first scene")
    m0 = StereoNet(StereoNetConfig(**CFG))
    m0.load_state_dict(state0, strict=True)
    m0.eval().to(device)
    with torch.no_grad():
        out, stages = m0(torch.from_numpy(normalize(s0.left)).to(device),
                         torch.from_numpy(normalize(s0.right)).to(device),
                         return_stages=True)
    resid = stages["refinement_residual"].detach().cpu().numpy()
    rmag = float(np.abs(resid).mean())
    record("refinement_residual_mean_abs_px", rmag, ">0", rmag > 0,
           "refinement contributes; enabled")
    if not np.isfinite(resid).all():
        raise ValueError("NaN/inf in refinement residual")
    del m0

    # ---- frozen re-evaluation, all 3 seeds (read-only mirror) ----
    per_seed, D1 = {}, {}
    for seed, run in RUNS.items():
        st = torch.load(REPO / "phase2" / "runs" / run / "p2a_best.pth",
                        map_location="cpu", weights_only=False)["model"]
        m = StereoNet(StereoNetConfig(**CFG))
        m.load_state_dict(st, strict=True)
        m.eval().to(device)
        preds, gts = [], []
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                if s.disparity.shape != (368, 1232):
                    raise ValueError(f"shape mismatch scene {s.name}: {s.disparity.shape}")
                o = m(torch.from_numpy(normalize(s.left)).to(device),
                      torch.from_numpy(normalize(s.right)).to(device))
                pred = o[0, 0].cpu().numpy().astype(np.float64)
                if not np.isfinite(pred).all():
                    raise ValueError(f"NaN/inf prediction seed {seed} scene {s.name}")
                valid = s.disparity > 0
                if valid.sum() == 0:
                    raise ValueError(f"empty valid mask seed {seed} scene {s.name}")
                preds.append(pred[valid])
                gts.append(s.disparity[valid].astype(np.float64))
        P, G = np.concatenate(preds), np.concatenate(gts)
        if not np.isfinite(P).all() or not np.isfinite(G).all():
            raise ValueError(f"NaN/inf in pooled vectors seed {seed}")
        if (G <= 0).any():
            raise ValueError(f"non-positive GT in valid mask seed {seed}")
        met = pooled_metrics(preds, gts)
        guard = refuse_unless_contract(len(ds), met["valid_pixels"], 256.0,
                                      "hailo_val", "disp_occ_0")
        per_seed[seed] = {"epe": float(met["epe"]), "d1": float(met["d1"]),
                          "rmse": float(met["rmse"]), "valid_pixels": int(met["valid_pixels"])}
        D1[seed] = (P, G)
        record(f"seed{seed}_valid_pixels", int(met["valid_pixels"]),
               EXPECTED["valid_pixels"], int(met["valid_pixels"]) == EXPECTED["valid_pixels"])
        record(f"seed{seed}_contract_match", guard["contract_match"], True,
               guard["contract_match"] is True)
        # atomic raw dump
        rp = RAW / f"d0_preds_s{seed}.npz"
        fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez_compressed(fh, P=P, G=G)
            os.replace(tmp, rp)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print(f"seed {seed} EPE {met['epe']:.7f} D1 {met['d1']:.4f}% valid {met['valid_pixels']}", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    e0 = per_seed[0]["epe"]
    mean_epe = float(np.mean([per_seed[s]["epe"] for s in (0, 1, 2)]))
    mean_d1 = float(np.mean([per_seed[s]["d1"] for s in (0, 1, 2)]))
    record("seed0_epe_reproduces", e0, EXPECTED["seed0_epe"],
           abs(e0 - 1.4149795736625577) < EXPECTED["tol_px"],
           f"diff={e0 - 1.4149795736625577:.2e} tol=1e-3")
    record("mean3_epe_reproduces", mean_epe, EXPECTED["mean_epe"],
           abs(mean_epe - 1.4409825930775115) < EXPECTED["tol_px"],
           f"diff={mean_epe - 1.4409825930775115:.2e} tol=1e-3")

    baseline = {
        "expected": {"seed0_epe": 1.4149795736625577, "mean_epe": 1.4409825930775115,
                     "tolerance_px": EXPECTED["tol_px"]},
        "reproduced": {str(s): per_seed[s] for s in (0, 1, 2)},
        "method_mean": {"epe": mean_epe, "d1": mean_d1},
        "reference_epe": REF_EPE,
        "total_gap_px": mean_epe - REF_EPE,
        "device": device,
        "gate": "PASS" if not fails else "FAIL",
    }
    atomic_write_json(OUT / "baseline.json", baseline)
    atomic_write_json(OUT / "integrity.json",
                      {"gate": "PASS" if not fails else "FAIL",
                       "failed_checks": fails, "checks": integrity})
    print("GATE: " + ("PASS" if not fails else "FAIL") + f" fails={fails}", flush=True)
    if fails:
        raise SystemExit(f"D0 GATE FAILED: {fails}")


if __name__ == "__main__":
    main()
