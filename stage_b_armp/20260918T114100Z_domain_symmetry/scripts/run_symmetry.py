"""Domain-symmetry control: FROZEN INFERENCE ONLY. NO TRAINING.

CELL 1: P2A seed-0 weights through the EXACT validate() path imported from
  stage_b_armp/20260918T062146Z_stage1_pretrain/scripts/train_armp_stage1.py
  (not copied, not altered), on the SAME FT3D pretrain-val holdout slice
  (same manifest, seed-0 split) with the SAME limit (VAL_SCENES) and SAME
  mask (gt>0 AND gt<184).
CELL 2: ARM-P stage1-best weights on KITTI hailo_val under the frozen
  evaluation contract, using a line-for-line mirror of the scoring in
  phase2/scripts/eval_p2a.py (not modified).

No repo file outside the output dir is created or modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
STAGE1_SCRIPTS = REPO / "stage_b_armp" / "20260918T062146Z_stage1_pretrain" / "scripts"
sys.path.insert(0, str(STAGE1_SCRIPTS))

from phase1.harness.frozen_eval import (  # noqa: E402
    pooled_metrics,
    refuse_unless_contract,
    sha256_file,
)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

import train_armp_stage1 as T1  # noqa: E402 (imported, not copied, not altered)
from ft3d import FlyingThings3DTrain  # noqa: E402

P2A_CKPT = REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_best.pth"
ARMP_CKPT = (REPO / "stage_b_armp" / "20260918T062146Z_stage1_pretrain"
             / "checkpoints" / "armp_stage1_best.pth")
MANIFEST = (REPO / "stage_b_armp" / "20260918T062146Z_stage1_pretrain"
            / "dataset_manifest.json")
D3 = REPO / "stage_a_diagnostics" / "scripts" / "d3_matching.py"

BINS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
MIN_BIN = 1000


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def assert_p2a_arch(state: dict, label: str) -> dict:
    cfg = StereoNetConfig(**T1.P2A_CONFIG)
    assert cfg.downsample_levels == 3, label
    assert cfg.num_disparities == 24, label
    assert cfg.cost_volume_shift == "right", label
    assert cfg.regression_normalize is True, label
    model = StereoNet(cfg)
    params = sum(p.numel() for p in model.parameters())
    assert params == 397954, (label, params)
    assert T1.P2A_PARAMS == 397954, label
    want = set(model.state_dict().keys())
    got = set(state.keys())
    missing = sorted(want - got)
    unexpected = sorted(got - want)
    model.load_state_dict(state, strict=True)  # raises on mismatch: abort
    assert not missing and not unexpected, (label, missing, unexpected)
    return {"config": dict(T1.P2A_CONFIG), "params": params,
            "strict_load_ok": True, "n_keys": len(want)}


def score_kitti_hailo_val(model, device: str) -> dict:
    """Line-for-line mirror of phase2/scripts/eval_p2a.py score(model, 'hailo_val', device)."""
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    preds, gts, names = [], [], []
    pmax_all = -1e30
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = model(torch.from_numpy(normalize(s.left)).to(device),
                        torch.from_numpy(normalize(s.right)).to(device))
            pred = out[0, 0].cpu().numpy().astype(np.float64)
            pmax_all = max(pmax_all, float(pred.max()))
            valid = s.disparity > 0
            preds.append(pred[valid])
            gts.append(s.disparity[valid].astype(np.float64))
            names.append(s.name)
    P, G = np.concatenate(preds), np.concatenate(gts)
    m = pooled_metrics(preds, gts)
    err = np.abs(P - G)

    def stratum(mask):
        if mask.sum() < MIN_BIN:
            return {"px": int(mask.sum()), "status": "INSUFFICIENT"}
        a, b = np.polyfit(G[mask], P[mask], 1)
        yh = a * G[mask] + b
        ss_res = float(((P[mask] - yh) ** 2).sum())
        ss_tot = float(((P[mask] - P[mask].mean()) ** 2).sum())
        return {"px": int(mask.sum()), "frac_px": float(mask.mean()),
                "epe": float(err[mask].mean()),
                "signed_err": float((P[mask] - G[mask]).mean()),
                "mean_gt": float(G[mask].mean()), "mean_pred": float(P[mask].mean()),
                "slope": float(a), "intercept": float(b),
                "r2": 1.0 - ss_res / ss_tot if ss_tot else None,
                "pearson_r": float(np.corrcoef(G[mask], P[mask])[0, 1])}

    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mm = (G >= lo) & (G < hi)
        row = {"lo": lo, "hi": hi, "px": int(mm.sum()),
               "frac_px": float(mm.mean())}
        if mm.sum() >= MIN_BIN:
            row.update({"epe": float(err[mm].mean()),
                        "signed_err": float((P[mm] - G[mm]).mean()),
                        "rmse": float(np.sqrt(((P[mm] - G[mm]) ** 2).mean())),
                        "d1_pct": float((((err[mm] > 3) & (err[mm] > 0.05 * G[mm])).mean()) * 100),
                        "mean_gt": float(G[mm].mean()), "mean_pred": float(P[mm].mean())})
        else:
            row["status"] = "INSUFFICIENT (<%d px)" % MIN_BIN
        bins.append(row)

    guard = refuse_unless_contract(len(ds), m["valid_pixels"], 256.0, "hailo_val",
                                   "disp_occ_0")
    return {"split": "hailo_val", "scenes": len(ds), "names": names,
            "metrics": m, "guard": guard,
            "gt_lt_64": stratum(G < 64), "gt_ge_64": stratum(G >= 64),
            "gt_ge_96": stratum(G >= 96), "bins": bins,
            "max_pred_valid_px": float(P.max()), "max_pred_all_px": pmax_all,
            "max_gt": float(G.max())}


def main() -> None:
    t0 = time.time()
    (OUT_DIR / "raw").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
    except Exception:
        git_head = None

    d3_md5_before = md5_file(D3)
    print("d3_matching.py md5 BEFORE: " + d3_md5_before, flush=True)
    assert d3_md5_before == "ab29a921c121f5e404014989532172a6", d3_md5_before

    p2a_sha256 = sha256_file(P2A_CKPT)
    armp_sha256 = sha256_file(ARMP_CKPT)
    print("p2a sha256:  " + p2a_sha256, flush=True)
    print("armp sha256: " + armp_sha256, flush=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["rng_seed"] == 0, manifest["rng_seed"]
    assert manifest["n_holdout"] == 723, manifest["n_holdout"]
    holdout_ids = list(manifest["holdout_ids"])
    assert len(holdout_ids) == 723

    # ---------------- CELL 1: P2A seed 0 on FT3D pretrain-val ----------------
    val_base = FlyingThings3DTrain(MANIFEST, split="pretrain_val")
    assert len(val_base) == 723, len(val_base)
    assert list(val_base.ids) == holdout_ids, "holdout ordering differs from manifest"
    limit = int(T1.VAL_SCENES)
    assert limit == 10, limit
    assert float(T1.MAX_DISP) == 184.0, T1.MAX_DISP

    p2a_blob = torch.load(str(P2A_CKPT), map_location="cpu", weights_only=False)
    p2a_state = p2a_blob["model"]
    p2a_arch = assert_p2a_arch(p2a_state, "p2a")
    p2a_model = StereoNet(StereoNetConfig(**T1.P2A_CONFIG))
    p2a_model.load_state_dict(p2a_state, strict=True)
    p2a_model.eval().to(device)
    with torch.no_grad():
        m1 = T1.validate(p2a_model, val_base, device, limit=limit)
    p2a_model.eval()  # validate() leaves model in train(); restore eval (no training occurred)
    cell1_ids = list(val_base.ids[:limit])
    slice_match = (cell1_ids == holdout_ids[:limit])
    assert slice_match, "FT3D slice is NOT identical to the original first-10 holdout slice"
    print("CELL1 slice match: True (first %d holdout ids)" % limit, flush=True)
    print("CELL1 ids: " + json.dumps(cell1_ids), flush=True)
    cell1 = {"model": "P2A seed 0 (phase2/runs/p2a_scale_coverage/p2a_best.pth)",
             "sha256": p2a_sha256, "arch": p2a_arch,
             "path": "EXACT imported validate() from train_armp_stage1.py (not copied, not altered)",
             "dataset": "FT3D pretrain-val holdout, rng seed 0, n_holdout=723",
             "limit": limit, "mask": "gt>0 AND gt<184",
             "scene_ids": cell1_ids,
             "slice_identical_to_original": slice_match,
             "metrics": {"epe": float(m1.epe), "d1": float(m1.d1),
                         "rmse": float(m1.rmse), "bad1": float(m1.bad1),
                         "bad2": float(m1.bad2), "bad3": float(m1.bad3),
                         "valid_pixels": int(m1.valid_pixels)}}
    print("CELL1 P2A-on-FT3D: EPE %.7f D1 %.4f%% px %d" %
          (m1.epe, m1.d1, m1.valid_pixels), flush=True)
    write_atomic(OUT_DIR / "raw" / "cell1_p2a_ft3d.json",
                 json.dumps(cell1, indent=2))
    del p2a_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------------- CELL 2: ARM-P stage1-best zero-shot on KITTI ----------------
    armp_blob = torch.load(str(ARMP_CKPT), map_location="cpu", weights_only=False)
    armp_state = armp_blob["model"]
    armp_arch = assert_p2a_arch(armp_state, "armp")
    armp_model = StereoNet(StereoNetConfig(**T1.P2A_CONFIG))
    armp_model.load_state_dict(armp_state, strict=True)
    armp_model.eval().to(device)
    with torch.no_grad():
        s2 = score_kitti_hailo_val(armp_model, device)
    cell2 = {"model": "ARM-P stage1 best (zero-shot, never saw KITTI)",
             "checkpoint": ("stage_b_armp/20260918T062146Z_stage1_pretrain/"
                            "checkpoints/armp_stage1_best.pth"),
             "sha256": armp_sha256, "arch": armp_arch,
             "path": ("line-for-line mirror of phase2/scripts/eval_p2a.py score() "
                      "for split='hailo_val' (eval_p2a.py NOT modified); dataset "
                      "split, GT scale, valid mask, pooled metrics, contract "
                      "guard imported unmodified from phase1.harness.frozen_eval"),
             "zero_shot_label": True,
             "score": s2}
    v = s2["metrics"]
    print("CELL2 ARM-P-zero-shot-on-KITTI: EPE %.7f D1 %.4f%% px %d contract %s" %
          (v["epe"], v["d1"], v["valid_pixels"],
           s2["guard"]["contract_match"]), flush=True)
    print("CELL2 GT>=96: " + json.dumps(s2["gt_ge_96"]), flush=True)
    for b in s2["bins"]:
        print("CELL2 bin %d-%d: %s" % (b["lo"], b["hi"], json.dumps(b)),
              flush=True)
    write_atomic(OUT_DIR / "raw" / "cell2_armp_kitti.json",
                 json.dumps(cell2, indent=2))
    del armp_model
    if device == "cuda":
        torch.cuda.empty_cache()

    d3_md5_after = md5_file(D3)
    print("d3_matching.py md5 AFTER:  " + d3_md5_after, flush=True)
    assert d3_md5_after == d3_md5_before == "ab29a921c121f5e404014989532172a6"

    # ---------------- penalties + classification ----------------
    kitti_p2a = 1.4149796
    ft3d_armp = 3.0486
    c1, c2 = float(m1.epe), float(v["epe"])
    pen_p2a = {"from_in_domain_kitti": kitti_p2a, "to_out_of_domain_ft3d": c1,
               "difference": c1 - kitti_p2a, "ratio": c1 / kitti_p2a}
    pen_armp = {"from_in_domain_ft3d": ft3d_armp, "to_out_of_domain_kitti": c2,
                "difference": c2 - ft3d_armp, "ratio": c2 / ft3d_armp}
    p2a_degrades = bool((c1 - kitti_p2a) > 1.0 and (c1 / kitti_p2a) > 1.5)
    if (c1 - kitti_p2a) > 1.0 and (c1 / kitti_p2a) > 1.5:
        classification = "DOMAIN CONFOUND PLAUSIBLE"
        why = ("P2A also degrades substantially out of domain (KITTI %.4f -> "
               "FT3D %.4f, +%.4f, x%.3f); the earlier ARM-P D3 negative "
               "cannot discriminate between a weak frozen representation "
               "and a pure domain shift." % (kitti_p2a, c1, c1 - kitti_p2a,
                                             c1 / kitti_p2a))
    elif (c2 / ft3d_armp) > 1.5 * (c1 / kitti_p2a) and c1 / kitti_p2a < 1.5:
        classification = "DOMAIN ASYMMETRY DETECTED"
        why = ("P2A transfers materially better (x%.3f) than ARM-P (x%.3f); "
               "evidence against ARM-P's frozen representation." %
               (c1 / kitti_p2a, c2 / ft3d_armp))
    else:
        classification = "CONTROL INCONCLUSIVE"
        why = ("Conventions could not be matched or the directional penalties "
               "do not separate cleanly; result uninterpretable.")

    wall_s = time.time() - t0
    control = {
        "experiment": "domain-symmetry control (frozen inference only, no training)",
        "table_2x2": {
            "columns": ["KITTI hailo_val EPE", "FT3D pretrain-val EPE"],
            "rows": {
                "P2A seed 0": {"kitti_hailo_val_epe": kitti_p2a,
                               "ft3d_pretrain_val_epe": c1,
                               "ft3d_pretrain_val_d1": float(m1.d1)},
                "ARM-P stage1 best": {"kitti_hailo_val_epe_zero_shot": c2,
                                      "kitti_hailo_val_d1_zero_shot": float(v["d1"]),
                                      "ft3d_pretrain_val_epe": ft3d_armp},
            },
            "known_cells": {"p2a_kitti": kitti_p2a,
                            "source": "phase2/runs/p2a_scale_coverage/p2a_eval_best.json seed 0",
                            "armp_ft3d": ft3d_armp,
                            "armp_source": ("stage_b_armp/20260918T062146Z_stage1_pretrain/"
                                            "pretrain_log.jsonl epoch 5 best-by-pretrain-val, "
                                            "first 10 pretrain-val triplets, mask gt>0 AND gt<184")},
            "measured_cells": {"cell1_p2a_on_ft3d_epe": c1,
                               "cell2_armp_on_kitti_zero_shot_epe": c2},
        },
        "transfer_penalty": {"p2a_kitti_to_ft3d": pen_p2a,
                             "armp_ft3d_to_kitti": pen_armp},
        "p2a_also_degrades_substantially": p2a_degrades,
        "penalties_comparable_character": None,
        "domain_note": ("Directional, asymmetric control: the two penalties are NOT "
                        "paired measurements from the same domain; domains, image "
                        "statistics and GT densities differ."),
        "classification": classification,
        "classification_why": why,
        "not_claimed": ("ARM-P is neither validated nor falsified; nothing is claimed "
                        "about EPE after fine-tuning (not tested). The earlier ARM-P "
                        "D3 result keeps its status: OBSERVED DEGRADATION / MECHANISM "
                        "NOT SUPPORTED / CAUSAL CONFIDENCE LOW / DOMAIN-CONFOUNDED / "
                        "REASSESS."),
        "integrity": {
            "d3_matching_py_md5_before": d3_md5_before,
            "d3_matching_py_md5_after": d3_md5_after,
            "d3_matching_py_unchanged": d3_md5_before == d3_md5_after,
            "d3_matching_py_expected": "ab29a921c121f5e404014989532172a6",
            "p2a_checkpoint_sha256": p2a_sha256,
            "armp_checkpoint_sha256": armp_sha256,
            "p2a_arch": p2a_arch,
            "armp_arch": armp_arch,
            "ft3d_slice_identical_to_original": slice_match,
            "ft3d_slice_ids": cell1_ids,
        },
        "runtime": {"wall_clock_s": wall_s, "device": device, "gpu": gpu,
                    "torch": torch.__version__, "numpy": np.__version__,
                    "python": sys.version.split()[0], "git_head": git_head},
    }
    c1r, c2r = c1 / kitti_p2a, c2 / ft3d_armp
    control["penalties_comparable_character"] = (
        "P2A x%.3f (+%.4f) vs ARM-P x%.3f (+%.4f): %s" %
        (c1r, c1 - kitti_p2a, c2r, c2 - ft3d_armp,
         "comparable degradation character" if p2a_degrades and c2r > 1.5 else "see classification"))
    write_atomic(OUT_DIR / "symmetry_control.json",
                 json.dumps(control, indent=2))
    print("classification: " + classification, flush=True)
    print("why: " + why, flush=True)
    print("wall %.1fs device %s gpu %s" % (wall_s, device, gpu), flush=True)
    print("wrote " + str(OUT_DIR / "symmetry_control.json"), flush=True)


if __name__ == "__main__":
    main()
