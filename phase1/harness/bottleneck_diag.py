"""Read-only bottleneck diagnostic for ARM-V. Trains nothing.

Runs the frozen-contract evaluation set (KITTI 2015, hailo_val scenes 160-199,
disp_occ_0, 368x1232 top-left crop, GT/256, valid = gt > 0, pooled) through an
ARM-V checkpoint with return_stages=True and decomposes the pooled error by:

  - error magnitude bins (where the EPE mass actually is)
  - GT disparity magnitude (incl. out-of-range gt > 184 px)
  - occlusion (disp_occ_0 valid AND disp_noc_0 == 0)
  - proximity to a GT depth discontinuity
  - left-image texture (gradient magnitude)
  - stage: initial soft-argmin readout vs refined output
  - disparity quantisation (soft-argmin index fractional part)
  - cost-volume discrimination (entropy, modality, hard-argmin vs GT)
  - an ORACLE probe: feed the refinement a GT-derived initial index

Nothing here is trained, tuned or written back into any arm's artifacts.
Usage: python phase1/harness/bottleneck_diag.py <run_dir> <ckpt_name> <out_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from phase1.harness.frozen_eval import refuse_unless_contract, sha256_file
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

SPACING = 8.0          # full-resolution px per disparity candidate (ARM-V)
NDISP = 24
MAXD = 184.0


def grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def gt_discontinuity(gt: np.ndarray, valid: np.ndarray, thresh: float = 3.0,
                     r: int = 2) -> np.ndarray:
    """True where a pixel sits within r px of a >thresh px jump in valid GT."""
    g = torch.from_numpy(np.where(valid, gt, np.nan))[None, None]
    k = 2 * r + 1
    mx = F.max_pool2d(torch.nan_to_num(g, nan=-1e9), k, 1, r)
    mn = -F.max_pool2d(-torch.nan_to_num(g, nan=1e9), k, 1, r)
    rng = (mx - mn)[0, 0].numpy()
    return (rng > thresh) & (rng < 1e8)


def main() -> None:
    run_dir = Path(sys.argv[1])
    ckpt_name = sys.argv[2]
    out = Path(sys.argv[3])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = REPO_ROOT / run_dir / ckpt_name
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=NDISP,
                                      cost_volume_shift="right",
                                      regression_normalize=True))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    ds_occ = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=True)
    ds_noc = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=False)

    keys = ("gt", "pred", "init", "resid", "err", "occluded", "disc", "tex",
            "cvargmin", "cvent", "cvmodes", "cvmargin", "oracle_err")
    acc = {k: [] for k in keys}
    for i in range(len(ds_occ)):
        s = ds_occ[i]
        gt = s.disparity.astype(np.float64)
        valid = gt > 0
        noc = ds_noc[i].disparity
        occluded = valid & (noc <= 0)
        disc = gt_discontinuity(gt.astype(np.float32), valid)
        tex = grad_mag(s.left.astype(np.float32).mean(2))

        left_t = torch.from_numpy(normalize(s.left)).to(device)
        right_t = torch.from_numpy(normalize(s.right)).to(device)
        with torch.no_grad():
            pred_t, st = model(left_t, right_t, return_stages=True)
            cost = st["aggregated_cost"]            # (1, 24, h/8, w/8)
            init = st["disparity_initial"]          # (1, 1, H, W), candidate units
            resid = st["refinement_residual"]
            cn = (cost - cost.mean(1, keepdim=True)) / (cost.std(1, keepdim=True) + 1e-6)
            p = torch.softmax(-cn, dim=1)
            ent = -(p * torch.log(p + 1e-12)).sum(1, keepdim=True)
            amin = cost.argmin(1, keepdim=True).float()
            srt = cost.sort(1).values
            margin = (srt[:, 1:2] - srt[:, 0:1]) / (cost.std(1, keepdim=True) + 1e-6)
            left_n = torch.cat([cost[:, :1], cost[:, :-1]], 1)
            right_n = torch.cat([cost[:, 1:], cost[:, -1:]], 1)
            modes = ((cost <= left_n) & (cost <= right_n)).sum(1, keepdim=True).float()

            def up(t):
                return F.interpolate(t, size=gt.shape, mode="nearest")

            ent, amin, margin, modes = up(ent), up(amin), up(margin), up(modes)

            vt = torch.from_numpy(valid).to(device)[None, None]
            gi = torch.from_numpy(np.where(valid, np.clip(gt, 0, MAXD) / SPACING,
                                           0.0)).to(device).float()[None, None]
            gi = torch.where(vt, gi, init)
            oracle = torch.relu(gi + model.refinement(gi, left_t))

        def nz(t):
            return t[0, 0].cpu().numpy().astype(np.float64)

        pred = nz(pred_t)
        err = np.abs(pred - gt)
        acc["gt"].append(gt[valid])
        acc["pred"].append(pred[valid])
        acc["init"].append(nz(init)[valid])
        acc["resid"].append(nz(resid)[valid])
        acc["err"].append(err[valid])
        acc["oracle_err"].append(np.abs(nz(oracle) - gt)[valid])
        acc["occluded"].append(occluded[valid])
        acc["disc"].append(disc[valid])
        acc["tex"].append(tex[valid].astype(np.float64))
        acc["cvargmin"].append(nz(amin)[valid])
        acc["cvent"].append(nz(ent)[valid])
        acc["cvmodes"].append(nz(modes)[valid])
        acc["cvmargin"].append(nz(margin)[valid])
        print(str(i + 1) + "/" + str(len(ds_occ)) + " " + s.name
              + " epe=" + format(err[valid].mean(), ".4f"), flush=True)

    A = {k: np.concatenate(v) for k, v in acc.items()}
    gt, err, init, resid = A["gt"], A["err"], A["init"], A["resid"]
    n_px = err.size
    tot = err.sum()
    res = {
        "checkpoint": (run_dir / ckpt_name).as_posix(),
        "sha256": sha256_file(ckpt),
        "guard": refuse_unless_contract(len(ds_occ), n_px, 256.0, "hailo_val",
                                        "disp_occ_0"),
        "valid_pixels": int(n_px),
        "epe": float(err.mean()),
        "d1": float((((err > 3) & (err > 0.05 * gt)).mean()) * 100),
    }

    edges = [0, 0.5, 1, 2, 3, 5, 10, 20, 50, 1e9]
    res["error_bins"] = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (err >= a) & (err < b)
        res["error_bins"].append({"lo": a, "hi": b, "frac_px": float(m.mean()),
                                  "epe_mass_frac": float(err[m].sum() / tot)})
    res["error_percentiles"] = {str(q): float(np.percentile(err, q))
                                for q in (50, 75, 90, 95, 99, 99.9)}

    gedges = [0, 8, 16, 32, 64, 96, 128, 160, 184, 1e9]
    res["gt_bins"] = []
    for a, b in zip(gedges[:-1], gedges[1:]):
        m = (gt >= a) & (gt < b)
        res["gt_bins"].append({"lo": a, "hi": b, "frac_px": float(m.mean()),
                               "epe": float(err[m].mean()) if m.any() else None,
                               "epe_mass_frac": float(err[m].sum() / tot)})
    res["gt_stats"] = {"mean": float(gt.mean()),
                       "p99": float(np.percentile(gt, 99)),
                       "max": float(gt.max()),
                       "frac_over_184": float((gt > MAXD).mean())}

    strata = (("occluded", A["occluded"].astype(bool)),
              ("discontinuity_r2_3px", A["disc"].astype(bool)),
              ("texture_low_q25", A["tex"] <= np.percentile(A["tex"], 25)),
              ("texture_high_q75", A["tex"] >= np.percentile(A["tex"], 75)))
    for name, m in strata:
        res[name] = {"px": int(m.sum()), "frac_px": float(m.mean()),
                     "epe": float(err[m].mean()) if m.any() else None,
                     "epe_mass_frac": float(err[m].sum() / tot),
                     "complement_epe": float(err[~m].mean())}

    a, b = np.polyfit(init, gt, 1)
    res["stage"] = {
        "init_units": "soft-argmin candidate index (0..23)",
        "init_mean": float(init.mean()), "init_std": float(init.std()),
        "init_min": float(init.min()), "init_max": float(init.max()),
        "corr_init_gt": float(np.corrcoef(init, gt)[0, 1]),
        "corr_resid_pred": float(np.corrcoef(resid, A["pred"])[0, 1]),
        "best_affine_a": float(a), "best_affine_b": float(b),
        "epe_of_best_affine_init": float(np.abs(a * init + b - gt).mean()),
        "epe_of_8x_init": float(np.abs(SPACING * init - gt).mean()),
        "resid_abs_mean": float(np.abs(resid).mean()),
        "pred_abs_mean": float(np.abs(A["pred"]).mean()),
        "resid_share_of_magnitude": float(np.abs(resid).mean()
                                          / (np.abs(A["pred"]).mean() + 1e-12)),
        "epe_final": float(err.mean()),
        "oracle_gt_init_epe": float(A["oracle_err"].mean()),
    }

    frac = init - np.floor(init)
    d2c = np.minimum(frac, 1 - frac)
    res["quantisation"] = {
        "frac_part_hist": np.histogram(frac, bins=10, range=(0, 1))[0].tolist(),
        "frac_part_uniform_expect": float(n_px / 10),
        "frac_near_integer_within_0p05": float((d2c < 0.05).mean()),
        "epe_vs_dist_to_candidate": [
            {"lo": float(lo), "hi": float(lo + 0.1),
             "epe": float(err[(d2c >= lo) & (d2c < lo + 0.1)].mean())}
            for lo in (0.0, 0.1, 0.2, 0.3, 0.4)],
        "affine_half_spacing_px": float(abs(a) * 0.5),
    }

    hard = SPACING * A["cvargmin"]
    res["cost_volume"] = {
        "entropy_mean": float(A["cvent"].mean()),
        "entropy_max_possible": float(np.log(NDISP)),
        "entropy_p10": float(np.percentile(A["cvent"], 10)),
        "entropy_p90": float(np.percentile(A["cvent"], 90)),
        "local_minima_mean": float(A["cvmodes"].mean()),
        "unimodal_frac": float((A["cvmodes"] <= 1).mean()),
        "runner_up_margin_mean_sd_units": float(A["cvmargin"].mean()),
        "argmin_index_mean": float(A["cvargmin"].mean()),
        "argmin_index_hist": np.histogram(A["cvargmin"], bins=NDISP,
                                          range=(0, NDISP))[0].tolist(),
        "epe_of_hard_argmin_x8": float(np.abs(hard - gt).mean()),
        "corr_argmin_gt": float(np.corrcoef(A["cvargmin"], gt)[0, 1]),
        "argmin_within_1_candidate_of_gt": float(
            (np.abs(A["cvargmin"] - np.clip(gt, 0, MAXD) / SPACING) <= 1).mean()),
    }
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("epe", "d1", "stage", "cost_volume",
                                          "quantisation")}, indent=2))


if __name__ == "__main__":
    main()
