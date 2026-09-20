"""STAGE-A D2b: CORRECTED readout comparison (frozen model, read-only).

CORRECTION of the D2 readout comparison, which was CONFOUNDED by sparse
substitution: D2 built alternative disparity_initial maps by scattering
replacement values at valid pixels only (~21% of the image) and pushing the
patchy map through the frozen dilated refinement. The refinement therefore
saw artificial step discontinuities it never saw in training, so every D2
oracle number measured the refinement's reaction to a corrupted input, not
the value of a better readout. Distribution statistics in
cost_distribution.json are sound and are left untouched.

Three confound-free protocols (this script):

Protocol A - FROZEN RESIDUAL (primary, zero confound). The refinement is NOT
  re-run at all. Take the residual the model ACTUALLY produced from its ACTUAL
  input: stages["refinement_residual"]. For each readout variant, at valid
  pixels only:
      disparity_alt_px = relu(alt_cand + residual_actual)
  and score EPE/D1 vs GT. Variants: soft-argmin (must reproduce baseline
  EXACTLY - gate), hard argmax, top-2/3/4 GT-oracle, continuous GT-oracle
  (gt_cand clipped to [0,23]). The refinement output is held literally
  fixed, so there is no out-of-distribution input anywhere.
  GATE: soft-argmin EPE under Protocol A must equal baseline EPE to ~1e-9
  per seed. If not, units or masking are wrong: STOP.

Protocol B - DENSE ARGMAX THROUGH REFINEMENT (secondary, no GT needed). The
  hard argmax is defined at EVERY pixel, so the argmax map is built DENSELY
  over the whole (1,1,368,1232) image and pushed through the unchanged frozen
  refinement, then scored on valid pixels. No patchiness. GT-based oracles
  CANNOT be made dense (GT is sparse by nature); they get Protocol A only.

Protocol C - PATCHINESS CONTROL. Baseline soft-argmin map + constant offset
  delta = +0.5 candidate units, two ways: C1 sparse (valid pixels only, same
  scatter pattern as the bad D2 run) and C2 dense (all pixels). Both go
  through the unchanged frozen refinement and are scored on valid pixels.
  (C1 - C2) is the EPE damage attributable purely to the sparse-substitution
  artefact, with the readout change held identical.

Conventions (same as D2): gt_cand = GT_px / 8. Candidate k covers 8*k px.
D1 = mean((abs_err > 3) & (abs_err > 0.05 * GT)) * 100 (percent).

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/cost_readout_corrected.json
  stage_a_diagnostics/raw/d2b_s{0,1,2}.npz
  stage_a_diagnostics/raw/d2b_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d2b_readout.py [--limit N] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_a_diagnostics"
RAW = OUT / "raw"
RUNS = {0: "p2a_scale_coverage", 1: "p2a_scale_coverage_s1", 2: "p2a_scale_coverage_s2"}
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
D = 24
STRIDE = 8
DELTA = 0.5  # candidate units, Protocol C offset
EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
EXPECTED_SEED0_EPE = 1.4149795736625577
EXPECTED_VALID = 3802797
TOL_BASELINE_PX = 1e-6   # same code path + same weights: must match ~exactly
TOL_GATE_PX = 1e-9       # Protocol A soft-argmin must reproduce baseline


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


def atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def bin_label(lo, hi) -> str:
    if np.isfinite(hi):
        return f"[{int(lo)},{int(hi)})"
    return f"[{int(lo)},inf)"


LABELS = [bin_label(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def d1_flags(err: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return ((err > 3.0) & (err > 0.05 * gt)).astype(np.float64)


class Acc:
    """Float64 accumulators: pooled sums + per-D1-bin sums + GT>=96 sums."""

    # A_: frozen-residual variants; B_: dense argmax; C1/C2: patchiness control
    VARIANTS = ("baseline", "A_soft", "A_argmax", "A_topk2", "A_topk3",
                "A_topk4", "A_cont", "B_argmax", "C1_sparse", "C2_dense")

    def __init__(self) -> None:
        self.n = 0
        self.n96 = 0
        nb = len(LABELS)
        self.bn = np.zeros(nb, dtype=np.int64)
        self.be = {v: np.zeros(nb, dtype=np.float64) for v in self.VARIANTS}
        self.bd1 = {v: np.zeros(nb, dtype=np.float64) for v in self.VARIANTS}
        self.e = {v: 0.0 for v in self.VARIANTS}
        self.d1 = {v: 0.0 for v in self.VARIANTS}
        self.e96 = {v: 0.0 for v in self.VARIANTS}
        self.d196 = {v: 0.0 for v in self.VARIANTS}
        self.images: list[dict] = []


@torch.no_grad()
def process_seed(seed: int, device: str, limit: int | None) -> Acc:
    ck = REPO / "phase2" / "runs" / RUNS[seed] / "p2a_best.pth"
    if not ck.exists():
        raise FileNotFoundError(f"checkpoint missing: {ck}")
    blob = torch.load(str(ck), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"seed {seed}: checkpoint missing ['model'] key")
    model = StereoNet(StereoNetConfig(**CFG))
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    if len(ds) != 40:
        raise ValueError(f"seed {seed}: split has {len(ds)} scenes, need 40")
    n_img = len(ds) if limit is None else min(limit, len(ds))

    # ---- pre-checks on scene 0 ----
    s0 = ds[0]
    if s0.disparity.shape != (368, 1232):
        raise ValueError(f"shape mismatch scene {s0.name}: {s0.disparity.shape}")
    left0 = torch.from_numpy(normalize(s0.left)).to(device)
    right0 = torch.from_numpy(normalize(s0.right)).to(device)
    ret = model(left0, right0, return_stages=True)
    if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict)):
        raise ValueError("forward(return_stages=True) did not return "
                         "(disparity, stages-dict); refusing to continue")
    _out0, st0 = ret
    for k in ("aggregated_cost", "disparity_initial", "refinement_residual",
              "disparity_final"):
        if k not in st0:
            raise ValueError(f"stages dict missing key: {k}")
    if tuple(st0["aggregated_cost"].shape) != (1, D, 368 // STRIDE, 1232 // STRIDE):
        raise ValueError(f"aggregated_cost shape {tuple(st0['aggregated_cost'].shape)}")
    if tuple(st0["disparity_initial"].shape) != (1, 1, 368, 1232):
        raise ValueError("disparity_initial shape mismatch")
    if tuple(st0["refinement_residual"].shape) != (1, 1, 368, 1232):
        raise ValueError("refinement_residual shape mismatch")
    for k in ("aggregated_cost", "disparity_initial", "refinement_residual",
              "disparity_final"):
        if not torch.isfinite(st0[k]).all():
            raise ValueError(f"NaN/inf in {k} on scene 0")
    del ret, st0
    print(f"seed {seed} pre-checks OK", flush=True)

    acc = Acc()
    idx_grid = torch.arange(D, dtype=torch.float32, device=device)

    for i in range(n_img):
        s = ds[i]
        if s.disparity.shape != (368, 1232):
            raise ValueError(f"shape mismatch scene {s.name}")
        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        gt_np = s.disparity.astype(np.float64)
        valid_np = gt_np > 0
        nv = int(valid_np.sum())
        if nv == 0:
            raise ValueError(f"empty valid mask scene {s.name}")
        gt_valid_np = gt_np[valid_np]
        gt_cand_np = gt_valid_np / STRIDE

        out, st = model(left, right, return_stages=True)
        agg = st["aggregated_cost"]
        d_init = st["disparity_initial"]
        resid = st["refinement_residual"]
        final = st["disparity_final"]
        if not (torch.isfinite(agg).all() and torch.isfinite(d_init).all()
                and torch.isfinite(resid).all() and torch.isfinite(final).all()):
            raise ValueError(f"NaN/inf in forward scene {s.name}")

        # frozen readout weights, exactly mirroring DisparityRegression (as D2)
        cu = F.interpolate(agg, size=(368, 1232), mode="bilinear", align_corners=True)
        cus = (cu - cu.mean(dim=1, keepdim=True)) / (cu.std(dim=1, keepdim=True) + 1e-6)
        w = torch.softmax(-cus, dim=1)
        if not torch.isfinite(w).all():
            raise ValueError(f"NaN/inf in softmax weights scene {s.name}")
        soft = (w * idx_grid.view(1, D, 1, 1)).sum(dim=1, keepdim=True)

        valid_t = torch.from_numpy(valid_np).to(device)
        W = w[0, :, valid_t]              # (24, N) float32
        N = W.shape[1]
        if N != nv:
            raise ValueError("valid count mismatch")
        gt_t = torch.from_numpy(gt_valid_np).to(device)    # float64 px
        gtc = torch.from_numpy(gt_cand_np).to(device)      # float64 cand
        Wd = W.double()
        soft_v = soft[0, 0, valid_t].double()
        resid_v = resid[0, 0, valid_t].double()

        # ---- PROTOCOL A: frozen residual, no refinement re-run ----
        order = torch.argsort(Wd, dim=0, descending=True)
        del order  # (kept topk below; full order not needed)
        topk_idx = torch.topk(Wd, 4, dim=0).indices       # (4, N)
        argmax = torch.argmax(Wd, dim=0).double()          # (N,) cand units
        alt: dict[str, torch.Tensor] = {"A_soft": soft_v, "A_argmax": argmax}
        for k in (2, 3, 4):
            cand = topk_idx[:k, :]
            diff = (cand.double() - gtc.unsqueeze(0)).abs()
            pick = cand[torch.argmin(diff, dim=0),
                        torch.arange(N, device=device)].double()
            alt[f"A_topk{k}"] = pick
        alt["A_cont"] = torch.clamp(gtc, 0.0, D - 1)
        errs: dict[str, torch.Tensor] = {}
        errs["baseline"] = (final[0, 0, valid_t].double() - gt_t).abs()
        for key, a in alt.items():
            disp_alt = torch.relu(a + resid_v)
            if not torch.isfinite(disp_alt).all():
                raise ValueError(f"NaN/inf protocol-A {key} scene {s.name}")
            errs[key] = (disp_alt - gt_t).abs()
            del disp_alt
        del alt, topk_idx, argmax

        # ---- PROTOCOL B: dense argmax through frozen refinement ----
        am_dense = torch.argmax(w, dim=1, keepdim=True).float()  # (1,1,H,W)
        if not torch.isfinite(am_dense).all():
            raise ValueError(f"NaN/inf dense-argmax scene {s.name}")
        res_b = model.refinement(am_dense, left)
        if not torch.isfinite(res_b).all():
            raise ValueError(f"NaN/inf refinement-B scene {s.name}")
        fin_b = torch.relu(am_dense.double() + res_b.double())
        if not torch.isfinite(fin_b).all():
            raise ValueError(f"NaN/inf final-B scene {s.name}")
        errs["B_argmax"] = (fin_b[0, 0, valid_t] - gt_t).abs()
        del am_dense, res_b, fin_b

        # ---- PROTOCOL C: patchiness control, delta=+0.5 cand units ----
        c1 = d_init.detach().clone()
        c1[0, 0, valid_t] += DELTA           # sparse: valid pixels only
        c2 = d_init.detach().clone() + DELTA  # dense: all pixels
        for key, cmap in (("C1_sparse", c1), ("C2_dense", c2)):
            res_c = model.refinement(cmap, left)
            if not torch.isfinite(res_c).all():
                raise ValueError(f"NaN/inf refinement-{key} scene {s.name}")
            fin_c = torch.relu(cmap.double() + res_c.double())
            if not torch.isfinite(fin_c).all():
                raise ValueError(f"NaN/inf final-{key} scene {s.name}")
            errs[key] = (fin_c[0, 0, valid_t] - gt_t).abs()
            del res_c, fin_c
        del c1, c2

        # ---- accumulate (float64) ----
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
        e_np = {k: to_np(v) for k, v in errs.items()}
        if not all(np.isfinite(a).all() for a in e_np.values()):
            raise ValueError(f"NaN/inf in accumulated stats scene {s.name}")
        f_np = {k: d1_flags(v, gt_valid_np) for k, v in e_np.items()}

        b = np.searchsorted(EDGES, gt_valid_np, side="right") - 1
        b = np.clip(b, 0, len(LABELS) - 1)
        m96 = gt_valid_np >= 96.0

        acc.n += N
        acc.n96 += int(m96.sum())
        np.add.at(acc.bn, b, 1)
        for v in Acc.VARIANTS:
            acc.e[v] += float(e_np[v].sum())
            acc.d1[v] += float(f_np[v].sum())
            acc.e96[v] += float(e_np[v][m96].sum())
            acc.d196[v] += float(f_np[v][m96].sum())
            np.add.at(acc.be[v], b, e_np[v])
            np.add.at(acc.bd1[v], b, f_np[v])
        acc.images.append({"scene": s.name, "valid": N, **{
            f"epe_{v}": float(e_np[v].mean()) for v in Acc.VARIANTS}})

        del out, st, agg, d_init, resid, final, cu, cus, w, soft
        del W, Wd, soft_v, resid_v, valid_t, gt_t, gtc
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        print(f"seed {seed} [{i + 1}/{n_img}] {s.name} valid={N} "
              f"base={float(e_np['baseline'].mean()):.4f} "
              f"A_soft={float(e_np['A_soft'].mean()):.4f} "
              f"A_cont={float(e_np['A_cont'].mean()):.4f} "
              f"B_argmax={float(e_np['B_argmax'].mean()):.4f} "
              f"C1={float(e_np['C1_sparse'].mean()):.4f} "
              f"C2={float(e_np['C2_dense'].mean()):.4f}", flush=True)

    if acc.n != EXPECTED_VALID and limit is None:
        raise ValueError(f"seed {seed}: valid pixels {acc.n} != {EXPECTED_VALID}")
    return acc


def summarize(seed: int, acc: Acc) -> dict:
    n = acc.n
    if n == 0:
        raise ValueError("zero valid pixels")
    out: dict = {
        "valid_pixels": n,
        "epe_pooled": {v: float(acc.e[v] / n) for v in Acc.VARIANTS},
        "d1_pooled": {v: float(acc.d1[v] / n * 100.0) for v in Acc.VARIANTS},
    }
    base = acc.e["baseline"] / n
    out["epe_delta_vs_baseline_pooled"] = {
        v: float(acc.e[v] / n - base) for v in Acc.VARIANTS if v != "baseline"}
    # Protocol A gate quantity
    out["gate_A_soft_minus_baseline_px"] = float(acc.e["A_soft"] / n - base)
    # Protocol C damage
    out["sparse_substitution_damage_px_C1_minus_C2"] = float(
        acc.e["C1_sparse"] / n - acc.e["C2_dense"] / n)
    n96 = acc.n96
    out["gt_ge_96"] = {"valid_pixels": int(n96), "frac_of_valid": float(n96 / n)}
    if n96:
        out["gt_ge_96"]["epe"] = {v: float(acc.e96[v] / n96) for v in Acc.VARIANTS}
        out["gt_ge_96"]["d1"] = {v: float(acc.d196[v] / n96 * 100.0)
                                 for v in Acc.VARIANTS}
        b96 = acc.e96["baseline"] / n96
        out["gt_ge_96"]["epe_delta_vs_baseline"] = {
            v: float(acc.e96[v] / n96 - b96) for v in Acc.VARIANTS
            if v != "baseline"}
    bins = []
    for bi, lab in enumerate(LABELS):
        cnt = int(acc.bn[bi])
        if cnt == 0:
            bins.append({"bin": lab, "px": 0, "status": "EMPTY"})
            continue
        row = {
            "bin": lab, "px": cnt, "frac_px": float(cnt / n),
            "epe": {v: float(acc.be[v][bi] / cnt) for v in Acc.VARIANTS},
            "d1": {v: float(acc.bd1[v][bi] / cnt * 100.0) for v in Acc.VARIANTS},
        }
        bb = acc.be["baseline"][bi] / cnt
        row["epe_delta_vs_baseline"] = {
            v: float(acc.be[v][bi] / cnt - bb)
            for v in Acc.VARIANTS if v != "baseline"}
        bins.append(row)
    out["per_bin"] = bins
    out["per_image_epe"] = acc.images
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D2b device={device} seeds={seeds} limit={args.limit}", flush=True)

    results: dict = {}
    accs: dict[int, Acc] = {}
    for seed in seeds:
        acc = process_seed(seed, device, args.limit)
        accs[seed] = acc
        results[str(seed)] = summarize(seed, acc)
        atomic_write_npz(
            RAW / f"d2b_s{seed}.npz",
            bin_counts=acc.bn.astype(np.float64),
            **{f"binepe_{v}": np.divide(acc.be[v], np.maximum(acc.bn, 1))
               for v in acc.be},
            **{f"bind1_{v}": np.divide(acc.bd1[v], np.maximum(acc.bn, 1)) * 100.0
               for v in acc.bd1})

    # GATE 1: baseline reproduces D2/D0 seed-0 number (full runs only)
    if args.limit is None and 0 in accs:
        e0 = results["0"]["epe_pooled"]["baseline"]
        ok = abs(e0 - EXPECTED_SEED0_EPE) < TOL_BASELINE_PX
        print(f"BASELINE seed0={e0:.10f} expected={EXPECTED_SEED0_EPE:.10f} "
              f"diff={e0 - EXPECTED_SEED0_EPE:.2e} -> "
              f"{'REPRODUCES' if ok else 'MISMATCH: STOP'}", flush=True)
        if not ok:
            raise SystemExit("D2b STOP: baseline does not reproduce step-1 number")

    # GATE 2 (Protocol A): frozen-residual soft-argmin reproduces baseline
    gate_ok = True
    for seed in seeds:
        if args.limit is not None:
            break
        g = results[str(seed)]["gate_A_soft_minus_baseline_px"]
        ok = abs(g) < TOL_GATE_PX
        gate_ok &= ok
        print(f"GATE-A seed{seed}: A_soft-baseline={g:.2e} -> "
              f"{'PASS' if ok else 'FAIL: STOP'}", flush=True)
    if args.limit is None and not gate_ok:
        raise SystemExit("D2b STOP: Protocol A gate failed; units or masking wrong")

    if args.limit is None:
        mm_epe = {v: float(np.mean([results[str(s)]["epe_pooled"][v]
                                    for s in seeds])) for v in Acc.VARIANTS}
        mm_d1 = {v: float(np.mean([results[str(s)]["d1_pooled"][v]
                                   for s in seeds])) for v in Acc.VARIANTS}
        base = mm_epe["baseline"]
        n96_tot = sum(results[str(s)]["gt_ge_96"]["valid_pixels"] for s in seeds)
        mm96_epe: dict = {}
        for v in Acc.VARIANTS:
            num = sum(results[str(s)]["gt_ge_96"]["epe"][v]
                      * results[str(s)]["gt_ge_96"]["valid_pixels"] for s in seeds)
            mm96_epe[v] = float(num / n96_tot)
        out = {
            "supersedes": ("cost_distribution.json readout_* keys, marked "
                           "CONFOUNDED (sparse-substitution artefact); distribution "
                           "statistics there remain authoritative and are untouched"),
            "conventions": {
                "gt_cand": "GT_px / 8.0; candidate units 0..23 cover 0..184 px",
                "protocol_A": ("frozen residual: relu(alt_cand + residual_actual) "
                               "at valid pixels only; refinement NOT re-run"),
                "protocol_B": ("dense hard-argmax map over the full image through "
                               "the unchanged frozen refinement; scored on valid "
                               "pixels. GT oracles cannot be made dense (GT is "
                               "sparse by nature) and get Protocol A only."),
                "protocol_C": ("soft-argmin map + 0.5 candidate units, sparse "
                               "(valid only, C1) vs dense (all pixels, C2), both "
                               "through the unchanged frozen refinement"),
                "d1": "mean((abs_err > 3) & (abs_err > 0.05*GT)) * 100, percent",
                "topk_oracle": ("among top-k weighted candidates, the index "
                                "closest to unclipped gt_cand (same rule as D2)"),
                "cont_oracle": "gt_cand clipped to [0, 23] candidate units",
                "d1_bins": ("per-bin rows reuse D1 edges "
                            "[0,16)...[160,184),[184,inf)"),
            },
            "config": {"downsample_levels": 3, "num_disparities": 24,
                       "cost_volume_shift": "right",
                       "regression_normalize": True, "stride_px": 8,
                       "candidate_range_px": [0, 184]},
            "gates": {
                "baseline_reproduces_D2_seed0": True,
                "protocol_A_soft_reproduces_baseline_1e9_per_seed": {
                    str(s): results[str(s)]["gate_A_soft_minus_baseline_px"]
                    for s in seeds},
            },
            "seeds": results,
            "method_mean_epe": mm_epe,
            "method_mean_epe_delta_vs_baseline": {
                v: mm_epe[v] - base for v in Acc.VARIANTS if v != "baseline"},
            "method_mean_d1": mm_d1,
            "method_mean_gt_ge_96_epe": mm96_epe,
            "method_mean_sparse_damage_C1_minus_C2_px": float(
                mm_epe["C1_sparse"] - mm_epe["C2_dense"]),
            "interpretation": ("Factual statement only, no verdict and no "
                "'proven' claim; step 8 does mechanism verdicts. Protocol A "
                "(frozen residual, zero confound) measures whether the correct "
                "disparity information is present in the frozen cost "
                "distribution but lost by the current readout: a negative "
                "delta vs baseline for an oracle variant means the frozen "
                "cost weights already ranked the right answer highly enough "
                "that only the readout stood in the way. Protocol B reports "
                "the dense hard-argmax through the unchanged frozen "
                "refinement. Protocol C (C1-C2) quantifies the EPE damage of "
                "the sparse-substitution artefact in the original D2 run."),
            "unmeasurable": ("NOT MEASURABLE WITH CURRENT ARTIFACTS: dense "
                "GT-oracle maps through the refinement (GT is sparse by "
                "nature); nothing else was unmeasurable."),
        }
        atomic_write_json(OUT / "cost_readout_corrected.json", out)
        import csv
        fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["seed", "bin", "px", "frac_px"] +
                           [f"epe_{v}" for v in Acc.VARIANTS] +
                           [f"d1_{v}" for v in Acc.VARIANTS])
                for s in seeds:
                    for row in results[str(s)]["per_bin"]:
                        if row.get("status") == "EMPTY":
                            w.writerow([s, row["bin"], 0, 0] +
                                       [""] * (2 * len(Acc.VARIANTS)))
                            continue
                        w.writerow([s, row["bin"], row["px"],
                                    f"{row['frac_px']:.9f}"] +
                                   [f"{row['epe'][v]:.6f}" for v in Acc.VARIANTS] +
                                   [f"{row['d1'][v]:.4f}" for v in Acc.VARIANTS])
            os.replace(tmp, RAW / "d2b_bins.csv")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print("wrote cost_readout_corrected.json + raw/d2b_s*.npz + raw/d2b_bins.csv",
              flush=True)


if __name__ == "__main__":
    main()
