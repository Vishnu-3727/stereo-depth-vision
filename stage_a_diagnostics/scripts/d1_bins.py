"""STAGE-A D1: per-disparity-bin error decomposition + gap attribution.

Frozen-model measurement only. No training, no tuning, no conclusions.

Bins (wider than eval_p2a.py; top bins matter):
  [0,16) [16,32) [32,48) [48,64) [64,80) [80,96) [96,112) [112,128)
  [128,144) [144,160) [160,184) [184,inf)

P2A per-seed (P,G) vectors are REUSED from D0 raw dumps (no re-inference).
Reference ONNX is scored with the SAME mechanism as
phase1/runs/arm_v_diag/reference_strata.py (onnxruntime, normalize(),
hailo_val, disparity_scale=256.0, occluded=True) but at THESE bin edges --
no interpolation of reference_strata.json (its bins differ).

Method-mean bin row: additive stats (EPE/D1/signed/means/max/contribution)
are the arithmetic mean across the 3 seeds (documented); slope/intercept/R2
are pooled regressions over concatenated 3-seed predictions. Pooled EPE is
asserted equal to mean-of-seeds EPE (mathematically identical; guards wiring).

Gap attribution per bin: Bin_gap = P2A_mean_EPE - Ref_EPE;
Gap_contribution = Pixel_mass * Bin_gap; must sum to the total method gap.

Output (atomic): stage_a_diagnostics/disparity_bins.json
Raw (atomic):    stage_a_diagnostics/raw/d1_reference.npz (P_ref, G)
                 stage_a_diagnostics/raw/d1_bins.csv (flat per-bin table)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

OUT = REPO / "stage_a_diagnostics"
RAW = OUT / "raw"

EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
MIN_REGRESSION_PX = 1000  # below this, slope/intercept/R2 are NOT reported
REF_EPE_FROZEN = 1.3134471


def atomic_write_json(path: Path, obj: dict) -> None:
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


def bin_stats(P: np.ndarray, G: np.ndarray, lo: float, hi: float, total_n: int) -> dict:
    m = (G >= lo) & (G < hi)
    n = int(m.sum())
    row: dict = {"lo": lo if np.isfinite(lo) else None,
                 "hi": hi if np.isfinite(hi) else None,
                 "px": n, "frac_px": float(n / total_n) if total_n else 0.0}
    if n == 0:
        row["status"] = "EMPTY"
        return row
    err = np.abs(P[m] - G[m])
    if not np.isfinite(err).all():
        raise ValueError(f"NaN/inf error in bin [{lo},{hi})")
    row["epe"] = float(err.mean())
    row["d1_pct"] = float((((err > 3) & (err > 0.05 * G[m])).mean()) * 100)
    row["signed_err"] = float((P[m] - G[m]).mean())
    row["mean_pred"] = float(P[m].mean())
    row["mean_gt"] = float(G[m].mean())
    row["max_pred"] = float(P[m].max())
    row["contribution_to_epe"] = float(row["frac_px"] * row["epe"])
    if n >= MIN_REGRESSION_PX and float(G[m].var()) > 0 and float(P[m].var()) > 0:
        a, b = np.polyfit(G[m], P[m], 1)
        yh = a * G[m] + b
        ss_res = float(((P[m] - yh) ** 2).sum())
        ss_tot = float(((P[m] - P[m].mean()) ** 2).sum())
        row["slope"] = float(a)
        row["intercept"] = float(b)
        row["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        row["regression_n"] = n
    else:
        row["slope"] = None
        row["intercept"] = None
        row["r2"] = None
        row["regression_status"] = (
            "INSUFFICIENT (px<%d or degenerate variance); regression not meaningful"
            % MIN_REGRESSION_PX)
    return row


def score_reference() -> tuple[np.ndarray, np.ndarray, dict]:
    import onnxruntime as ort  # noqa: N812

    from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract
    from src.datasets.kitti2015 import Kitti2015Stereo, normalize

    onnx_path = REPO / "reference" / "onnx" / "stereonet.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"reference ONNX missing: {onnx_path}")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    if len(ins) < 2:
        raise ValueError(f"reference ONNX has <2 inputs: {ins}")
    outn = sess.get_outputs()[0].name
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    if len(ds) != 40:
        raise ValueError(f"reference scoring split has {len(ds)} scenes, need 40")
    preds, gts = [], []
    for i in range(len(ds)):
        s = ds[i]
        if s.disparity.shape != (368, 1232):
            raise ValueError(f"shape mismatch scene {s.name}")
        o = sess.run([outn], {ins[0]: normalize(s.left),
                              ins[1]: normalize(s.right)})[0]
        p = o[0, 0].astype(np.float64)
        if not np.isfinite(p).all():
            raise ValueError(f"NaN/inf reference prediction scene {s.name}")
        v = s.disparity > 0
        if v.sum() == 0:
            raise ValueError(f"empty valid mask scene {s.name}")
        preds.append(p[v])
        gts.append(s.disparity[v].astype(np.float64))
        print(f"ref {i + 1}/40 {s.name}", flush=True)
    P, G = np.concatenate(preds), np.concatenate(gts)
    if (G <= 0).any():
        raise ValueError("non-positive GT inside valid mask")
    m = pooled_metrics(preds, gts)
    guard = refuse_unless_contract(len(ds), m["valid_pixels"], 256.0,
                                  "hailo_val", "disp_occ_0")
    if not guard["contract_match"]:
        raise ValueError("reference scoring broke the frozen contract")
    meta = {"epe": float(m["epe"]), "d1": float(m["d1"]),
            "valid_pixels": int(m["valid_pixels"]),
            "contract_match": guard["contract_match"]}
    return P, G, meta


def main() -> None:
    seeds = {}
    for seed in (0, 1, 2):
        rp = RAW / f"d0_preds_s{seed}.npz"
        if not rp.exists():
            raise FileNotFoundError(f"D0 raw dump missing: {rp}; run d0_verify.py first")
        z = np.load(rp)
        P, G = z["P"].astype(np.float64), z["G"].astype(np.float64)
        if P.shape != G.shape or P.size == 0:
            raise ValueError(f"bad raw vectors seed {seed}: {P.shape} {G.shape}")
        if not np.isfinite(P).all() or not np.isfinite(G).all():
            raise ValueError(f"NaN/inf in D0 raw dump seed {seed}")
        seeds[seed] = (P, G)
    P0, G0 = seeds[0]
    total_n = int(G0.size)
    for seed in (1, 2):
        if not np.array_equal(seeds[seed][1], G0):
            raise ValueError(f"GT vector differs between seeds 0 and {seed}: GT must be identical")

    P_ref, G_ref, ref_meta = score_reference()
    if not np.array_equal(G_ref, G0):
        raise ValueError("reference GT vector differs from P2A GT vector: split mismatch")
    fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, P=P_ref, G=G_ref)
        os.replace(tmp, RAW / "d1_reference.npz")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    print(f"reference EPE {ref_meta['epe']:.7f} (frozen {REF_EPE_FROZEN}) "
          f"diff={ref_meta['epe'] - REF_EPE_FROZEN:.2e}", flush=True)

    per_seed_rows, method_rows, ref_rows, gap_rows = {}, [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        srows = {}
        for seed in (0, 1, 2):
            P, G = seeds[seed]
            srows[str(seed)] = bin_stats(P, G, lo, hi, total_n)
        per_seed_rows[f"[{lo},{hi})" if np.isfinite(hi) else f"[{lo},inf)"] = srows
        # method mean: additive stats = mean across seeds; regression pooled
        Ppool = np.concatenate([seeds[s][0][(G0 >= lo) & (G0 < hi)] for s in (0, 1, 2)])
        Gpool = np.concatenate([G0[(G0 >= lo) & (G0 < hi)] for _ in (0, 1, 2)])
        n = int(((G0 >= lo) & (G0 < hi)).sum())
        mrow: dict = {"lo": lo if np.isfinite(lo) else None,
                      "hi": hi if np.isfinite(hi) else None,
                      "px": n, "frac_px": float(n / total_n)}
        if n == 0:
            mrow["status"] = "EMPTY"
        else:
            for k in ("epe", "d1_pct", "signed_err", "mean_pred", "mean_gt",
                      "max_pred", "contribution_to_epe"):
                vals = [srows[str(s)][k] for s in (0, 1, 2)]
                mrow[k] = float(np.mean(vals))
                mrow[k + "_per_seed"] = [float(v) for v in vals]
            pooled = bin_stats(Ppool, Gpool, lo, hi, 3 * total_n)
            assert abs(pooled["epe"] - mrow["epe"]) < 1e-9, \
                f"pooled EPE != mean-of-seeds in bin [{lo},{hi})"
            for k in ("slope", "intercept", "r2"):
                mrow[k] = pooled[k]
            mrow["regression"] = "pooled over 3x{0} concatenated seed predictions".format(n)
            if pooled.get("regression_status"):
                mrow["regression_status"] = pooled["regression_status"]
        method_rows.append(mrow)
        rrow = bin_stats(P_ref, G_ref, lo, hi, total_n)
        ref_rows.append(rrow)
        # gap attribution
        if mrow.get("epe") is not None and rrow.get("epe") is not None:
            gap = float(mrow["epe"] - rrow["epe"])
            contrib = float(mrow["frac_px"] * gap)
        else:
            gap, contrib = None, 0.0
        gap_rows.append({"lo": mrow["lo"], "hi": mrow["hi"], "px": n,
                         "pixel_mass": float(n / total_n),
                         "p2a_bin_epe": mrow.get("epe"), "ref_bin_epe": rrow.get("epe"),
                         "bin_gap": gap, "gap_contribution": contrib})

    # closure checks: bin contributions must reconstruct each total EPE
    def check_closure(rows, total, name):
        rec = sum(r.get("contribution_to_epe", 0.0) for r in rows
                  if r.get("contribution_to_epe") is not None)
        print(f"{name}: reconstructed {rec:.10f} vs total {total:.10f} "
              f"diff={rec - total:.2e}", flush=True)
        return rec
    mean_epe = float(np.mean([float(np.abs(seeds[s][0] - G0).mean()) for s in (0, 1, 2)]))
    check_closure(method_rows, mean_epe, "P2A method-mean")
    check_closure(ref_rows, ref_meta["epe"], "reference")
    gap_sum = float(sum(g["gap_contribution"] for g in gap_rows))
    total_gap = float(mean_epe - ref_meta["epe"])
    gap_sums_ok = abs(gap_sum - total_gap) < 1e-9
    print(f"gap attribution sums to {gap_sum:.10f} vs total gap {total_gap:.10f} "
          f"diff={gap_sum - total_gap:.2e} -> {'CLOSES' if gap_sums_ok else 'DOES NOT CLOSE'}",
          flush=True)
    if not gap_sums_ok:
        print("LOUD: Gap_contribution column does NOT sum to the total gap -- INVESTIGATE",
              flush=True)

    top3 = sorted([g for g in gap_rows if g["gap_contribution"] is not None],
                  key=lambda g: g["gap_contribution"], reverse=True)[:3]
    print("top3 by gap_contribution:", flush=True)
    for g in top3:
        print(f"  [{g['lo']},{g['hi']}) mass={g['pixel_mass']:.6f} "
              f"p2a={g['p2a_bin_epe']:.4f} ref={g['ref_bin_epe']:.4f} "
              f"gap={g['bin_gap']:.4f} contrib={g['gap_contribution']:.6f}", flush=True)

    out = {
        "bin_edges": "[0,16) [16,32) [32,48) [48,64) [64,80) [80,96) "
                     "[96,112) [112,128) [128,144) [144,160) [160,184) [184,inf)",
        "conventions": {
            "d1_pct": "mean((abs err>3) & (abs err>0.05*GT))*100, GT in px (scale 256.0)",
            "contribution_to_epe": "frac_px * bin_epe; bin contributions sum to total EPE",
            "method_mean": "additive stats = mean across 3 seeds; "
                           "slope/intercept/r2 = pooled regression over concatenated "
                           "3-seed predictions; pooled EPE asserted == mean-of-seeds",
            "regression_rule": "slope/intercept/r2 reported only if px>=1000 and "
                               "var(GT)>0 and var(pred)>0, else null with reason",
            "reference": "reference/onnx/stereonet.onnx scored by reference_strata.py "
                         "mechanism at THESE edges (no interpolation)",
        },
        "totals": {"p2a_method_mean_epe": mean_epe, "reference_epe": ref_meta["epe"],
                   "total_gap_px": total_gap, "gap_attribution_sum": gap_sum,
                   "gap_attribution_closes": gap_sums_ok,
                   "reference_valid_pixels": ref_meta["valid_pixels"],
                   "reference_contract_match": ref_meta["contract_match"]},
        "per_seed_bins": per_seed_rows,
        "method_mean_bins": method_rows,
        "reference_bins": ref_rows,
        "gap_attribution": gap_rows,
        "top3_by_gap_contribution": top3,
    }
    atomic_write_json(OUT / "disparity_bins.json", out)

    # flat CSV (atomic)
    fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bin", "px", "mass", "p2a_epe_s0", "p2a_epe_s1", "p2a_epe_s2",
                        "p2a_mean_epe", "ref_epe", "bin_gap", "gap_contribution",
                        "p2a_d1_mean", "ref_d1", "p2a_signed_mean", "ref_signed",
                        "p2a_slope_pooled", "ref_slope", "p2a_r2_pooled", "ref_r2"])
            for mrow, rrow, grow in zip(method_rows, ref_rows, gap_rows):
                lab = f"[{mrow['lo']},{mrow['hi']})" if mrow["hi"] is not None else f"[{mrow['lo']},inf)"
                pes = mrow.get("epe_per_seed", [None, None, None])

                def f6(v):
                    return f"{v:.6f}" if v is not None else ""

                def f4(v):
                    return f"{v:.4f}" if v is not None else ""

                w.writerow([lab, mrow["px"], f"{grow['pixel_mass']:.9f}",
                            *(f6(v) for v in pes),
                            f6(mrow.get("epe")),
                            f6(rrow.get("epe")),
                            f6(grow["bin_gap"]) if grow["bin_gap"] is not None else "",
                            f"{grow['gap_contribution']:.9f}",
                            f4(mrow.get("d1_pct")),
                            f4(rrow.get("d1_pct")),
                            f4(mrow.get("signed_err")),
                            f4(rrow.get("signed_err")),
                            mrow.get("slope", ""), rrow.get("slope", ""),
                            mrow.get("r2", ""), rrow.get("r2", "")])
        os.replace(tmp, RAW / "d1_bins.csv")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print("wrote disparity_bins.json + raw/d1_reference.npz + raw/d1_bins.csv", flush=True)


if __name__ == "__main__":
    main()
