"""ARM-Z smoke test (preregistration section 10 + liveness section 11).

Run as `python phase1\\scripts\\smoke_arm_z.py`. Prints PASS/FAIL per check
and exits non-zero on any failure. No training, no checkpoint written.
Saves its output to phase1/runs/arm_z_smoke/smoke.json.
"""
from __future__ import annotations

import io
import json
import sys
import contextlib
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.stereonet import StereoNet, StereoNetConfig
from src.models.stereonet.cost_volume import build_cost_volume, shift_right
from src.models.stereonet.stereonet import normalize_features_l2

FAILURES: list[str] = []
LINES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    line = ("PASS" if ok else "FAIL") + " " + name + (((" - " + detail) if detail else ""))
    print(line, flush=True)
    LINES.append(line)
    if not ok:
        FAILURES.append(name)


def arm_v_config() -> StereoNetConfig:
    return StereoNetConfig(
        downsample_levels=3,
        num_disparities=24,
        cost_volume_shift="right",
        regression_normalize=True,
    )


def arm_z_config() -> StereoNetConfig:
    cfg = arm_v_config()
    cfg.feature_normalize = True
    return cfg


def main() -> int:
    torch.manual_seed(0)
    B, H0, W0 = 1, 256, 512
    C, D = 32, 24
    stride = 2 ** 3
    Hf, Wf = H0 // stride, W0 // stride
    results: dict = {}

    try:
        model = StereoNet(arm_z_config())
        model.train()
    except Exception as e:  # noqa: BLE001
        check("0 model construction (ARM-Z config builds)", False, str(e))
        print("aborting: construction failed")
        return 1

    left = torch.randn(B, 3, H0, W0)
    right = torch.randn(B, 3, H0, W0)

    # 1. left/right feature tensor shapes at the ARM-V config
    try:
        model.eval()
        with torch.no_grad():
            lf = model.feature_extractor(left)
            rf = model.feature_extractor(right)
        ok = tuple(lf.shape) == (B, C, Hf, Wf) and tuple(rf.shape) == (B, C, Hf, Wf)
        results["feature_shapes"] = [tuple(lf.shape), tuple(rf.shape)]
        check("1 feature shapes (B,32,H/8,W/8)", ok,
              "left %s right %s" % (tuple(lf.shape), tuple(rf.shape)))
    except Exception as e:  # noqa: BLE001
        check("1 feature shapes (B,32,H/8,W/8)", False, str(e))

    # 2. normalization is over channels only
    try:
        torch.manual_seed(1)
        direction = torch.randn(B, C, 1, 1).expand(B, C, Hf, Wf).contiguous()
        scale = torch.linspace(0.5, 3.0, Wf).view(1, 1, 1, Wf).expand(B, 1, Hf, Wf)
        x = direction * scale
        out = normalize_features_l2(x)
        maxdiff = max((out[..., w] - out[..., 0]).abs().max().item() for w in range(1, Wf))
        cos = (out * x).sum(dim=1) / (out.norm(p=2, dim=1) * x.norm(p=2, dim=1)).clamp_min(1e-12)
        cosmin = float(cos.min())
        results["channels_only"] = {"max_col_diff": maxdiff, "min_cosine": cosmin}
        check("2 normalization over channels only", maxdiff < 1e-5 and cosmin > 1 - 1e-5,
              "max col diff %.3g min cosine %.8f" % (maxdiff, cosmin))
    except Exception as e:  # noqa: BLE001
        check("2 normalization over channels only", False, str(e))

    # 3. nonzero feature vectors have ~unit L2 norm
    try:
        with torch.no_grad():
            lfn = normalize_features_l2(lf)
        norms = lfn.norm(p=2, dim=1)
        nmin, nmax, nmean = float(norms.min()), float(norms.max()), float(norms.mean())
        results["unit_norm"] = {"min": nmin, "max": nmax, "mean": nmean}
        check("3 nonzero vectors ~unit norm", abs(nmin - 1) < 1e-4 and abs(nmax - 1) < 1e-4,
              "min %.6f max %.6f mean %.6f" % (nmin, nmax, nmean))
    except Exception as e:  # noqa: BLE001
        check("3 nonzero vectors ~unit norm", False, str(e))

    # 4. zero vectors numerically safe via max(norm, 1e-5)
    try:
        z = torch.zeros(1, 32, 3, 4)
        zn = normalize_features_l2(z)
        ok = bool(torch.isfinite(zn).all().item())
        results["zero_safe"] = {"finite": ok, "max_abs": float(zn.abs().max())}
        check("4 zero vectors safe via max(norm,1e-5)", ok, "max abs %.3g" % float(zn.abs().max()))
    except Exception as e:  # noqa: BLE001
        check("4 zero vectors safe via max(norm,1e-5)", False, str(e))

    # 5. cost volume shape == (B, 32, 24, H, W)
    try:
        with torch.no_grad():
            lfn = normalize_features_l2(lf)
            rfn = normalize_features_l2(rf)
            vol = build_cost_volume(lfn, rfn, D, method="subtract", shift="right")
        ok = tuple(vol.shape) == (B, C, D, Hf, Wf)
        results["volume_shape"] = tuple(vol.shape)
        check("5 cost volume shape (B,32,24,H,W)", ok, "got %s" % (tuple(vol.shape),))
    except Exception as e:  # noqa: BLE001
        check("5 cost volume shape (B,32,24,H,W)", False, str(e))

    # 6. cost semantics == left_normalized - shifted_right_normalized
    try:
        maxd = 0.0
        for k in range(D):
            expected = lfn - shift_right(rfn, k)
            maxd = max(maxd, float((vol[:, :, k] - expected).abs().max().item()))
        results["semantics_maxdiff"] = maxd
        check("6 cost == Lhat - shift_right(Rhat,k)", maxd == 0.0, "max diff %.3g" % maxd)
    except Exception as e:  # noqa: BLE001
        check("6 cost == Lhat - shift_right(Rhat,k)", False, str(e))

    # 7. no sign negation introduced
    try:
        torch.manual_seed(3)
        sl, sr = torch.randn(1, 3, 2, 7), torch.randn(1, 3, 2, 7)
        lh = normalize_features_l2(sl)
        rh = normalize_features_l2(sr)
        v = build_cost_volume(lh, rh, 4, method="subtract", shift="right")
        d_sub = float((v[:, :, 0] - (lh - rh)).abs().max().item())
        d_add = float((v[:, :, 0] - (lh + rh)).abs().max().item())
        results["sign"] = {"sub_diff": d_sub, "add_diff": d_add}
        check("7 no sign negation", d_sub == 0.0 and d_add > 1e-6,
              "sub %.3g add %.3g" % (d_sub, d_add))
    except Exception as e:  # noqa: BLE001
        check("7 no sign negation", False, str(e))

    # 8. parameter count unchanged vs ARM-V (397954 both)
    try:
        torch.manual_seed(0)
        m_v = StereoNet(arm_v_config())
        torch.manual_seed(0)
        m_z = StereoNet(arm_z_config())
        n_v = sum(p.numel() for p in m_v.parameters())
        n_z = sum(p.numel() for p in m_z.parameters())
        results["params"] = {"arm_v": n_v, "arm_z": n_z}
        check("8 param count unchanged 397954", n_v == 397954 and n_z == 397954,
              "arm_v %d arm_z %d" % (n_v, n_z))
    except Exception as e:  # noqa: BLE001
        check("8 param count unchanged 397954", False, str(e))

    # 9. no BN module anywhere
    try:
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                     for m in model.modules())
        results["has_bn"] = has_bn
        check("9 no BatchNorm anywhere", not has_bn)
    except Exception as e:  # noqa: BLE001
        check("9 no BatchNorm anywhere", False, str(e))

    # 10. gradients propagate through normalization -> aggregation -> refinement
    try:
        mz = StereoNet(arm_z_config())
        mz.train()
        mz.zero_grad(set_to_none=True)
        out = mz(left, right)
        loss = out.abs().mean()
        loss.backward()
        fe_grads = [p.grad for p in mz.feature_extractor.parameters() if p.grad is not None]
        fe_norm = sum(float(g.norm()) for g in fe_grads)
        ag_grads = [p.grad for p in mz.aggregation.parameters() if p.grad is not None]
        ag_norm = sum(float(g.norm()) for g in ag_grads)
        results["grads"] = {"feat_norm": fe_norm, "agg_norm": ag_norm,
                            "loss": float(loss)}
        ok = len(fe_grads) > 0 and fe_norm > 0 and ag_norm > 0
        check("10 gradients reach feature extractor + aggregation", ok,
              "feat %.3g agg %.3g loss %.6f" % (fe_norm, ag_norm, float(loss)))
    except Exception as e:  # noqa: BLE001
        check("10 gradients reach feature extractor + aggregation", False, str(e))

    # 11. shift liveness: slices NOT all identical with normalize on
    try:
        with torch.no_grad():
            diffs = [float((vol[:, :, k] - vol[:, :, 0]).abs().max().item())
                     for k in range(1, D)]
        mmax = max(diffs)
        results["liveness"] = {"max_slice_diff": mmax}
        check("11 shift liveness (slices differ)", mmax > 0.0, "max diff %.6g" % mmax)
    except Exception as e:  # noqa: BLE001
        check("11 shift liveness (slices differ)", False, str(e))

    # Save output
    outdir = REPO_ROOT / "phase1" / "runs" / "arm_z_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "lines": LINES,
               "failures": FAILURES, "pass": not FAILURES}
    with open(outdir / "smoke.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(outdir / "smoke.txt", "w") as f:
        f.write("\n".join(LINES) + "\n")

    if FAILURES:
        print("SMOKE FAILED: %d check(s) failed: %s" % (len(FAILURES), FAILURES), flush=True)
        return 1
    print("SMOKE ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
