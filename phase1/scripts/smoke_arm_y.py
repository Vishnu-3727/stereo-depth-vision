"""ARM-Y smoke test (preregistration section 10 + liveness section 11).

Run as `python phase1\\scripts\\smoke_arm_y.py`. Prints PASS/FAIL per check
and exits non-zero on any failure. No training, no checkpoint written.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.stereonet import StereoNet, StereoNetConfig

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS" if ok else "FAIL") + " " + name + ((" - " + detail) if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def arm_v_config() -> StereoNetConfig:
    return StereoNetConfig(
        downsample_levels=3,
        num_disparities=24,
        cost_volume_shift="right",
        regression_normalize=True,
    )


def arm_y_config() -> StereoNetConfig:
    cfg = arm_v_config()
    cfg.cost_volume_excitation = True
    return cfg


def main() -> int:
    torch.manual_seed(0)
    B, H0, W0 = 1, 256, 512
    C, D = 32, 24
    stride = 2 ** 3
    Hf, Wf = H0 // stride, W0 // stride

    # 1. model construction (ARM-Y config builds)
    try:
        model = StereoNet(arm_y_config())
        model.train()
        check("1 model construction (ARM-Y config builds)", True)
    except Exception as e:  # noqa: BLE001
        check("1 model construction (ARM-Y config builds)", False, str(e))
        print("aborting: construction failed")
        return 1

    left = torch.randn(B, 3, H0, W0)
    right = torch.randn(B, 3, H0, W0)

    # 2. exact tensor shapes
    try:
        model.eval()
        with torch.no_grad():
            lf = model.feature_extractor(left)
            rf = model.feature_extractor(right)
            vol = model.cost_volume(lf, rf)
            gate_before = torch.sigmoid(model.excitation.conv2(model.excitation.act(model.excitation.conv1(lf))))
            gate_after = gate_before.unsqueeze(2)
            gated = model.excitation(vol, lf)
        ok = (
            tuple(gate_before.shape) == (B, C, Hf, Wf)
            and tuple(gate_after.shape) == (B, C, 1, Hf, Wf)
            and tuple(gated.shape) == (B, C, D, Hf, Wf)
        )
        check(
            "2 exact tensor shapes",
            ok,
            "gate %s gate_u %s gated %s" % (tuple(gate_before.shape), tuple(gate_after.shape), tuple(gated.shape)),
        )
    except Exception as e:  # noqa: BLE001
        check("2 exact tensor shapes", False, str(e))

    # 3. forward pass runs
    model.train()
    try:
        out = model(left, right)
        check("3 forward pass runs", True, "out %s" % (tuple(out.shape),))
    except Exception as e:  # noqa: BLE001
        check("3 forward pass runs", False, str(e))
        print("aborting: forward failed")
        return 1

    # 4. backward pass runs
    try:
        model.zero_grad(set_to_none=True)
        loss = out.abs().mean()
        loss.backward()
        check("4 backward pass runs", True, "loss %.6f" % float(loss))
    except Exception as e:  # noqa: BLE001
        check("4 backward pass runs", False, str(e))

    # 5. no NaNs or Infs in output or gradients
    try:
        bad_out = not torch.isfinite(out).all().item()
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all().item():
                bad_grad = True
                break
        check("5 no NaNs or Infs in output or gradients", (not bad_out) and (not bad_grad))
    except Exception as e:  # noqa: BLE001
        check("5 no NaNs or Infs in output or gradients", False, str(e))

    # 6. gradient reaches ALL of: excitation conv1/conv2, feature extractor, aggregation
    try:
        g1 = model.excitation.conv1.weight.grad
        g2 = model.excitation.conv2.weight.grad
        fe_grads = [p.grad for p in model.feature_extractor.parameters() if p.grad is not None]
        ag_grads = [p.grad for p in model.aggregation.parameters() if p.grad is not None]
        fe_norm = sum(float(g.norm()) for g in fe_grads)
        ag_norm = sum(float(g.norm()) for g in ag_grads)
        ok = (
            g1 is not None and float(g1.norm()) > 0
            and g2 is not None and float(g2.norm()) > 0
            and len(fe_grads) > 0 and fe_norm > 0
            and len(ag_grads) > 0 and ag_norm > 0
        )
        check(
            "6 gradient reaches excitation conv1/conv2, feature extractor, aggregation",
            ok,
            "norms conv1 %.3g conv2 %.3g feat %.3g agg %.3g"
            % (float(g1.norm()) if g1 is not None else -1.0,
               float(g2.norm()) if g2 is not None else -1.0, fe_norm, ag_norm),
        )
    except Exception as e:  # noqa: BLE001
        check("6 gradient reaches excitation conv1/conv2, feature extractor, aggregation", False, str(e))

    # 7. final inference output shape equals input spatial size
    try:
        model.eval()
        with torch.no_grad():
            inf = model(left, right)
        check(
            "7 final inference output shape equals input spatial size",
            tuple(inf.shape) == (B, 1, H0, W0),
            "got %s want %s" % (tuple(inf.shape), (B, 1, H0, W0)),
        )
    except Exception as e:  # noqa: BLE001
        check("7 final inference output shape equals input spatial size", False, str(e))

    # 8. parameter count: ARM-Y total 399,010, delta exactly 1,056
    try:
        torch.manual_seed(0)
        m_v = StereoNet(arm_v_config())
        torch.manual_seed(0)
        m_y = StereoNet(arm_y_config())
        n_v = sum(p.numel() for p in m_v.parameters())
        n_y = sum(p.numel() for p in m_y.parameters())
        delta = n_y - n_v
        check(
            "8 parameter count ARM-Y total 399010 and delta 1056",
            (n_y == 399010) and (delta == 1056),
            "arm_v %d arm_y %d delta %d" % (n_v, n_y, delta),
        )
    except Exception as e:  # noqa: BLE001
        check("8 parameter count ARM-Y total 399010 and delta 1056", False, str(e))

    # 9. ARM-V and ARM-Y outputs differ (same seed, same input)
    try:
        torch.manual_seed(0)
        m_v = StereoNet(arm_v_config())
        torch.manual_seed(0)
        m_y = StereoNet(arm_y_config())
        m_v.eval()
        m_y.eval()
        torch.manual_seed(123)
        l2 = torch.randn(B, 3, H0, W0)
        r2 = torch.randn(B, 3, H0, W0)
        with torch.no_grad():
            o_v = m_v(l2, r2)
            o_y = m_y(l2, r2)
        diff = float((o_v - o_y).abs().max().item())
        check("9 ARM-V and ARM-Y outputs differ", diff > 0.0, "max abs diff %.6g" % diff)
    except Exception as e:  # noqa: BLE001
        check("9 ARM-V and ARM-Y outputs differ", False, str(e))

    # 10. no BatchNorm anywhere in ARM-Y model
    try:
        has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules()
        )
        check("10 no BatchNorm anywhere in ARM-Y model", not has_bn)
    except Exception as e:  # noqa: BLE001
        check("10 no BatchNorm anywhere in ARM-Y model", False, str(e))

    # Gate statistics (DIAGNOSTIC ONLY, not a tuning signal)
    try:
        model.eval()
        stats = []
        with torch.no_grad():
            for s in (0, 1, 2):
                torch.manual_seed(1000 + s)
                lb = torch.randn(B, 3, H0, W0)
                lfb = model.feature_extractor(lb)
                gb = torch.sigmoid(model.excitation.conv2(model.excitation.act(model.excitation.conv1(lfb))))
                stats.append(gb)
            allg = torch.cat(stats, dim=0).float()
        print(
            "DIAGNOSTIC ONLY gate stats over 3 random batches: min %.6f max %.6f mean %.6f std %.6f"
            % (float(allg.min()), float(allg.max()), float(allg.mean()), float(allg.std())),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print("DIAGNOSTIC ONLY gate stats failed: %s" % e, flush=True)

    # Liveness (section 11): shift right vs none must change output (ARM-Y config)
    try:
        torch.manual_seed(0)
        m_r = StereoNet(arm_y_config())  # shift=right
        torch.manual_seed(0)
        cfg_n = arm_y_config()
        cfg_n.cost_volume_shift = "none"
        m_n = StereoNet(cfg_n)
        m_r.eval()
        m_n.eval()
        torch.manual_seed(7)
        ll = torch.randn(B, 3, H0, W0)
        rr = torch.randn(B, 3, H0, W0)
        with torch.no_grad():
            o_r = m_r(ll, rr)
            o_n = m_n(ll, rr)
        ldiff = float((o_r - o_n).abs().max().item())
        check("11 liveness shift right vs none changes output", ldiff > 0.0, "max abs diff %.6g" % ldiff)
    except Exception as e:  # noqa: BLE001
        check("11 liveness shift right vs none changes output", False, str(e))

    if FAILURES:
        print("SMOKE FAILED: %d check(s) failed: %s" % (len(FAILURES), FAILURES), flush=True)
        return 1
    print("SMOKE ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
