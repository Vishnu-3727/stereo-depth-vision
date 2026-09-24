#!/usr/bin/env python
"""Local smoke check for the M1-patched bundle (no dataset needed).

Builds the model from `bundle_f1m` src with the E0/E3 config (read from the
bundle copy of `scripts/finetune_pilot.py` exactly as finetune_pilot defines
it), loads the ARM-P init checkpoint from `bundle_f1m/checkpoints` (sha
verified), and runs ONE training step of the patched loss code on a synthetic
batch (2x3x256x512 random images, disparity random in (0, 184) with some
zeros) on CPU.

Asserts: loss finite, ce finite and > 0, gradient on aggregation params
nonzero, q rows sum to 1, and the M1 term's gradient reaches aggregated_cost.

The smoke re-states the few patched loss lines (the step lives inline in
main(), so it cannot be imported) and then diffs them textually against the
patched file: every re-stated line must appear verbatim in
`bundle_f1m/scripts/finetune_pilot.py`.

    python stage_f/followup/kaggle/smoke_m1.py [--lam {0.1,1.0}]

--lam 0.1 (default) smokes bundle_f1m/ and asserts the `0.1` literal;
--lam 1.0 smokes bundle_f1m10/ and asserts the `1.0` literal.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAM_BUNDLE = {0.1: "bundle_f1m", 1.0: "bundle_f1m10"}
BUNDLE = HERE / "bundle_f1m"
PATCHED_FILE = BUNDLE / "scripts" / "finetune_pilot.py"
EXPECTED_INIT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_arm_v_config() -> dict:
    """Read ARM_V_CONFIG from the bundle copy, exactly as finetune_pilot
    defines it (parsed AST literal, not a re-typed copy)."""
    tree = ast.parse(PATCHED_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ARM_V_CONFIG"
                for t in node.targets):
            if isinstance(node.value, ast.Dict):  # {...} literal
                return ast.literal_eval(node.value)
            if (isinstance(node.value, ast.Call)  # dict(...) call, as in finetune_pilot
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict"
                    and not node.value.args):
                return {kw.arg: ast.literal_eval(kw.value)
                        for kw in node.value.keywords}
    raise SystemExit("SMOKE FAIL: ARM_V_CONFIG not found in finetune_pilot.py")


def parse_args(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.1,
                    choices=(0.1, 1.0))
    return ap.parse_args(argv).lam


def main(lam: float = 0.1) -> None:
    global BUNDLE, PATCHED_FILE
    lam = float(lam)
    if lam not in LAM_BUNDLE:
        raise SystemExit(f"SMOKE FAIL: --lam must be 0.1 or 1.0, got {lam!r}")
    BUNDLE = HERE / LAM_BUNDLE[lam]
    PATCHED_FILE = BUNDLE / "scripts" / "finetune_pilot.py"
    lam_literal = "0.1" if lam == 0.1 else "1.0"
    print(f"smoke lam={lam_literal} bundle={LAM_BUNDLE[lam]}")
    torch.manual_seed(0)
    cfg_dict = read_arm_v_config()
    assert cfg_dict["regression_normalize"] is True, cfg_dict

    sys.path.insert(0, str(BUNDLE))
    from src.losses.disparity import masked_smooth_l1  # noqa: E402
    from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
    import src as _src_check  # noqa: E402
    assert Path(_src_check.__file__).resolve().is_relative_to(BUNDLE.resolve()), \
        f"imported src is not the bundle copy: {_src_check.__file__}"

    config = StereoNetConfig(**cfg_dict)
    assert config.num_disparities == 24, config.num_disparities
    assert config.feature_stride == 8, config.feature_stride
    assert config.max_disparity_px == 184, config.max_disparity_px

    ckpt = BUNDLE / "checkpoints" / "armp_stage1_best.pth"
    ck_sha = sha256(ckpt)
    assert ck_sha == EXPECTED_INIT_SHA, f"init hash {ck_sha}"
    print(f"init checkpoint sha: MATCH ({ck_sha[:12]}...)")

    model = StereoNet(config)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd, strict=True)
    model.train()

    B, H, W = 2, 256, 512
    left = torch.randn(B, 3, H, W)
    right = torch.randn(B, 3, H, W)
    disparity = torch.rand(B, 1, H, W) * float(config.max_disparity_px)
    disparity[torch.rand(B, 1, H, W) < 0.2] = 0.0  # sparse GT: some zeros

    # --- patched training-step lines, re-stated (see module docstring) ---
    out, _st = model(left, right, return_stages=True)
    loss, n = masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))
    _m1_cost = _st["aggregated_cost"]
    _m1_cost.retain_grad()  # smoke-only: observe the gradient at the cost
    _m1_up = torch.nn.functional.interpolate(_m1_cost, size=disparity.shape[-2:], mode="bilinear", align_corners=True)
    _m1_norm = (_m1_up - _m1_up.mean(1, keepdim=True)) / (_m1_up.std(1, keepdim=True) + 1e-6)
    _m1_logp = torch.nn.functional.log_softmax(-_m1_norm, dim=1)
    _m1_valid = (disparity > 0) & (disparity < float(config.max_disparity_px))
    _m1_t = disparity / float(config.feature_stride)
    _m1_k = torch.arange(config.num_disparities, dtype=_m1_up.dtype, device=_m1_up.device).view(1, -1, 1, 1)
    _m1_q = torch.softmax(-torch.abs(_m1_k - _m1_t) / 0.5, dim=1)
    _m1_ce_map = -(_m1_q * _m1_logp).sum(1, keepdim=True)
    _m1_ce = _m1_ce_map[_m1_valid].mean() if int(_m1_valid.sum()) > 0 else _m1_up.sum() * 0.0
    # Executed via float(lam) (bit-identical to the literal); the literal
    # itself is asserted textually below (`loss = loss + <lam> * _m1_ce`).
    loss = loss + float(lam) * _m1_ce
    # --- end of re-stated lines ---

    assert torch.isfinite(loss).all(), loss
    assert torch.isfinite(_m1_ce).all(), _m1_ce
    assert float(_m1_ce) > 0, float(_m1_ce)
    q_sum = _m1_q.sum(1)
    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-5), \
        f"q rows do not sum to 1: min {q_sum.min()} max {q_sum.max()}"

    model.zero_grad(set_to_none=True)
    loss.backward()
    agg_grads = [p.grad for p in model.aggregation.parameters()
                 if p.grad is not None]
    assert agg_grads, "no gradients on aggregation params"
    gnorm = sum(float(g.pow(2).sum()) for g in agg_grads) ** 0.5
    assert gnorm > 0, "zero gradient on aggregation params"
    assert _m1_cost.grad is not None, "M1 gradient did not reach aggregated_cost"
    assert float(_m1_cost.grad.pow(2).sum()) > 0, "zero M1 gradient at aggregated_cost"

    # Textual diff: every re-stated line must appear verbatim in the patched file.
    text = PATCHED_FILE.read_text(encoding="utf-8")
    restated = [
        "            out, _st = model(left, right, return_stages=True)",
        "            loss, n = masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))",
        '            _m1_cost = _st["aggregated_cost"]',
        "            _m1_up = F.interpolate(_m1_cost, size=disparity.shape[-2:], mode=\"bilinear\", align_corners=True)",
        "            _m1_norm = (_m1_up - _m1_up.mean(1, keepdim=True)) / (_m1_up.std(1, keepdim=True) + 1e-6)",
        "            _m1_logp = F.log_softmax(-_m1_norm, dim=1)",
        "            _m1_valid = (disparity > 0) & (disparity < float(config.max_disparity_px))",
        "            _m1_t = disparity / float(config.feature_stride)",
        "            _m1_k = torch.arange(config.num_disparities, dtype=_m1_up.dtype, device=_m1_up.device).view(1, -1, 1, 1)",
        "            _m1_q = torch.softmax(-torch.abs(_m1_k - _m1_t) / 0.5, dim=1)",
        "            _m1_ce_map = -(_m1_q * _m1_logp).sum(1, keepdim=True)",
        "            _m1_ce = _m1_ce_map[_m1_valid].mean() if int(_m1_valid.sum()) > 0 else _m1_up.sum() * 0.0",
        f"            loss = loss + {lam_literal} * _m1_ce",
    ]
    # The smoke spells torch.nn.functional in full; the patched file imports it
    # as F, so compare modulo that alias.
    norm = text.replace("torch.nn.functional", "F")
    for line in restated:
        assert line in norm, f"SMOKE FAIL: line not in patched file: {line!r}"
    assert "import torch.nn.functional as F" in text, "F import missing"
    assert '"m1_ce": epoch_ce / max(batches, 1)' in text, "m1_ce log row missing"

    print(f"config: num_disparities=24 feature_stride=8 max_disparity_px=184 "
          f"regression_normalize=True")
    print(f"cost_std={float(_m1_cost.std()):.3e} (synthetic input is off-manifold; "
          f"both standardised softmaxes suppress grads by ~1/std, so the cost "
          f"grad below is tiny but must be present, i.e. the path is connected)")
    print(f"loss={float(loss):.6f} smoothl1_n={n} ce={float(_m1_ce):.6f} "
          f"q_rowsum_min={float(q_sum.min()):.6f} q_rowsum_max={float(q_sum.max()):.6f}")
    print(f"aggregation grad norm={gnorm:.3e} "
          f"cost grad norm={float(_m1_cost.grad.pow(2).sum()) ** 0.5:.3e}")
    print(f"textual diff: {len(restated)} lines verified in patched file")
    print("SMOKE PASS")


if __name__ == "__main__":
    main(parse_args())
