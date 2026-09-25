#!/usr/bin/env python
"""Local smoke check for the width-48 twin bundle (no dataset needed).

Builds the model from `bundle_twin` src with the twin config (read from the
bundle copy of `scripts/finetune_pilot.py` exactly as finetune_pilot defines
it: ARM_V_CONFIG parsed as an AST literal, not re-typed), strict-loads the
A2-finalized twin init from `bundle_twin/checkpoints/twin48_init.pth` (sha
verified against `stage_f/followup/twin/preflight_a2.json`), runs the
bundle's `integrity_guard` on the model (must be all_ok), and runs ONE E0
training step (masked Smooth-L1, the loss the twin trains with) on a
synthetic batch (2x3x256x512 random images, disparity random in (0, 184)
with some zeros) on CPU.

Asserts: ARM_V_CONFIG carries feature_channels=48; param count is 891074;
init sha matches preflight_a2 and strict-loads; loss finite with >0 valid
pixels; cross (old-out <- new-in) grad slices nonzero at step 1 (new-output
rows/biases are EXPECTED zero here — structural step-1 block, proven
nonzero at step 2 by finalize_twin_init.py's A2-G2); integrity guard
all_ok; twin literals present verbatim in the patched bundle files; marker
is stage-f-twin48.

Writes `stage_f/followup/kaggle/smoke_twin.json`. No training beyond the
single probe step. No Kaggle.

    python stage_f/followup/kaggle/smoke_twin.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUNDLE = HERE / "bundle_twin"
PATCHED_FINETUNE = BUNDLE / "scripts" / "finetune_pilot.py"
PATCHED_RUN_ARM = BUNDLE / "scripts" / "run_arm.py"
PATCHED_EVAL = BUNDLE / "scripts" / "eval_tier2.py"
TWIN_CKPT = BUNDLE / "checkpoints" / "twin48_init.pth"
PREFLIGHT_A2 = REPO / "stage_f" / "followup" / "twin" / "preflight_a2.json"

TWIN_PARAMS = 891074
OLD_C = 32
TWIN_C = 48


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dict_literal(path: Path, name: str) -> dict:
    """Read a `NAME = dict(...)` / `{...}` assignment as an AST literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name
                for t in node.targets):
            if isinstance(node.value, ast.Dict):
                return ast.literal_eval(node.value)
            if (isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict"
                    and not node.value.args):
                return {kw.arg: ast.literal_eval(kw.value)
                        for kw in node.value.keywords}
    raise SystemExit(f"SMOKE FAIL: {name} not found in {path}")


def main() -> None:
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("smoke bundle_twin (twin arm, LOCAL ONLY)")
    pre = json.loads(PREFLIGHT_A2.read_text(encoding="utf-8"))
    assert pre.get("verdict") == "A2_PASS", pre.get("verdict")
    twin_sha = pre["twin_init"]["sha256"]
    assert twin_sha, "empty twin sha in preflight_a2.json"

    torch.manual_seed(0)
    cfg_dict = read_dict_literal(PATCHED_FINETUNE, "ARM_V_CONFIG")
    assert cfg_dict.get("feature_channels") == 48, cfg_dict
    assert cfg_dict["regression_normalize"] is True, cfg_dict
    print(f"ARM_V_CONFIG: {cfg_dict}")

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
    assert config.feature_channels == 48, config.feature_channels

    ck_sha = sha256(TWIN_CKPT)
    assert ck_sha == twin_sha, f"twin init hash {ck_sha} != {twin_sha}"
    print(f"twin init checkpoint sha: MATCH ({ck_sha[:12]}...)")

    model = StereoNet(config)
    params = sum(p.numel() for p in model.parameters())
    assert params == TWIN_PARAMS, params
    print(f"param count: {params} (expected {TWIN_PARAMS})")
    blob = torch.load(TWIN_CKPT, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    assert set(sd) == set(model.state_dict()), "state_dict keys mismatch on strict load"
    model.load_state_dict(sd, strict=True)
    print("strict init load: OK")
    model.train()

    # One E0 training step on a synthetic batch (CPU, tiny budget).
    B, H, W = 2, 256, 512
    left = torch.randn(B, 3, H, W)
    right = torch.randn(B, 3, H, W)
    disparity = torch.rand(B, 1, H, W) * float(config.max_disparity_px)
    disparity[torch.rand(B, 1, H, W) < 0.2] = 0.0
    out = model(left, right)
    loss, n = masked_smooth_l1(out, disparity,
                               max_disparity=float(config.max_disparity_px))
    assert n > 0, "zero valid pixels in smoke batch"
    assert torch.isfinite(loss).all(), loss
    model.zero_grad(set_to_none=True)
    loss.backward()
    cross_norms, cross_zero, newrow_zero = [], [], []
    for name, p in model.named_parameters():
        g = p.grad
        assert g is not None, name
        if p.dim() == 1:
            continue
        if p.shape[0] == TWIN_C and p.shape[1] == TWIN_C:
            nc = float(g[:OLD_C, OLD_C:].norm())
            nr = float(g[OLD_C:].norm())
            cross_norms.append((name, nc))
            if nc == 0.0:
                cross_zero.append(name)
            if nr == 0.0:
                newrow_zero.append(name)
        elif p.shape[0] == 1 and p.shape[1] == TWIN_C:
            nc = float(g[:, OLD_C:].norm())
            cross_norms.append((name, nc))
            if nc == 0.0:
                cross_zero.append(name)
    assert cross_norms, "no cross slices found"
    assert not cross_zero, f"zero cross grad at step 1: {cross_zero}"
    print(f"step-1 probe: loss={float(loss):.6f} valid_px={n} "
          f"cross_nonzero={len(cross_norms)}/{len(cross_norms)} "
          f"(new-output rows zero here by the structural step-1 block: "
          f"{len(newrow_zero)} rows, expected)")

    # Bundle integrity guard (imported from the bundle copy, run on CPU).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "twin_finetune", PATCHED_FINETUNE)
    # finetune_pilot parses CLI args on import; guard needs REPO_ROOT to be
    # the assembled tree, so call the guard function source directly instead:
    # re-implement the check by exec'ing only integrity_guard with a stub.
    guard_src = PATCHED_FINETUNE.read_text(encoding="utf-8")
    start = guard_src.find("def integrity_guard(model, config) -> dict:")
    assert start != -1, "integrity_guard not found in patched finetune"
    end = guard_src.find("\ndef apply_init(", start)
    assert end != -1, "apply_init not found after integrity_guard"
    guard_fn_src = guard_src[start:end]
    assert "twin48_init.pth" in guard_fn_src, "guard does not reference twin48_init"
    assert "ARM_V_PARAMS" in guard_fn_src, "guard missing param-count gate"
    # Execute the guard with REPO_ROOT pointed at the bundle-assembled layout:
    # the guard loads REPO_ROOT/"checkpoints"/"twin48_init.pth"; emulate by
    # binding REPO_ROOT to a namespace where checkpoints/ mirrors the bundle.
    import types
    ns = {"torch": torch, "Path": Path,
          "REPO_ROOT": BUNDLE,  # bundle/ has checkpoints/ + scripts/ at top level
          "ARM_V_CONFIG": cfg_dict, "ARM_V_PARAMS": TWIN_PARAMS}
    # The bundle checkpoints dir sits at bundle/checkpoints (same relative
    # layout the assembled Kaggle repo will have), so the guard resolves.
    exec(guard_fn_src, ns)  # noqa: S102 - local bundle file, same as smoke_m1's approach
    guard = ns["integrity_guard"](model, config)
    assert guard["param_count"] == TWIN_PARAMS, guard["param_count"]
    assert guard["param_count_ok"] is True, guard
    assert guard["state_dict_keys_match_arm_v"] is True, guard
    assert guard["state_dict_shapes_match_arm_v"] is True, guard
    assert guard["config_matches_arm_v"] is True, guard
    assert guard["all_ok"] is True, guard
    print(f"integrity guard: all_ok (params={guard['param_count']}, "
          f"keys={guard['n_keys']}, shapes_match=True)")

    # Textual asserts: twin literals verbatim in the patched bundle files.
    ft = PATCHED_FINETUNE.read_text(encoding="utf-8")
    ra = PATCHED_RUN_ARM.read_text(encoding="utf-8")
    ev = PATCHED_EVAL.read_text(encoding="utf-8")
    marker = json.loads((BUNDLE / "BUNDLE_MARKER.json").read_text(encoding="utf-8"))
    assert "feature_channels=48" in ft, "width literal missing in finetune"
    assert f"ARM_V_PARAMS = {TWIN_PARAMS}" in ft, "param literal missing in finetune"
    assert "EXPECTED_SHA_TWIN48_INIT" in ra and twin_sha in ra, "twin sha missing in run_arm"
    assert f"EXPECTED_PARAMS = {TWIN_PARAMS}" in ra, "param gate missing in run_arm"
    assert "feature_channels=48" in ev, "width literal missing in eval_tier2"
    assert f"EXPECTED_PARAMS = {TWIN_PARAMS}" in ev, "param gate missing in eval_tier2"
    assert marker.get("bundle") == "stage-f-twin48", marker
    assert marker.get("checkpoint_sha256") == twin_sha, marker
    assert marker.get("param_count") == TWIN_PARAMS, marker
    print("textual diff: width/guard/sha/marker literals verified in bundle files")

    rec = {
        "experiment": "twin bundle smoke (local only, synthetic batch, 1 step)",
        "utc": utc,
        "bundle": "stage_f/followup/kaggle/bundle_twin",
        "marker": marker.get("bundle"),
        "config": cfg_dict,
        "param_count": params,
        "param_expected": TWIN_PARAMS,
        "init_sha256": ck_sha,
        "init_strict_load": True,
        "probe": {"loss": float(loss), "loss_finite": True,
                  "valid_pixels": n,
                  "cross_slices_nonzero": f"{len(cross_norms)}/{len(cross_norms)}",
                  "new_output_rows_zero_step1": len(newrow_zero),
                  "note": "new-output rows zero at step 1 by the structural "
                          "block (A2-G2 proves all 99 nonzero at step 2)"},
        "integrity_guard": {"all_ok": True, "param_count": guard["param_count"],
                            "n_keys": guard["n_keys"]},
        "verdict": "SMOKE PASS",
    }
    (HERE / "smoke_twin.json").write_text(json.dumps(rec, indent=2))
    print(f"loss={float(loss):.6f} n={n}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
