#!/usr/bin/env python
"""Local tiny smoke for BOTH L1-continuation bundles (no dataset needed).

Per arm (bundle_l1c, bundle_l1m): builds the model from the bundle src with
the L1 config (ARM_V_CONFIG read from the bundle copy of
`scripts/finetune_pilot.py` exactly as finetune_pilot defines it: AST
literal, not re-typed), strict-loads the E3 seed0 FINAL continuation init
from the bundle's `checkpoints/e3_seed0_final.pth` (sha verified), runs the
arm's actual loss code for ONE training step on a synthetic batch
(2x3x256x512 random images, disparity random in (0, 184) with some zeros) on
CPU, checks the LR schedule starts at 1e-4 (LR literal + a real
Adam/CosineAnnealingLR construction stepped zero times), and runs the
bundle's `integrity_guard` on the model (must be all_ok).

Loss-code handling: the loss call lives inline in finetune_pilot's training
loop (which parses CLI args on import, so it cannot be imported). l1c's
`masked_smooth_l1` is imported from the bundle copy of src (byte-identical
across both bundles — verified by hash-walk here); l1m's bundle-local
`masked_l1` is exec'd from the patched finetune source (same technique as
smoke_twin.py's guard exec). Both arms then textually verify their loss
call line verbatim in the patched file.

Asserts per arm: params 397954; E3 init sha matches and strict-loads; loss
finite with >0 valid pixels; LR literal 1e-4 and scheduler starts at 1e-4
with T_max=EPOCHS (env default 40); run_arm carries the E3 init sha and
40-epoch guards; integrity guard all_ok; marker fields (bundle, checkpoint
sha, epochs 40, lr); l1c has no masked_l1 (control purity); l1m calls
masked_l1.

Writes `stage_f/followup/kaggle/smoke_l1.json`. No training beyond the two
single probe steps. No Kaggle.

    python stage_f/followup/kaggle/smoke_l1.py
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
ARMS = ("l1c", "l1m")

E3_SHA = "82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d"
ARMP_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
PARAMS = 397954


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_name_literal(path: Path, name: str):
    """Read a top-level `NAME = <literal>` assignment as an AST literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name
                for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except Exception:
                raise SystemExit(f"SMOKE FAIL: {name} in {path} is not a literal")
    raise SystemExit(f"SMOKE FAIL: {name} not found in {path}")


def read_dict_literal(path: Path, name: str) -> dict:
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


def hash_walk(root: Path) -> dict:
    # __pycache__ excluded: interpreter byproducts, never part of the bundle.
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def clean_pycache(*roots: Path) -> None:
    for root in roots:
        for d in sorted(root.rglob("__pycache__")):
            if d.is_dir():
                import shutil
                shutil.rmtree(d)


def smoke_arm(arm: str, loss_fn, loss_name: str) -> dict:
    bundle = HERE / f"bundle_{arm}"
    fp = bundle / "scripts" / "finetune_pilot.py"
    rp = bundle / "scripts" / "run_arm.py"
    ckpt = bundle / "checkpoints" / "e3_seed0_final.pth"
    print(f"--- smoke {arm} ({bundle.name}) ---")

    cfg_dict = read_dict_literal(fp, "ARM_V_CONFIG")
    assert "feature_channels" not in cfg_dict, cfg_dict  # 32-ch default, unchanged
    lr = read_name_literal(fp, "LR")
    assert lr == 1e-4, f"LR literal {lr!r} != 1e-4"
    sched = read_name_literal(fp, "SCHEDULER")
    assert "40" in sched, f"SCHEDULER string has no 40: {sched!r}"
    ft = fp.read_text(encoding="utf-8")
    assert 'EPOCHS = int(os.environ.get("P2A_EPOCHS", "40"))' in ft, "EPOCHS default not 40"
    assert "CosineAnnealingLR(optimizer, T_max=EPOCHS)" in ft, "T_max=EPOCHS missing"
    print(f"ARM_V_CONFIG ok; LR={lr}; SCHEDULER={sched!r}")

    from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
    config = StereoNetConfig(**cfg_dict)
    assert config.num_disparities == 24 and config.feature_stride == 8
    assert config.max_disparity_px == 184

    ck_sha = sha256(ckpt)
    assert ck_sha == E3_SHA, f"E3 init hash {ck_sha} != {E3_SHA}"
    print(f"e3-final init checkpoint sha: MATCH ({ck_sha[:12]}...)")

    model = StereoNet(config)
    params = sum(p.numel() for p in model.parameters())
    assert params == PARAMS, params
    print(f"param count: {params} (expected {PARAMS})")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    assert set(sd) == set(model.state_dict()), "state_dict keys mismatch on strict load"
    model.load_state_dict(sd, strict=True)
    print("strict E3-final load: OK")
    model.train()

    # One continuation training step on a synthetic batch (CPU).
    B, H, W = 2, 256, 512
    left = torch.randn(B, 3, H, W)
    right = torch.randn(B, 3, H, W)
    disparity = torch.rand(B, 1, H, W) * float(config.max_disparity_px)
    disparity[torch.rand(B, 1, H, W) < 0.2] = 0.0
    out = model(left, right)
    loss, n = loss_fn(out, disparity,
                      max_disparity=float(config.max_disparity_px))
    assert n > 0, "zero valid pixels in smoke batch"
    assert torch.isfinite(loss).all(), loss
    model.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                if p.grad is not None) ** 0.5
    assert gnorm > 0, "zero gradient on model params"
    print(f"probe: {loss_name} loss={float(loss):.6f} valid_px={n} grad_norm={gnorm:.3e}")

    # LR schedule really starts at 1e-4 (real optimizer + scheduler, 0 steps).
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    lr0 = float(sch.get_last_lr()[0])
    assert lr0 == 1e-4, f"scheduler starts at {lr0}, not 1e-4"
    print(f"lr schedule: starts at {lr0:.1e}, T_max=40")

    # Bundle integrity guard (exec'd from the bundle copy, REPO_ROOT=bundle).
    guard_src = fp.read_text(encoding="utf-8")
    start = guard_src.find("def integrity_guard(model, config) -> dict:")
    assert start != -1, "integrity_guard not found in patched finetune"
    end = guard_src.find("\ndef apply_init(", start)
    assert end != -1, "apply_init not found after integrity_guard"
    ns = {"torch": torch, "Path": Path, "REPO_ROOT": bundle,
          "ARM_V_CONFIG": cfg_dict, "ARM_V_PARAMS": PARAMS}
    exec(guard_src[start:end], ns)  # noqa: S102 - local bundle file
    guard = ns["integrity_guard"](model, config)
    assert guard["param_count"] == PARAMS, guard["param_count"]
    assert guard["all_ok"] is True, guard
    print(f"integrity guard: all_ok (params={guard['param_count']}, keys={guard['n_keys']})")

    # Textual asserts on the patched bundle files.
    ra = rp.read_text(encoding="utf-8")
    marker = json.loads((bundle / "BUNDLE_MARKER.json").read_text(encoding="utf-8"))
    assert "EXPECTED_SHA_E3_SEED0_FINAL" in ra and E3_SHA in ra, "E3 sha missing in run_arm"
    assert "EXPECTED_SHA_ARMP_STAGE1" not in ra, "stale armp sha name in run_arm"
    assert f"EXPECTED_PARAMS = {PARAMS}" in ra
    assert "default=40" in ra and "len(rows) != 40" in ra, "40-epoch guards missing"
    if arm == "l1c":
        assert "masked_l1" not in ft, "control bundle must not contain masked_l1"
        assert "loss, n = masked_smooth_l1(out, disparity" in ft, "control loss call changed"
    else:
        assert "loss, n = masked_l1(out, disparity" in ft, "l1m loss call missing"
        assert "def masked_l1(" in ft, "l1m def missing"
    assert marker.get("bundle") == f"stage-f-{arm}", marker
    assert marker.get("checkpoint_sha256") == E3_SHA, marker
    assert marker.get("epochs") == 40, marker
    print("textual diff: init-sha/epoch/loss/marker literals verified")
    return {"loss": float(loss), "valid_pixels": n, "grad_norm": gnorm,
            "lr_start": lr0, "param_count": params, "init_sha256": ck_sha,
            "guard_all_ok": True}


def main() -> None:
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    torch.manual_seed(0)
    print("smoke L1 bundles (l1c + l1m, LOCAL ONLY)")

    # Both bundles' src trees must be byte-identical (the loss knob lives in
    # the training script only, never in src/): import once, from bundle_l1c.
    hc = hash_walk(HERE / "bundle_l1c" / "src")
    hm = hash_walk(HERE / "bundle_l1m" / "src")
    assert hc == hm, "bundle src trees differ between arms"
    print(f"src trees byte-identical across arms ({len(hc)} files)")
    sys.path.insert(0, str(HERE / "bundle_l1c"))
    from src.losses.disparity import masked_smooth_l1  # noqa: E402
    import src as _src_check  # noqa: E402
    assert Path(_src_check.__file__).resolve().is_relative_to(
        (HERE / "bundle_l1c").resolve()), _src_check.__file__

    # l1m's bundle-local masked_l1, exec'd from the patched finetune source.
    ft_m = (HERE / "bundle_l1m" / "scripts" / "finetune_pilot.py").read_text(
        encoding="utf-8")
    start = ft_m.find("\ndef masked_l1(")
    assert start != -1, "masked_l1 def not found in l1m finetune"
    end = ft_m.find("\n# DELTA-1 vs P2A", start)
    assert end != -1, "DELTA-1 comment not found after masked_l1"
    import torch.nn.functional as F  # noqa: E402
    ns_m = {"torch": torch, "F": F}
    exec(ft_m[start:end], ns_m)  # noqa: S102 - local bundle file
    masked_l1 = ns_m["masked_l1"]

    rec = {"experiment": "L1 bundle smoke (local only, synthetic batch, 1 step/arm)",
           "utc": utc,
           "e3_init_sha256": E3_SHA,
           "src_identical_across_arms": True,
           "arms": {}}
    rec["arms"]["l1c"] = smoke_arm("l1c", masked_smooth_l1, "smooth-l1")
    rec["arms"]["l1m"] = smoke_arm("l1m", masked_l1, "masked-l1")
    rec["verdict"] = "SMOKE PASS"
    (HERE / "smoke_l1.json").write_text(json.dumps(rec, indent=2))
    clean_pycache(HERE / "bundle_l1c", HERE / "bundle_l1m")
    print("SMOKE PASS (both arms)")


if __name__ == "__main__":
    main()
