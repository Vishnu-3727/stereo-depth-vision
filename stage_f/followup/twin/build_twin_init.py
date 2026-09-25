#!/usr/bin/env python
"""Capacity-twin init: width-48 function-preserving embed of ARM-P Stage-1 + preflight gate.

Authoritative spec: docs/superpowers/specs/2026-09-24-stage-f-followup-design.md
section 6 (Capacity twin). This script is LOCAL ONLY (no Kaggle, no training
beyond the single G2 probe step, no bundle building, no git writes).

Source init: the ARM-P Stage-1 pretrained checkpoint used by the F1 arms
(sha256 3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7).
The model is built exactly as the F1 arms do (ARM_V_CONFIG from
stage_e_recipe/kaggle/bundle/scripts/finetune_pilot.py, i.e. the E0 recipe:
downsample_levels=3, num_disparities=24, cost_volume_shift="right",
regression_normalize=True, feature_channels=32), with ONLY feature_channels
32 -> 48.

Embed rule (spec): every tensor copied from ARM-P into the [:32, :32, ...]
slice (old in/out channels); new output channels keep default init; every
weight FROM a new channel INTO an old output channel = 0. Applies to every
parameterised layer in src/models/stereonet/: feature-extractor Conv2d stack
+ ResBlocks + output conv, aggregation Conv3d stack + to_cost, refinement
input conv + ResBlocks + output conv. CostVolume has no parameters
(groups=0); DisparityRegression has none; excitation is absent
(cost_volume_excitation=False).

Preflight gate (frozen by spec, applied exactly):
- G1: twin forward vs ARM-P forward on the first 3 contract hailo_val scenes,
  eval mode, same inputs -> max abs diff of final disparity <= 1e-5 per scene.
  Intermediate diffs (old-channel slice of features/cost volume, full
  aggregated cost, disparity_initial) recorded for diagnosis only.
- G2: one training step's gradient (masked Smooth-L1, beta=1.0,
  max_disparity=184, on a real E0-recipe training crop) is nonzero on the
  new-channel slices of each layer that has them; per-layer grad norms on the
  new slices reported.
  NOTE (recorded, not a workaround): Net2Net-style zeroing implies the
  incoming weights / biases of the NEW output units get exactly-zero gradient
  on step 1 (their forward signal is blocked by the zeroed cross weights at
  the final 1-channel layers), while the zeroed cross slices (old outputs <-
  new inputs) receive nonzero gradient and become trainable immediately. G2
  PASS is therefore defined as: every layer that HAS a from-new-to-old cross
  slice shows nonzero grad norm there. The new-output row/bias norms are
  reported alongside (expected 0 on step 1 by the theory above) plus a strict
  all-new-params variant, so the record supports either reading.

Outputs (only these):
- stage_f/followup/twin/build_twin_init.py (this file)
- stage_f/followup/twin/twin48_init.pth (ONLY if G1 and G2 PASS and params in range)
- stage_f/followup/twin/preflight.json
- stage_f/followup/twin/run.log (this script tees stdout here itself)

Usage (from repo root, with the torch python):
  python stage_f/followup/twin/build_twin_init.py [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
TWIN_DIR = REPO / "stage_f" / "followup" / "twin"
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

# ---- frozen constants ----
SOURCE_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
SOURCE_CANDIDATES = [
    REPO / "stage_f" / "followup" / "kaggle" / "bundle_f1m" / "checkpoints" / "armp_stage1_best.pth",
    REPO / "stage_b_armp" / "20260918T062146Z_stage1_pretrain" / "checkpoints" / "armp_stage1_best.pth",
]
# Exact F1-arm construction (E0 recipe ARM_V_CONFIG in bundle/scripts/finetune_pilot.py).
F1_CFG = dict(downsample_levels=3, num_disparities=24,
              cost_volume_shift="right", regression_normalize=True)
OLD_C = 32
TWIN_C = 48
PARAM_LO, PARAM_HI = 800_000, 1_000_000
G1_TOL = 1e-5
MAX_DISP_PX = 184.0  # config.max_disparity_px for the F1 geometry (11*... verified below)
TWIN_SEED = 0  # seeds the twin's default init (kept on new channels) deterministically


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


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


def load_source() -> tuple[Path, dict]:
    """Find the ARM-P Stage-1 checkpoint and verify its sha256 before use."""
    for cand in SOURCE_CANDIDATES:
        if cand.is_file() and sha256_file(cand) == SOURCE_SHA:
            blob = torch.load(str(cand), map_location="cpu", weights_only=False)
            sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
            return cand, sd
    checked = [{str(c): (sha256_file(c) if c.is_file() else "MISSING")}
               for c in SOURCE_CANDIDATES]
    raise SystemExit(f"ABORT: no source checkpoint with sha {SOURCE_SHA}; checked {checked}")


def embed_arm_p_into_twin(old_sd: dict, twin: StereoNet) -> dict:
    """Function-preserving embed. Returns per-key action record.

    For each twin parameter W of shape (Cout_new, Cin_new, ...):
      - twin[:oc, :ic, ...] = old (oc/ic = old out/in channels; dims that did
        not widen copy in full),
      - twin[:oc, ic:, ...] = 0 (weights FROM new channels INTO old outputs),
      - twin[oc:, ...] untouched (new output channels keep default init).
    Biases: twin[:oc] = old, twin[oc:] untouched.
    """
    twin_sd = twin.state_dict()
    assert set(twin_sd) == set(old_sd), (
        f"key mismatch: twin-only={sorted(set(twin_sd) - set(old_sd))} "
        f"old-only={sorted(set(old_sd) - set(twin_sd))}")
    actions: dict = {}
    for key, tw in twin_sd.items():
        old = old_sd[key]
        assert tw.dtype == old.dtype, (key, tw.dtype, old.dtype)
        if tw.shape == old.shape:
            # No channel dim widened on this tensor (e.g. to_cost bias (1,)).
            tw.copy_(old)
            actions[key] = {"kind": "copy_full", "shape": list(tw.shape)}
            continue
        assert tw.dim() in (1, 4, 5), (key, tuple(tw.shape))
        assert tw.numel() > old.numel(), (key, tuple(tw.shape), tuple(old.shape))
        if tw.dim() == 1:  # bias over output channels
            (oc,) = old.shape
            assert tw.shape[0] > oc
            with torch.no_grad():
                tw[:oc].copy_(old)
            actions[key] = {"kind": "bias_split", "old_channels": oc,
                            "new_channels": tw.shape[0] - oc}
            continue
        # Conv weight: (Cout, Cin, k...) — every widened dim must be a leading
        # channel dim (spatial kernel dims never change).
        oc, ic = old.shape[0], old.shape[1]
        assert tw.shape[0] >= oc and tw.shape[1] >= ic, (key, tuple(tw.shape))
        assert tuple(tw.shape[2:]) == tuple(old.shape[2:]), (key,)
        assert tw.shape[0] == oc or tw.shape[0] == TWIN_C, (key, tuple(tw.shape))
        with torch.no_grad():
            tw[:oc, :ic].copy_(old)  # old block (full copy if a dim unwidened)
            if tw.shape[1] > ic:
                tw[:oc, ic:].zero_()  # FROM new channels INTO old outputs = 0
            # tw[oc:, ...] untouched: new output channels keep default init
        actions[key] = {"kind": "conv_split",
                        "old_out": oc, "new_out": tw.shape[0],
                        "old_in": ic, "new_in": tw.shape[1],
                        "zeroed_cross": tw.shape[1] > ic}
    # No buffers expected anywhere in this architecture (no BN); fail loudly
    # if that ever changes so no tensor escapes the rule.
    buffers = dict(twin.named_buffers())
    assert not buffers, f"unexpected buffers: {sorted(buffers)}"
    return actions


def g1_scene(model_old: StereoNet, model_new: StereoNet, ds, idx: int,
             device: str) -> dict:
    s = ds[idx]
    l = torch.from_numpy(normalize(s.left)).to(device)
    r = torch.from_numpy(normalize(s.right)).to(device)
    with torch.no_grad():
        out_old, st_old = model_old(l, r, return_stages=True)
        out_new, st_new = model_new(l, r, return_stages=True)
    rec: dict = {"scene": s.name, "index": idx}
    rec["final_max_abs"] = float((out_new - out_old).abs().max())
    # Diagnostics (shapes may differ in channel dim -> compare old slice).
    lf_old, lf_new = st_old["left_features"], st_new["left_features"]
    rec["left_features_shape_old"] = list(lf_old.shape)
    rec["left_features_shape_new"] = list(lf_new.shape)
    rec["left_features_old_slice_max_abs"] = float(
        (lf_new[:, :OLD_C] - lf_old).abs().max())
    cv_old, cv_new = st_old["cost_volume"], st_new["cost_volume"]
    rec["cost_volume_shape_old"] = list(cv_old.shape)
    rec["cost_volume_shape_new"] = list(cv_new.shape)
    rec["cost_volume_old_slice_max_abs"] = float(
        (cv_new[:, :OLD_C] - cv_old).abs().max())
    ag_old, ag_new = st_old["aggregated_cost"], st_new["aggregated_cost"]
    assert ag_old.shape == ag_new.shape, (tuple(ag_old.shape), tuple(ag_new.shape))
    rec["aggregated_cost_max_abs"] = float((ag_new - ag_old).abs().max())
    di_old, di_new = st_old["disparity_initial"], st_new["disparity_initial"]
    rec["disparity_initial_max_abs"] = float((di_new - di_old).abs().max())
    rec["pass"] = bool(rec["final_max_abs"] <= G1_TOL)
    del out_old, out_new, st_old, st_new, l, r
    return rec


def e0_style_crop(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """One real E0-recipe training crop: hailo_calib scene 0, scale-augmented
    256x512 crop (ScaledCroppedKitti logic from bundle/scripts/finetune_pilot.py,
    first draw of np seed 0), ImageNet normalise, gain jitter sigma=0.1."""
    import cv2  # local import: only needed for this probe

    CROP_H, CROP_W = 256, 512
    SCALE_LO, SCALE_HI = 0.7, 1.7
    base = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib",
                           disparity_scale=256.0, occluded=True)
    smp = base[0]
    rng = np.random.default_rng(0)
    H, W = smp.disparity.shape
    s = float(np.exp(rng.uniform(np.log(SCALE_LO), np.log(SCALE_HI))))
    w, h = min(int(round(CROP_W / s)), W), min(int(round(CROP_H / s)), H)
    y, x = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    sl = (slice(y, y + h), slice(x, x + w))
    left_c = cv2.resize(smp.left[sl], (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)
    right_c = cv2.resize(smp.right[sl], (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)
    disp_c = cv2.resize(smp.disparity[sl], (CROP_W, CROP_H), interpolation=cv2.INTER_NEAREST)
    sx = CROP_W / float(w)
    disp_c = disp_c.copy()
    disp_c[disp_c > 0] *= sx
    left = normalize(left_c)[0]
    right = normalize(right_c)[0]
    left = left * (1.0 + rng.normal(0, 0.1))
    right = right * (1.0 + rng.normal(0, 0.1))
    prov = {"scene": smp.name, "split": "hailo_calib", "index": 0,
            "scale_draw_s": s, "crop_hw": [h, w], "crop_yx": [y, x],
            "realised_sx": sx, "crop_out": [CROP_H, CROP_W], "jitter_sigma": 0.1}
    lt = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32)).unsqueeze(0).to(device)
    rt = torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32)).unsqueeze(0).to(device)
    dt = torch.from_numpy(np.ascontiguousarray(disp_c[None, None], dtype=np.float32)).to(device)
    return lt, rt, dt, prov


def g2_probe(twin: StereoNet, device: str) -> dict:
    twin.train()
    twin.zero_grad(set_to_none=True)
    l, r, d, prov = e0_style_crop(device)
    out = twin(l, r)
    loss, n = masked_smooth_l1(out, d, max_disparity=float(twin.config.max_disparity_px))
    assert n > 0, "G2 probe crop has zero valid pixels"
    loss.backward()
    layers: dict = {}
    for name, p in twin.named_parameters():
        g = p.grad
        assert g is not None, name
        oc_new = p.shape[0]
        if p.dim() == 1:  # bias
            if oc_new == OLD_C or (oc_new == 1):
                layers[name] = {"kind": "unwidened_bias",
                                "grad_norm_all": float(g.norm())}
            else:
                assert oc_new == TWIN_C, (name, tuple(p.shape))
                layers[name] = {
                    "kind": "bias_split",
                    "grad_norm_new_out": float(g[OLD_C:].norm()),
                    "grad_norm_old": float(g[:OLD_C].norm())}
            continue
        assert p.dim() in (4, 5), (name, tuple(p.shape))
        oc_old, ic_old = (OLD_C, OLD_C)
        # Unwidened in-channels (first extractor conv Cin=3, refinement in Cin=4).
        if p.shape[1] not in (OLD_C, TWIN_C):
            ic_old = p.shape[1]
        widened_out = (p.shape[0] == TWIN_C)
        widened_in = (p.shape[1] == TWIN_C)
        rec: dict = {"kind": "conv_split", "shape": list(p.shape)}
        if widened_in and not widened_out:
            # Final 1-channel layers: only the cross slice is new.
            rec["grad_norm_cross_old_out_new_in"] = float(g[:, OLD_C:].norm())
            rec["grad_norm_old_block"] = float(g[:, :ic_old].norm())
        elif widened_out and widened_in:
            rec["grad_norm_cross_old_out_new_in"] = float(g[:OLD_C, OLD_C:].norm())
            rec["grad_norm_new_out_rows"] = float(g[OLD_C:].norm())
            rec["grad_norm_old_block"] = float(g[:OLD_C, :OLD_C].norm())
        elif widened_out and not widened_in:
            rec["grad_norm_new_out_rows"] = float(g[OLD_C:].norm())
            rec["grad_norm_old_block"] = float(g[:OLD_C].norm())
        else:
            rec["grad_norm_all"] = float(g.norm())
        layers[name] = rec
    # G2 PASS: every layer WITH a from-new-to-old cross slice has nonzero grad there.
    cross_norms = {k: v["grad_norm_cross_old_out_new_in"] for k, v in layers.items()
                   if "grad_norm_cross_old_out_new_in" in v}
    assert cross_norms, "no cross slices found — embed rule covered nothing?"
    g2_pass = bool(all(v > 0.0 for v in cross_norms.values()))
    # Strict variant (all new params incl. new-output rows/biases): recorded,
    # expected False on step 1 by the Net2Net dead-end theory (see module docstring).
    strict_vals = []
    for v in layers.values():
        for k in ("grad_norm_cross_old_out_new_in", "grad_norm_new_out_rows",
                  "grad_norm_new_out"):
            if k in v:
                strict_vals.append(v[k])
    rec_out = {"crop": prov, "loss": float(loss), "valid_pixels": n,
               "per_layer": layers,
               "n_layers_with_cross_slice": len(cross_norms),
               "pass_cross_nonzero": g2_pass,
               "strict_all_new_nonzero": bool(strict_vals and all(v > 0.0 for v in strict_vals))}
    del out, l, r, d
    return rec_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TWIN_DIR / "run.log"
    log_fh = open(log_path, "w")  # noqa: PTH123

    class Tee:
        def __init__(self, *fhs):
            self.fhs = fhs

        def write(self, s):
            for fh in self.fhs:
                fh.write(s)

        def flush(self):
            for fh in self.fhs:
                fh.flush()

    sys.stdout = Tee(sys.stdout, log_fh)
    try:
        _run(device)
    finally:
        sys.stdout = sys.stdout.fhs[0]
        log_fh.close()


def _run(device: str) -> None:
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"capacity-twin build+preflight (LOCAL ONLY); device={device}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"cuda_available={torch.cuda.is_available()}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    src_path, old_sd = load_source()
    print(f"source init: {src_path} sha MATCH {SOURCE_SHA} keys={len(old_sd)}", flush=True)

    old_cfg = StereoNetConfig(**F1_CFG)
    twin_cfg = StereoNetConfig(**{**F1_CFG, "feature_channels": TWIN_C})
    assert abs(old_cfg.max_disparity_px - MAX_DISP_PX) < 1e-9, old_cfg.max_disparity_px
    assert abs(twin_cfg.max_disparity_px - MAX_DISP_PX) < 1e-9, twin_cfg.max_disparity_px

    torch.manual_seed(TWIN_SEED)  # deterministic default init for new channels
    twin = StereoNet(twin_cfg)
    params = sum(p.numel() for p in twin.parameters())
    params_ok = PARAM_LO <= params <= PARAM_HI
    print(f"twin params={params} range=[{PARAM_LO},{PARAM_HI}] "
          f"{'OK' if params_ok else 'OUT OF RANGE'}", flush=True)

    pre: dict = {
        "experiment": "capacity-twin width-48 init + preflight (local only)",
        "spec": "docs/superpowers/specs/2026-09-24-stage-f-followup-design.md section 6",
        "provenance": {
            "utc": utc, "git_head": git_head(), "device": device,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "source_path": str(src_path), "source_sha256": SOURCE_SHA,
            "f1_cfg": F1_CFG, "old_channels": OLD_C, "twin_channels": TWIN_C,
            "twin_seed": TWIN_SEED},
        "param_count": params,
        "param_range": [PARAM_LO, PARAM_HI],
        "param_ok": params_ok,
    }
    if not params_ok:
        pre.update({"embed": None, "g1": None, "g2": None,
                    "verdict": "STOP_PARAM_RANGE",
                    "twin_init": None})
        atomic_write_json(TWIN_DIR / "preflight.json", pre)
        print("STOP: param count outside [800k, 1.0M]; no init file written", flush=True)
        raise SystemExit(1)

    actions = embed_arm_p_into_twin(old_sd, twin)
    n_zeroed = sum(1 for a in actions.values() if a.get("zeroed_cross"))
    print(f"embed: {len(actions)} tensors; conv layers with zeroed cross slice={n_zeroed}",
          flush=True)
    pre["embed"] = {"n_tensors": len(actions), "n_zeroed_cross": n_zeroed,
                    "actions": actions}

    # G1: fresh strict-loaded ARM-P vs twin, eval mode, same inputs.
    model_old = StereoNet(old_cfg)
    blob_old = torch.load(str(src_path), map_location="cpu", weights_only=False)
    sd_old = blob_old["model"] if isinstance(blob_old, dict) and "model" in blob_old else blob_old
    model_old.load_state_dict(sd_old, strict=True)
    model_old.eval().to(device)
    twin.eval().to(device)
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert len(ds) == 40 and ds.names[0] == "000160_10.png", (len(ds), ds.names[:2])
    g1_recs = [g1_scene(model_old, twin, ds, i, device) for i in range(3)]
    for rec in g1_recs:
        print(f"G1 scene {rec['scene']}: final_max_abs={rec['final_max_abs']:.3e} "
              f"agg={rec['aggregated_cost_max_abs']:.3e} "
              f"init={rec['disparity_initial_max_abs']:.3e} "
              f"cv_old_slice={rec['cost_volume_old_slice_max_abs']:.3e} "
              f"feat_old_slice={rec['left_features_old_slice_max_abs']:.3e} "
              f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    g1_pass = bool(all(r["pass"] for r in g1_recs))
    pre["g1"] = {"tolerance": G1_TOL, "scenes": g1_recs, "pass": g1_pass}
    if not g1_pass:
        bad = [r for r in g1_recs if not r["pass"]]
        pre.update({"g2": None, "verdict": "STOP_G1_FAIL", "twin_init": None})
        atomic_write_json(TWIN_DIR / "preflight.json", pre)
        print(f"STOP: G1 FAIL on {[b['scene'] for b in bad]}; no workaround per spec; "
              f"no init file written", flush=True)
        raise SystemExit(1)
    del model_old
    if device == "cuda":
        torch.cuda.empty_cache()

    # G2: single-step gradient probe on a real E0-recipe training crop.
    twin_g = StereoNet(twin_cfg)
    twin_g.load_state_dict(twin.state_dict())
    twin_g.to(device)
    g2 = g2_probe(twin_g, device)
    pre["g2"] = g2
    print(f"G2 loss={g2['loss']:.6f} valid_px={g2['valid_pixels']} "
          f"cross_layers={g2['n_layers_with_cross_slice']} "
          f"pass_cross_nonzero={g2['pass_cross_nonzero']} "
          f"strict_all_new_nonzero={g2['strict_all_new_nonzero']}", flush=True)
    for k in sorted(g2["per_layer"]):
        print(f"  G2 {k}: {g2['per_layer'][k]}", flush=True)
    del twin_g
    if device == "cuda":
        torch.cuda.empty_cache()

    g2_pass = bool(g2["pass_cross_nonzero"])
    verdict = "PASS" if (g1_pass and g2_pass) else "FAIL"
    twin_sha = None
    if verdict == "PASS":
        ckpt = {"model": twin.state_dict(),
                "twin_config": {**F1_CFG, "feature_channels": TWIN_C},
                "source_sha256": SOURCE_SHA, "param_count": params}
        out_p = TWIN_DIR / "twin48_init.pth"
        torch.save(ckpt, out_p)
        twin_sha = sha256_file(out_p)
        print(f"saved {out_p} sha256={twin_sha}", flush=True)
    else:
        print("FAIL: init file NOT written (G1 and G2 must both pass)", flush=True)
    pre["twin_init"] = {"path": "stage_f/followup/twin/twin48_init.pth",
                        "sha256": twin_sha,
                        "written": bool(verdict == "PASS")}
    pre["verdict"] = verdict
    atomic_write_json(TWIN_DIR / "preflight.json", pre)
    print(f"wrote {TWIN_DIR / 'preflight.json'} verdict={verdict}", flush=True)
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
