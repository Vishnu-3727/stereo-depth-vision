#!/usr/bin/env python
"""Twin G2 gradient probe ONLY (LOCAL ONLY, no training).

Task: run the twin preflight G2 gradient check only. G1 stays FAILED as
recorded (see g1_diagnosis.json); this script does not touch G1 and does not
declare the preflight passed.

Authoritative spec: docs/superpowers/specs/2026-09-24-stage-f-followup-design.md
section 6 (G2: a 1-step gradient is nonzero on the new slices).

Method (imports build_twin_init.py functions; that file is never edited):
- Build the embedded twin EXACTLY as build_twin_init.py does: ARM-P Stage-1
  source checkpoint (sha-verified by bti.load_source), StereoNetConfig from
  bti.F1_CFG with ONLY feature_channels 32 -> 48, torch.manual_seed(
  bti.TWIN_SEED), bti.embed_arm_p_into_twin. fp32, GPU, train mode.
- One forward+backward on real E0-recipe training crops with the E0-recipe
  Smooth-L1 loss and valid mask (masked_smooth_l1, beta=1.0,
  max_disparity=184, as the F1 arms train): batch 2 of 256x512
  scale-augmented crops (ScaledCroppedKitti logic, seed 0). No optimizer step.
- For every parameter tensor with new-channel slices: grad norm on each new
  slice, grad norm on the old block, nonzero flag per new slice. Loss value
  and finiteness reported.
- G2 PASS iff every layer with new slices has a nonzero new-slice grad. Any
  exactly-zero new slice is listed (with structural explanation where one
  applies, e.g. a new output channel whose only consumer is a zeroed cross
  weight). The cross-slice-only variant (build_twin_init.py's G2 definition)
  is reported alongside for reference.

Outputs (only): run_g2.py (this file), g2.json, g2.log.
Prohibitions: no edit of existing files, no twin48_init.pth, no git, no
Kaggle, no optimizer step, no G1 gate change.

Usage (from repo root, with the torch python):
  python stage_f/followup/twin/run_g2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
TWIN_DIR = REPO / "stage_f" / "followup" / "twin"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TWIN_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import build_twin_init as bti  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

# Frozen E0 recipe (same values as build_twin_init.e0_style_crop and the F1 arms).
CROP_H, CROP_W = 256, 512
SCALE_LO, SCALE_HI = 0.7, 1.7
JITTER_SIGMA = 0.1
CROP_SEED = 0
BATCH = 2
DEVICE = "cuda"


def e0_sample(base, idx: int, rng, device: str):
    """One real E0-recipe training crop (ScaledCroppedKitti logic).

    Identical steps to build_twin_init.e0_style_crop, but draws from a shared
    rng so a batch of B samples consumes sequential draws exactly as the F1
    dataset would (sample 0 == build_twin_init's single probe crop).
    Returns (left, right, disp) CPU tensors + provenance dict.
    """
    import cv2  # local import: only needed for this probe

    smp = base[idx]
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
    left = left * (1.0 + rng.normal(0, JITTER_SIGMA))
    right = right * (1.0 + rng.normal(0, JITTER_SIGMA))
    prov = {"scene": smp.name, "split": "hailo_calib", "index": idx,
            "scale_draw_s": s, "crop_hw": [h, w], "crop_yx": [y, x],
            "realised_sx": sx, "crop_out": [CROP_H, CROP_W],
            "jitter_sigma": JITTER_SIGMA}
    lt = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32))
    rt = torch.from_numpy(np.ascontiguousarray(right, dtype=np.float32))
    dt = torch.from_numpy(np.ascontiguousarray(disp_c[None], dtype=np.float32))
    return lt, rt, dt, prov


def e0_batch2(device: str):
    """Batch-2 E0-recipe training input: hailo_calib scenes 0,1, seed 0."""
    base = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib",
                           disparity_scale=256.0, occluded=True)
    assert len(base) == 160 and base.names[0] == "000000_10.png", (
        len(base), base.names[:2])
    rng = np.random.default_rng(CROP_SEED)
    ls, rs, ds, provs = [], [], [], []
    for i in range(BATCH):
        lt, rt, dt, prov = e0_sample(base, i, rng, device)
        ls.append(lt)
        rs.append(rt)
        ds.append(dt)
        provs.append(prov)
    left = torch.stack(ls).to(device)    # (B,3,256,512)
    right = torch.stack(rs).to(device)
    disp = torch.stack(ds).to(device)    # (B,1,256,512)
    return left, right, disp, provs


def g2_probe_batch(twin: StereoNet, left, right, disp) -> dict:
    """One forward+backward (NO optimizer step); per-layer new/old grad norms.

    Slicing convention is exactly build_twin_init.g2_probe's, extended with
    per-slice nonzero flags and a strict task verdict (every new slice
    nonzero), plus the cross-slice-only variant for reference.
    """
    twin.train()
    twin.zero_grad(set_to_none=True)
    out = twin(left, right)
    loss, n = masked_smooth_l1(out, disp,
                               max_disparity=float(twin.config.max_disparity_px))
    assert n > 0, "G2 probe batch has zero valid pixels"
    finite = bool(torch.isfinite(loss).item())
    assert finite, f"G2 probe loss not finite: {loss}"
    loss.backward()  # probe only: no optimizer step follows

    oc = bti.OLD_C
    layers: dict = {}
    for name, p in twin.named_parameters():
        g = p.grad
        assert g is not None, name
        if p.dim() == 1:  # bias over output channels
            if p.shape[0] == oc or p.shape[0] == 1:
                layers[name] = {"kind": "unwidened_bias",
                                "shape": list(p.shape),
                                "new_slices": {},
                                "grad_norm_all": float(g.norm())}
            else:
                assert p.shape[0] == bti.TWIN_C, (name, tuple(p.shape))
                nn, no = float(g[oc:].norm()), float(g[:oc].norm())
                layers[name] = {"kind": "bias_split",
                                "shape": list(p.shape),
                                "old_norm": no,
                                "new_slices": {
                                    "bias_new_out": {"norm": nn,
                                                     "nonzero": bool(nn > 0.0)}}}
            continue
        assert p.dim() in (4, 5), (name, tuple(p.shape))
        ic_old = oc
        # Unwidened in-channels (first extractor conv Cin=3, refinement in Cin=4).
        if p.shape[1] not in (oc, bti.TWIN_C):
            ic_old = p.shape[1]
        widened_out = (p.shape[0] == bti.TWIN_C)
        widened_in = (p.shape[1] == bti.TWIN_C)
        rec: dict = {"kind": "conv_split", "shape": list(p.shape),
                     "new_slices": {}}
        if widened_in and not widened_out:
            # Final 1-channel layers: only the cross slice is new.
            nc, ob = float(g[:, oc:].norm()), float(g[:, :ic_old].norm())
            rec["new_slices"]["cross_old_out_new_in"] = {
                "norm": nc, "nonzero": bool(nc > 0.0)}
            rec["old_norm"] = ob
        elif widened_out and widened_in:
            nc = float(g[:oc, oc:].norm())
            nr = float(g[oc:].norm())
            ob = float(g[:oc, :oc].norm())
            rec["new_slices"]["cross_old_out_new_in"] = {
                "norm": nc, "nonzero": bool(nc > 0.0)}
            rec["new_slices"]["new_out_rows"] = {
                "norm": nr, "nonzero": bool(nr > 0.0)}
            rec["old_norm"] = ob
        elif widened_out and not widened_in:
            nr, ob = float(g[oc:].norm()), float(g[:oc].norm())
            rec["new_slices"]["new_out_rows"] = {
                "norm": nr, "nonzero": bool(nr > 0.0)}
            rec["old_norm"] = ob
        else:
            rec["grad_norm_all"] = float(g.norm())
        layers[name] = rec

    zero_list = [{"param": k, "slice": s, "norm": d["norm"]}
                 for k, v in layers.items()
                 for s, d in v.get("new_slices", {}).items()
                 if not d["nonzero"]]
    n_new_slices = sum(len(v.get("new_slices", {})) for v in layers.values())
    n_layers_with_new = sum(1 for v in layers.values() if v.get("new_slices"))
    assert n_new_slices > 0, "no new slices found — embed rule covered nothing?"
    cross_vals = [d["norm"] for v in layers.values()
                  for s, d in v.get("new_slices", {}).items()
                  if s == "cross_old_out_new_in"]
    cross_pass = bool(cross_vals and all(v > 0.0 for v in cross_vals))
    strict_pass = bool(n_new_slices > 0 and not zero_list)
    return {"loss": float(loss), "loss_finite": finite, "valid_pixels": n,
            "per_layer": layers,
            "n_layers_with_new_slices": n_layers_with_new,
            "n_new_slices": n_new_slices,
            "n_cross_slices": len(cross_vals),
            "zero_list": zero_list,
            "pass_strict_all_new_nonzero": strict_pass,
            "pass_cross_nonzero": cross_pass}


def main() -> None:
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TWIN_DIR / "g2.log"
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
        _run()
    finally:
        sys.stdout = sys.stdout.fhs[0]
        log_fh.close()


def _run() -> None:
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    assert torch.cuda.is_available(), "G2 probe requires GPU (fp32 on cuda)"
    device = DEVICE
    print(f"twin G2 probe ONLY (LOCAL ONLY, no optimizer step); device={device}",
          flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    # Embedded twin built EXACTLY as build_twin_init.py does.
    src_path, old_sd = bti.load_source()
    print(f"source init: {src_path} sha MATCH {bti.SOURCE_SHA} "
          f"keys={len(old_sd)}", flush=True)
    twin_cfg = StereoNetConfig(**{**bti.F1_CFG, "feature_channels": bti.TWIN_C})
    torch.manual_seed(bti.TWIN_SEED)  # deterministic default init (new channels)
    twin = StereoNet(twin_cfg)
    params = sum(p.numel() for p in twin.parameters())
    print(f"twin params={params} (record only; range gate belongs to preflight)",
          flush=True)
    actions = bti.embed_arm_p_into_twin(old_sd, twin)
    n_zeroed = sum(1 for a in actions.values() if a.get("zeroed_cross"))
    print(f"embed: {len(actions)} tensors; "
          f"conv layers with zeroed cross slice={n_zeroed}", flush=True)
    twin.to(device)  # fp32 throughout; train mode set in the probe

    # One forward+backward on a real batch-2 E0-recipe training input.
    left, right, disp, provs = e0_batch2(device)
    print(f"batch: B={BATCH} crop={CROP_H}x{CROP_W} seed={CROP_SEED} "
          f"scenes={[p['scene'] for p in provs]}", flush=True)
    g2 = g2_probe_batch(twin, left, right, disp)
    g2["batch"] = {"size": BATCH, "crop_hw": [CROP_H, CROP_W],
                   "crop_seed": CROP_SEED, "samples": provs}
    print(f"G2 loss={g2['loss']:.6f} finite={g2['loss_finite']} "
          f"valid_px={g2['valid_pixels']} "
          f"layers_with_new={g2['n_layers_with_new_slices']} "
          f"new_slices={g2['n_new_slices']} "
          f"strict_all_new_nonzero={g2['pass_strict_all_new_nonzero']} "
          f"cross_nonzero={g2['pass_cross_nonzero']}", flush=True)
    print("per-layer new/old grad norms:", flush=True)
    for k in sorted(g2["per_layer"]):
        v = g2["per_layer"][k]
        ns = ", ".join(f"{s}={d['norm']:.3e}("
                       f"{'nonzero' if d['nonzero'] else 'ZERO'})"
                       for s, d in v.get("new_slices", {}).items())
        old = (f"old={v['old_norm']:.3e}" if "old_norm" in v
               else f"all={v['grad_norm_all']:.3e}")
        print(f"  G2 {k} [{v['kind']} {v['shape']}]: new: {ns}; {old}",
              flush=True)
    if g2["zero_list"]:
        print(f"zero-grad new slices ({len(g2['zero_list'])}):", flush=True)
        for z in g2["zero_list"]:
            print(f"  ZERO {z['param']} :: {z['slice']} norm={z['norm']}",
                  flush=True)
    else:
        print("zero-grad new slices: none", flush=True)

    verdict = "PASS" if g2["pass_strict_all_new_nonzero"] else "FAIL"
    print(f"G2 verdict (strict, per task): {verdict} "
          f"(cross-only reference: "
          f"{'PASS' if g2['pass_cross_nonzero'] else 'FAIL'})", flush=True)
    print("G1 untouched (stays FAILED as recorded); preflight verdict "
          "not declared here; no init file written", flush=True)

    out = {
        "experiment": "twin G2 gradient probe only (local only, no optimizer step)",
        "spec": "docs/superpowers/specs/2026-09-24-stage-f-followup-design.md "
                "section 6",
        "provenance": {
            "utc": utc, "device": device,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "source_path": str(src_path), "source_sha256": bti.SOURCE_SHA,
            "f1_cfg": bti.F1_CFG, "old_channels": bti.OLD_C,
            "twin_channels": bti.TWIN_C, "twin_seed": bti.TWIN_SEED,
            "batch": BATCH, "crop_hw": [CROP_H, CROP_W],
            "crop_seed": CROP_SEED, "dtype": "float32", "mode": "train"},
        "param_count": params,
        "g2": g2,
        "verdict": ("G2_PASS" if verdict == "PASS" else "G2_FAIL"),
        "structural_note": (
            "Net2Net-style zeroing implies the incoming weights/biases of the "
            "NEW output units get exactly-zero gradient on step 1: every path "
            "from the loss to a new-channel activation passes through a zeroed "
            "from-new-to-old cross weight (exactly 0 at the final 1-channel "
            "layers), while the zeroed cross slices themselves see nonzero "
            "activations and nonzero old-output grads, hence nonzero grads."),
        "scope": ("G2 only; G1 stays FAILED as recorded in preflight.json / "
                  "g1_diagnosis.json; no twin48_init.pth written; no optimizer "
                  "step taken."),
    }
    bti.atomic_write_json(TWIN_DIR / "g2.json", out)
    print(f"wrote {TWIN_DIR / 'g2.json'} verdict={out['verdict']}", flush=True)
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
