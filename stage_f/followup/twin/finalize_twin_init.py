#!/usr/bin/env python
"""Finalize the width-48 capacity-twin init under amendment A2 (LOCAL ONLY).

Authoritative spec: docs/superpowers/specs/2026-09-24-stage-f-followup-design.md
section 6, as amended by docs/superpowers/specs/2026-09-25-stage-f-followup-amendment-A2.md.
User approval for A2 recorded 2026-09-25, before any twin training.

Method (imports build_twin_init.py; that file and every other existing file
are never edited):
- Build the embedded twin EXACTLY as build_twin_init.py does: ARM-P Stage-1
  source checkpoint (sha-verified by bti.load_source), StereoNetConfig from
  bti.F1_CFG with ONLY feature_channels 32 -> 48,
  torch.manual_seed(bti.TWIN_SEED), bti.embed_arm_p_into_twin.
- A2-G1 (fp64 CPU): ARM-P and twin both .double() on CPU, eval mode, same
  float64 inputs, first 3 contract hailo_val scenes. PASS iff per-scene final
  disparity max-abs diff <= 1e-5 (tolerance unchanged).
- A2-G2 (fp32 GPU): train mode. Step 1 forward+backward on a real batch-2
  E0-recipe training input (hailo_calib scenes 0,1, 256x512 scale-augmented
  crops, seed 0 — same draws as run_g2.py, whose sample 0 equals
  build_twin_init.e0_style_crop's probe crop); record every cross
  (old-out <- new-in) slice grad norm. One optimizer step of the E0 recipe
  (Adam lr 1e-3, betas (0.9, 0.999), as finetune_pilot.py trains). Step 2
  forward+backward on the SAME batch; record every new slice (all 99:
  cross + new_out_rows + bias_new_out). PASS iff all 33 cross slices nonzero
  at step 1 AND all 99 new slices nonzero at step 2.
- ONLY if both PASS: write twin48_init.pth as {"model": state_dict,
  "config": {...}} — the same wrapping the E0 bundle's armp_stage1_best.pth
  uses (keys "model" + "config"), so apply_init's blob["model"] strict-load
  path works unchanged — plus preflight_a2.json. Else STOP, no pth, exit 1.

Outputs (only): finalize_twin_init.py (this file), twin48_init.pth (gates
pass only), preflight_a2.json, finalize.log. No training beyond the two G2
probe steps + one optimizer step. No Kaggle. No git writes (HEAD read only
for provenance, as build_twin_init.py does).

Usage (from repo root, with the torch python):
  python stage_f/followup/twin/finalize_twin_init.py
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

# Frozen A2 constants (amendment A2 section 2).
G1_TOL = 1e-5
G2_LR = 1e-3
G2_BETAS = (0.9, 0.999)
CROP_H, CROP_W = 256, 512
SCALE_LO, SCALE_HI = 0.7, 1.7
JITTER_SIGMA = 0.1
CROP_SEED = 0
BATCH = 2
EXPECTED_PARAMS = 891074


def e0_sample(base, idx: int, rng, device: str):
    """One real E0-recipe training crop (ScaledCroppedKitti logic, seed 0).

    Identical steps to build_twin_init.e0_style_crop / run_g2.e0_sample:
    sample 0 == build_twin_init's single probe crop. Returns CPU tensors +
    provenance; the caller moves the stacked batch to device.
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


def e0_batch2():
    """Batch-2 E0-recipe training input: hailo_calib scenes 0,1, seed 0."""
    base = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib",
                           disparity_scale=256.0, occluded=True)
    assert len(base) == 160 and base.names[0] == "000000_10.png", (
        len(base), base.names[:2])
    rng = np.random.default_rng(CROP_SEED)
    ls, rs, ds, provs = [], [], [], []
    for i in range(BATCH):
        lt, rt, dt, prov = e0_sample(base, i, rng, "cpu")
        ls.append(lt)
        rs.append(rt)
        ds.append(dt)
        provs.append(prov)
    left = torch.stack(ls)
    right = torch.stack(rs)
    disp = torch.stack(ds)
    return left, right, disp, provs


def slice_grads(twin: StereoNet) -> dict:
    """Per-layer new/old grad norms with run_g2.py's slicing convention."""
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
        if p.shape[1] not in (oc, bti.TWIN_C):
            ic_old = p.shape[1]
        widened_out = (p.shape[0] == bti.TWIN_C)
        widened_in = (p.shape[1] == bti.TWIN_C)
        rec: dict = {"kind": "conv_split", "shape": list(p.shape),
                     "new_slices": {}}
        if widened_in and not widened_out:
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
    return layers


def summarize(layers: dict) -> dict:
    zero_list = [{"param": k, "slice": s, "norm": d["norm"]}
                 for k, v in layers.items()
                 for s, d in v.get("new_slices", {}).items()
                 if not d["nonzero"]]
    n_new = sum(len(v.get("new_slices", {})) for v in layers.values())
    n_layers = sum(1 for v in layers.values() if v.get("new_slices"))
    cross_vals = [d["norm"] for v in layers.values()
                  for s, d in v.get("new_slices", {}).items()
                  if s == "cross_old_out_new_in"]
    return {"n_layers_with_new_slices": n_layers,
            "n_new_slices": n_new,
            "n_cross_slices": len(cross_vals),
            "zero_list": zero_list,
            "pass_cross_nonzero": bool(cross_vals and all(v > 0.0 for v in cross_vals)),
            "pass_all_new_nonzero": bool(n_new > 0 and not zero_list)}


def a2_g1() -> list[dict]:
    """A2-G1: fp64 CPU exact-embed check on the same 3 scenes. Returns recs."""
    old_cfg = StereoNetConfig(**bti.F1_CFG)
    twin_cfg = StereoNetConfig(**{**bti.F1_CFG, "feature_channels": bti.TWIN_C})
    src_path, old_sd = bti.load_source()
    torch.manual_seed(bti.TWIN_SEED)
    twin_f = StereoNet(twin_cfg)
    bti.embed_arm_p_into_twin(old_sd, twin_f)
    arm_d = StereoNet(old_cfg)
    arm_d.load_state_dict(old_sd, strict=True)
    arm_d.eval().double().to("cpu")
    twin_d = twin_f.double().to("cpu")
    twin_d.eval()
    del twin_f
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert len(ds) == 40 and ds.names[0] == "000160_10.png", (len(ds), ds.names[:2])
    recs = []
    for i in range(3):
        s = ds[i]
        l = torch.from_numpy(normalize(s.left)).double().to("cpu")
        r = torch.from_numpy(normalize(s.right)).double().to("cpu")
        with torch.no_grad():
            out_o, st_o = arm_d(l, r, return_stages=True)
            out_n, st_n = twin_d(l, r, return_stages=True)
        oc = bti.OLD_C
        pairs = {
            "left_features": (st_o["left_features"], st_n["left_features"][:, :oc]),
            "cost_volume": (st_o["cost_volume"], st_n["cost_volume"][:, :oc]),
            "aggregated_cost": (st_o["aggregated_cost"], st_n["aggregated_cost"]),
            "disparity_initial": (st_o["disparity_initial"], st_n["disparity_initial"]),
            "disparity_final": (out_o, out_n),
        }
        stages = {}
        for k, (a, b) in pairs.items():
            d = float((a - b).abs().max())
            m = float(a.abs().max())
            stages[k] = {"max_abs_diff": d, "arm_magnitude": m,
                         "rel": (d / m if m > 0 else 0.0)}
        final = stages["disparity_final"]["max_abs_diff"]
        rec = {"scene": s.name, "index": i, "stages": stages,
               "final_max_abs": final, "pass": bool(final <= G1_TOL)}
        recs.append(rec)
        print(f"A2-G1 scene {s.name}: " +
              " ".join(f"{k}={stages[k]['max_abs_diff']:.3e}" for k in stages) +
              f" -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
        del l, r, out_o, out_n, st_o, st_n
    del arm_d, twin_d
    return recs


def a2_g2() -> dict:
    """A2-G2: step-1 cross-nonzero + one Adam step + step-2 all-nonzero."""
    assert torch.cuda.is_available(), "A2-G2 requires GPU (fp32 on cuda)"
    device = "cuda"
    twin_cfg = StereoNetConfig(**{**bti.F1_CFG, "feature_channels": bti.TWIN_C})
    src_path, old_sd = bti.load_source()
    torch.manual_seed(bti.TWIN_SEED)
    twin = StereoNet(twin_cfg)
    params = sum(p.numel() for p in twin.parameters())
    print(f"twin params={params} (expected {EXPECTED_PARAMS})", flush=True)
    assert params == EXPECTED_PARAMS, (params, EXPECTED_PARAMS)
    bti.embed_arm_p_into_twin(old_sd, twin)
    twin.to(device)

    left_c, right_c, disp_c, provs = e0_batch2()
    left, right, disp = left_c.to(device), right_c.to(device), disp_c.to(device)
    print(f"batch: B={BATCH} crop={CROP_H}x{CROP_W} seed={CROP_SEED} "
          f"scenes={[p['scene'] for p in provs]}", flush=True)

    # Step 1 (fp32 GPU, train mode, E0 Smooth-L1, no optimizer yet).
    twin.train()
    twin.zero_grad(set_to_none=True)
    out1 = twin(left, right)
    loss1, n1 = masked_smooth_l1(out1, disp,
                                 max_disparity=float(twin.config.max_disparity_px))
    assert n1 > 0 and bool(torch.isfinite(loss1).item()), (n1, float(loss1))
    loss1.backward()
    layers1 = slice_grads(twin)
    sum1 = summarize(layers1)
    nz1 = sum1["n_cross_slices"] - sum(
        1 for z in sum1["zero_list"] if z["slice"] == "cross_old_out_new_in")
    print(f"A2-G2 step1: loss={float(loss1):.6f} valid_px={n1} "
          f"cross_nonzero={nz1}/{sum1['n_cross_slices']} "
          f"pass_cross={sum1['pass_cross_nonzero']}", flush=True)
    del out1

    # One optimizer step of the E0 recipe.
    opt = torch.optim.Adam(twin.parameters(), lr=G2_LR, betas=G2_BETAS)
    opt.step()
    print(f"A2-G2 optimizer step: Adam lr={G2_LR} betas={list(G2_BETAS)} "
          f"on the step-1 batch", flush=True)

    # Step 2 on the SAME real batch.
    twin.zero_grad(set_to_none=True)
    out2 = twin(left, right)
    loss2, n2 = masked_smooth_l1(out2, disp,
                                 max_disparity=float(twin.config.max_disparity_px))
    assert n2 > 0 and bool(torch.isfinite(loss2).item()), (n2, float(loss2))
    loss2.backward()
    layers2 = slice_grads(twin)
    sum2 = summarize(layers2)
    print(f"A2-G2 step2: loss={float(loss2):.6f} valid_px={n2} "
          f"new_nonzero={sum2['n_new_slices'] - len(sum2['zero_list'])}/"
          f"{sum2['n_new_slices']} pass_all={sum2['pass_all_new_nonzero']}",
          flush=True)
    if sum2["zero_list"]:
        for z in sum2["zero_list"]:
            print(f"  STILL-ZERO step2 {z['param']} :: {z['slice']} "
                  f"norm={z['norm']}", flush=True)
    else:
        print("  step2 zero slices: none (all 99 new slices nonzero)", flush=True)
    del out2, left, right, disp
    torch.cuda.empty_cache()
    return {"loss1": float(loss1), "valid_pixels1": n1,
            "loss2": float(loss2), "valid_pixels2": n2,
            "optimizer": {"name": "Adam", "lr": G2_LR, "betas": list(G2_BETAS),
                          "steps": 1, "batch": "same real batch-2 as step 1"},
            "batch": {"size": BATCH, "crop_hw": [CROP_H, CROP_W],
                      "crop_seed": CROP_SEED, "samples": provs},
            "step1": {"per_layer": layers1, **sum1},
            "step2": {"per_layer": layers2, **sum2},
            "pass": bool(sum1["pass_cross_nonzero"] and sum2["pass_all_new_nonzero"])}


def main() -> None:
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TWIN_DIR / "finalize.log"
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
    print(f"twin finalize under amendment A2 (LOCAL ONLY); "
          f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"gpu={'none' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}",
          flush=True)
    src_path, _ = bti.load_source()
    print(f"source init: {src_path} sha MATCH {bti.SOURCE_SHA}", flush=True)

    twin_cfg = StereoNetConfig(**{**bti.F1_CFG, "feature_channels": bti.TWIN_C})
    torch.manual_seed(bti.TWIN_SEED)
    probe = StereoNet(twin_cfg)
    params = sum(p.numel() for p in probe.parameters())
    del probe
    print(f"param count={params} expected={EXPECTED_PARAMS} "
          f"{'OK' if params == EXPECTED_PARAMS else 'MISMATCH'}", flush=True)

    g1_recs = a2_g1()
    g1_pass = bool(all(r["pass"] for r in g1_recs) and params == EXPECTED_PARAMS)

    g2 = a2_g2()
    g2_pass = bool(g2["pass"])

    verdict = "PASS" if (g1_pass and g2_pass) else "FAIL"
    twin_sha = None
    written = False
    if verdict == "PASS":
        # Same wrapping as the E0 bundle's armp_stage1_best.pth
        # (keys "model" + "config") so apply_init's blob["model"]
        # strict-load path works unchanged.
        torch.manual_seed(bti.TWIN_SEED)
        twin = StereoNet(twin_cfg)
        _, old_sd = bti.load_source()
        bti.embed_arm_p_into_twin(old_sd, twin)
        sd = {k: v.cpu().clone() for k, v in twin.state_dict().items()}
        ckpt = {"model": sd,
                "config": {**bti.F1_CFG, "feature_channels": bti.TWIN_C,
                           "source_sha256": bti.SOURCE_SHA,
                           "param_count": params,
                           "amendment": "A2",
                           "spec": "docs/superpowers/specs/2026-09-25-stage-f-followup-amendment-A2.md"}}
        out_p = TWIN_DIR / "twin48_init.pth"
        torch.save(ckpt, out_p)
        twin_sha = bti.sha256_file(out_p)
        written = True
        print(f"saved {out_p} sha256={twin_sha}", flush=True)
        del twin, sd
    else:
        print(f"STOP: A2 gates g1_pass={g1_pass} g2_pass={g2_pass}; "
              f"init file NOT written", flush=True)

    out = {
        "experiment": "capacity-twin width-48 init finalized under amendment A2 (local only)",
        "spec": "docs/superpowers/specs/2026-09-24-stage-f-followup-design.md section 6",
        "amendment": "docs/superpowers/specs/2026-09-25-stage-f-followup-amendment-A2.md",
        "approval": "user-approved 2026-09-25 before any twin training",
        "provenance": {
            "utc": utc, "git_head": bti.git_head(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "g1_device": "cpu", "g1_dtype": "float64",
            "g2_device": "cuda", "g2_dtype": "float32",
            "source_path": str(src_path), "source_sha256": bti.SOURCE_SHA,
            "f1_cfg": bti.F1_CFG, "old_channels": bti.OLD_C,
            "twin_channels": bti.TWIN_C, "twin_seed": bti.TWIN_SEED},
        "param_count": params,
        "param_range": [800000, 1000000],
        "param_ok": params == EXPECTED_PARAMS,
        "g1": {"tolerance": G1_TOL, "scenes": g1_recs, "pass": g1_pass},
        "g2": {**g2, "pass": g2_pass},
        "twin_init": {"path": "stage_f/followup/twin/twin48_init.pth",
                      "sha256": twin_sha, "written": written,
                      "format": "{'model': state_dict, 'config': {...}} "
                                "(same wrapping as armp_stage1_best.pth)"},
        "verdict": ("A2_PASS" if verdict == "PASS" else "A2_FAIL"),
    }
    bti.atomic_write_json(TWIN_DIR / "preflight_a2.json", out)
    print(f"wrote {TWIN_DIR / 'preflight_a2.json'} verdict={out['verdict']}",
          flush=True)
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
