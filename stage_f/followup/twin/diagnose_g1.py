#!/usr/bin/env python
"""G1 preflight failure diagnosis (LOCAL ONLY, diagnosis only).

Answers: is the width-48 embed WRONG, or correct with the fp32 mismatch
explained by float32 rounding amplified by huge activations?

Reuses stage_f/followup/twin/build_twin_init.py functions (import only;
that file and preflight.json are never edited). Same 3 scenes as G1
(first 3 contract hailo_val scenes), eval mode, same inputs.

1. Float32 noise floor: ARM-P width-32 run twice on GPU, and ARM-P GPU vs CPU.
   Record max abs diff per stage.
2. Float64 on CPU: ARM-P and twin both .double(), same float64 inputs.
   Per stage (left features old slice, cost volume old slice, aggregation
   output, init disparity, final disparity): max abs diff, max abs magnitude
   of the ARM-P tensor, relative diff = diff / magnitude.
3. Also in float64: max abs of twin NEW feature channels and NEW cost-volume
   channels (new channels must not leak into old outputs).
4. If float64 final disparity diff still > 1e-5: forward hooks to find the
   FIRST layer where the old slice diverges in float64, with reason.

Outputs (only): diagnose_g1.py (this file), g1_diagnosis.json,
g1_diagnosis.log. No twin48_init.pth, no gate change, no training.

Usage (from repo root):
  python stage_f/followup/twin/diagnose_g1.py
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

import torch  # noqa: E402

import build_twin_init as bti  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

STAGES = ["left_features", "cost_volume", "aggregated_cost",
          "disparity_initial", "disparity_final"]
G1_TOL = 1e-5


def _stages_fp32(model, l, r):
    with torch.no_grad():
        out, st = model(l, r, return_stages=True)
    rec = {
        "left_features": st["left_features"].detach(),
        "cost_volume": st["cost_volume"].detach(),
        "aggregated_cost": st["aggregated_cost"].detach(),
        "disparity_initial": st["disparity_initial"].detach(),
        "disparity_final": out.detach(),
    }
    return rec


def _maxabs(a, b) -> float:
    return float((a - b).abs().max())


def fp32_noise_floor(arm_gpu, arm_cpu, ds, gpu: str) -> list[dict]:
    """ARM-P vs itself: GPU run twice + GPU vs CPU, per stage, per scene."""
    recs = []
    for i in range(3):
        s = ds[i]
        lg = torch.from_numpy(normalize(s.left)).to(gpu)
        rg = torch.from_numpy(normalize(s.right)).to(gpu)
        lc = torch.from_numpy(normalize(s.left)).to("cpu")
        rc = torch.from_numpy(normalize(s.right)).to("cpu")
        with torch.no_grad():
            a = _stages_fp32(arm_gpu, lg, rg)
            b = _stages_fp32(arm_gpu, lg, rg)
            c = _stages_fp32(arm_cpu, lc, rc)
        gg = {k: _maxabs(a[k], b[k]) for k in STAGES}
        gc = {k: float((a[k].to("cpu") - c[k]).abs().max()) for k in STAGES}
        # magnitude reference (GPU run, fp32) for scale context
        mag = {k: float(a[k].abs().max()) for k in STAGES}
        recs.append({"scene": s.name, "index": i,
                     "gpu_twice_max_abs": gg,
                     "gpu_vs_cpu_max_abs": gc,
                     "magnitude_fp32": mag})
        print(f"noise scene {s.name}: gpu2x=" +
              " ".join(f"{k}={gg[k]:.3e}" for k in STAGES), flush=True)
        print(f"noise scene {s.name}: gpu/cpu=" +
              " ".join(f"{k}={gc[k]:.3e}" for k in STAGES), flush=True)
        del lg, rg, lc, rc, a, b, c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return recs


def fp64_compare(arm_d, twin_d, ds) -> tuple[list[dict], dict | None]:
    """Twin vs ARM-P in float64 on CPU. Returns (per-scene recs, hook info)."""
    recs = []
    first_div = None
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
        per = {}
        for k, (a, b) in pairs.items():
            d = float((a - b).abs().max())
            m = float(a.abs().max())
            per[k] = {"max_abs_diff": d, "arm_magnitude": m,
                      "rel": (d / m if m > 0 else 0.0)}
        new_feat = float(st_n["left_features"][:, oc:].abs().max())
        new_cv = float(st_n["cost_volume"][:, oc:].abs().max())
        per["twin_new_left_features_max_abs"] = new_feat
        per["twin_new_cost_volume_max_abs"] = new_cv
        recs.append({"scene": s.name, "index": i, "stages": per})
        print(f"fp64 scene {s.name}: " +
              " ".join(f"{k}={per[k]['max_abs_diff']:.3e}(rel {per[k]['rel']:.3e})"
                       for k in STAGES) +
              f" newfeat={new_feat:.3e} newcv={new_cv:.3e}", flush=True)
        if per["disparity_final"]["max_abs_diff"] > G1_TOL and first_div is None:
            first_div = find_first_divergence(arm_d, twin_d, l, r, s.name)
        del l, r, out_o, out_n, st_o, st_n
    return recs, first_div


def find_first_divergence(arm_d, twin_d, l, r, scene: str) -> dict:
    """Forward hooks on every Conv2d/Conv3d; first old-slice mismatch in fp64."""
    conv_names: list[str] = []
    for name, mod in arm_d.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv3d)):
            conv_names.append(name)
    old_acts: dict[str, list] = {n: [] for n in conv_names}
    twin_acts: dict[str, list] = {n: [] for n in conv_names}

    def mk(store, n):
        def h(_m, _inp, out):
            store[n].append(out.detach().clone())
        return h

    ho, ht = [], []
    for n, m in arm_d.named_modules():
        if n in old_acts:
            ho.append(m.register_forward_hook(mk(old_acts, n)))
    for n, m in twin_d.named_modules():
        if n in twin_acts:
            ht.append(m.register_forward_hook(mk(twin_acts, n)))
    try:
        with torch.no_grad():
            arm_d(l, r)
            twin_d(l, r)
    finally:
        for h in ho + ht:
            h.remove()
    oc = bti.OLD_C
    for n in conv_names:
        to = twin_d.get_submodule(n)
        wo = dict(arm_d.named_parameters())[n + ".weight"]
        wt = dict(twin_d.named_parameters())[n + ".weight"]
        no = len(old_acts[n])
        nt = len(twin_acts[n])
        assert no == nt and no >= 1, (n, no, nt)
        # feature extractor fires twice (left+right shared tower): check both,
        # report the first diverging call.
        for call in range(no):
            a = old_acts[n][call]
            b_full = twin_acts[n][call]
            if b_full.shape == a.shape:
                b = b_full
            else:
                # channel dim is dim=1 for both Conv2d (N,C,H,W) and
                # Conv3d (N,C,D,H,W); old slice is [:, :32].
                sl = [slice(None)] * b_full.dim()
                sl[1] = slice(0, a.shape[1])
                b = b_full[tuple(sl)]
            d = float((a - b).abs().max())
            if d > 0.0:
                # reason: inspect weights/bias directly
                reasons = []
                oc_o, ic_o = wo.shape[0], wo.shape[1]
                if tuple(wt[:oc_o, :ic_o].shape) == tuple(wo.shape):
                    wd = float((wt[:oc_o, :ic_o] - wo).abs().max())
                    reasons.append(f"old-block weight diff={wd:.3e}")
                else:
                    reasons.append("weight old-block shape mismatch")
                if wt.shape[1] > ic_o:
                    cross = float(wt[:oc_o, ic_o:].abs().max())
                    reasons.append(f"cross FROM-new-INTO-old maxabs={cross:.3e}"
                                   + (" (LEAK: nonzero)" if cross > 0 else " (zero OK)"))
                bo = dict(arm_d.named_parameters()).get(n + ".bias")
                bt = dict(twin_d.named_parameters()).get(n + ".bias")
                if bo is not None and bt is not None:
                    bd = float((bt[:bo.shape[0]] - bo).abs().max())
                    reasons.append(f"old-bias diff={bd:.3e}")
                print(f"FIRST DIVERGENCE scene {scene}: {n} call#{call} "
                      f"diff={d:.3e}; " + "; ".join(reasons), flush=True)
                return {"scene": scene, "layer": n, "call": call,
                        "max_abs_diff": d, "reasons": reasons}
    print(f"no diverging conv layer in fp64 for scene {scene} "
          f"(all old slices bit-identical)", flush=True)
    return {"scene": scene, "layer": None, "call": None,
            "max_abs_diff": 0.0,
            "reasons": ["all conv old slices bit-identical; check non-parametric "
                        "stages (cost-volume shift/mode, regression normalize)"]}


def main() -> None:
    log_path = TWIN_DIR / "g1_diagnosis.log"
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
    has_cuda = torch.cuda.is_available()
    gpu = "cuda" if has_cuda else "cpu"
    print(f"g1 diagnosis (LOCAL ONLY, no gate change); gpu_avail={has_cuda} "
          f"torch={torch.__version__}", flush=True)

    src_path, old_sd = bti.load_source()
    print(f"source: {src_path} keys={len(old_sd)}", flush=True)

    old_cfg = StereoNetConfig(**bti.F1_CFG)
    twin_cfg = StereoNetConfig(**{**bti.F1_CFG, "feature_channels": bti.TWIN_C})

    # fp32 models for the noise floor
    torch.manual_seed(bti.TWIN_SEED)
    arm_gpu = StereoNet(old_cfg)
    arm_gpu.load_state_dict(old_sd, strict=True)
    arm_gpu.eval().to(gpu)
    arm_cpu = StereoNet(old_cfg)
    arm_cpu.load_state_dict(old_sd, strict=True)
    arm_cpu.eval().to("cpu")

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert len(ds) == 40 and ds.names[0] == "000160_10.png", (len(ds), ds.names[:2])

    noise = fp32_noise_floor(arm_gpu, arm_cpu, ds, gpu)

    # fp64 models on CPU: embed in fp32 (exact), then convert (lossless)
    torch.manual_seed(bti.TWIN_SEED)
    twin_f = StereoNet(twin_cfg)
    bti.embed_arm_p_into_twin(old_sd, twin_f)
    arm_d = StereoNet(old_cfg)
    arm_d.load_state_dict(old_sd, strict=True)
    arm_d.eval().double().to("cpu")
    twin_d = twin_f.double().to("cpu")
    twin_d.eval()
    del twin_f
    fp64_recs, first_div = fp64_compare(arm_d, twin_d, ds)

    worst_fp64 = max(r["stages"]["disparity_final"]["max_abs_diff"]
                     for r in fp64_recs)
    if worst_fp64 <= G1_TOL:
        conclusion = "EMBED CORRECT (fp64 matches)"
    else:
        layer = (first_div or {}).get("layer")
        conclusion = f"EMBED WRONG at layer {layer}"
    print(f"CONCLUSION: {conclusion} "
          f"(worst fp64 final diff={worst_fp64:.3e}, tol={G1_TOL:.0e})", flush=True)

    out = {
        "experiment": "twin G1 failure diagnosis (local only, diagnosis only)",
        "provenance": {"utc": utc, "device_noise": gpu,
                       "python": sys.version.split()[0],
                       "torch": torch.__version__,
                       "source_sha256": bti.SOURCE_SHA,
                       "twin_seed": bti.TWIN_SEED},
        "fp32_noise_floor": noise,
        "fp64_twin_vs_armp_cpu": fp64_recs,
        "first_divergence_fp64": first_div,
        "conclusion": conclusion,
    }
    bti.atomic_write_json(TWIN_DIR / "g1_diagnosis.json", out)
    print(f"wrote {TWIN_DIR / 'g1_diagnosis.json'}", flush=True)


if __name__ == "__main__":
    main()
