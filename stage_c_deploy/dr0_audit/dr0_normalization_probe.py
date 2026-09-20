"""DR-0 normalization scale-invariance probe (STAGE C, PHASE DR-0). DIAGNOSIS ONLY.

ZERO TRAINING. Inference-only compute, CPU, torch.no_grad(). Static-graph
probe, not a model change: the captured aggregated_cost is multiplied by a
positive scalar a, and only regression+refinement are re-run. No existing repo
file is modified; this script and its JSON output are new files only.

Static analysis (maths + code cites) is recorded in the JSON provenance notes
and expanded in stage_c_deploy/DYNAMIC_RANGE_AUDIT.md section 11:
  src/models/stereonet/regression.py:48 (soft_argmin normalize),
  src/models/stereonet/regression.py:64-67 (DisparityRegression.forward),
  src/models/stereonet/stereonet.py:150-154 (forward: regression -> refinement
    -> add -> relu; normalization sits BEFORE the loss),
  src/losses/disparity.py:21-51 (masked_smooth_l1 sees ONLY the final
    disparity vs GT; no term sees the pre-normalization magnitude).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_c_deploy" / "dr0_audit"

CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
CKPTS = {
    "armp": ("stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth",
             "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"),
    "p2a": ("phase2/runs/p2a_scale_coverage/p2a_best.pth",
            "0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033"),
}
SCALES = [1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def regression_refinement(net, agg_scaled: torch.Tensor, left: torch.Tensor,
                           size: tuple[int, int]) -> torch.Tensor:
    """Re-run ONLY regression+refinement on a scaled aggregated cost.

    Replicates DisparityRegression.forward (upsample_first=True) +
    soft_argmin(normalize=True) + refinement + add + ReLU, exactly as in
    src/models/stereonet/regression.py:64-67,33-53 and
    src/models/stereonet/stereonet.py:150-154.
    """
    with torch.no_grad():
        cost = F.interpolate(agg_scaled, size=size, mode="bilinear",
                             align_corners=True)
        cost = (cost - cost.mean(1, keepdim=True)) / (
            cost.std(1, keepdim=True) + 1e-6)
        weights = torch.softmax(-cost, dim=1)
        index = torch.arange(cost.shape[1], dtype=cost.dtype,
                             device=cost.device)
        shape = [1] * cost.dim()
        shape[1] = -1
        disp_init = (weights * index.view(shape)).sum(dim=1, keepdim=True)
        residual = net.refinement(disp_init, left)
        disp = disp_init + residual
        disp = torch.relu(disp)
    return disp


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
    except Exception:
        git_head = None

    out: dict = {
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head, "host": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__,
            "command": "python stage_c_deploy/dr0_audit/dr0_normalization_probe.py",
            "argv": sys.argv,
            "static_cites": {
                "soft_argmin_normalize": "src/models/stereonet/regression.py:48",
                "regression_forward": "src/models/stereonet/regression.py:64-67",
                "stereonet_forward": "src/models/stereonet/stereonet.py:150-154",
                "loss": "src/losses/disparity.py:21-51",
            },
            "static_statement": (
                "soft_argmin(cost, normalize=True) computes "
                "z=(c-mean_d(c))/(std_d(c)+1e-6), then softmax(-z). For a>0, "
                "mean_d(a*c)=a*mean_d(c) and std_d(a*c)=a*std_d(c) (torch std, "
                "default unbiased=True, is positively homogeneous), so "
                "z(a*c)=(c-mean_d)/(std_d+1e-6/a). Exact scale invariance "
                "holds only at eps=0. The fixed eps=1e-6 breaks it only when "
                "it is NOT negligible against a*std_d, i.e. when a*std_d << "
                "1e-6: then z->a*(c-mean_d)/1e-6->0, softmax->uniform over 24 "
                "candidates, disparity->mean index 11.5 candidates. For LARGE "
                "a the formula approaches the scale-free value, but fp32 "
                "overflow of a*c itself (inf/nan) can occur BEFORE the "
                "normalization rescues it. With the measured std_d (~1e16 "
                "ARM-P, ~1e8 P2A), a*std_d >> 1e-6 over the whole tested "
                "a-range, so no eps-driven collapse is expected there; only "
                "fp32 overflow at extreme a could break invariance. In "
                "forward(), "
                "normalization sits inside regression, i.e. BEFORE refinement "
                "and BEFORE the loss; masked_smooth_l1 takes only "
                "(disparity_final, target) and no term in the training "
                "pipeline sees the pre-normalization magnitude."),
            "note": ("No existing repo artifact was modified. Static-graph "
                     "probe only: captured aggregated_cost rescaled, "
                     "regression+refinement re-run under no_grad."),
        },
        "scales": SCALES,
        "models": {},
    }
    for mname, (rel, exp) in CKPTS.items():
        got = sha256(REPO / rel)
        assert got == exp, f"{mname} checkpoint hash changed -> STOP"
        out["provenance"][mname + "_ckpt_sha256"] = got

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert [ds[i].name for i in range(5)] == \
        [f"00016{i}_10.png" for i in range(5)], "scene order changed -> STOP"
    out["scenes"] = [ds[i].name for i in range(5)]

    for mname, (rel, _) in CKPTS.items():
        blob = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        net = StereoNet(StereoNetConfig(**CFG))
        net.load_state_dict(blob["model"], strict=True)
        net.eval().cpu()
        mrec: dict = {"per_scene": [], "invariant_range": None}
        pooled_max = {a: 0.0 for a in SCALES}
        with torch.no_grad():
            for i in range(5):
                s = ds[i]
                L, R = normalize(s.left), normalize(s.right)
                tl, tr = torch.from_numpy(L), torch.from_numpy(R)
                _, st = net(tl, tr, return_stages=True)
                agg = st["aggregated_cost"]
                size = (tl.shape[-2], tl.shape[-1])
                base = regression_refinement(net, agg, tl, size)
                # verify the re-run reproduces forward disparity_final
                fwd = st["disparity_final"]
                selfdiff = float((base - fwd).abs().max().item())
                row = {"scene": s.name,
                       "rerun_selfdiff_vs_forward": selfdiff,
                       "max_abs_delta_vs_a1": {}}
                for a in SCALES:
                    d = regression_refinement(net, agg * a, tl, size)
                    delta = d - base
                    if not np.all(np.isfinite(
                            delta.detach().cpu().numpy())):
                        row["max_abs_delta_vs_a1"][str(a)] = "NONFINITE"
                    else:
                        md = float(delta.abs().max().item())
                        row["max_abs_delta_vs_a1"][str(a)] = md
                        pooled_max[a] = max(pooled_max[a], md)
                mrec["per_scene"].append(row)
        # fp32-noise invariance: max delta <= 1e-3 px (the frozen C1 scale)
        inv = [a for a in SCALES
               if isinstance(pooled_max[a], float) and pooled_max[a] <= 1e-3]
        mrec["pooled_max_abs_delta_vs_a1"] = {str(k): v for k, v in
                                              pooled_max.items()}
        mrec["invariant_range_fp32_1e3"] = (
            f"a in [{min(inv)}, {max(inv)}]" if inv else "NONE at 1e-3 px")
        out["models"][mname] = mrec
        print(f"{mname}: pooled max|d(a)-d(1)|: " + ", ".join(
            f"a={a:g}:{(v if isinstance(v, float) else v)}"
            for a, v in sorted(pooled_max.items())), flush=True)

    with open(OUT / "dr0_normalization_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT / "dr0_normalization_probe.json", flush=True)


if __name__ == "__main__":
    main()
