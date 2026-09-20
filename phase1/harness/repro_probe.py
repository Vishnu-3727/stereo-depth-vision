"""CLI: reproducibility bound — two identical seeded forward passes.

Runs the frozen-architecture model twice from the same seed on CPU with
determinism controls on and reports whether the outputs are bitwise identical.
No training. Mirrors the in-process repeat probe of
`phase2/diagnostics/determinism/diag_determinism.py:266-274`.

Single-line use:
  python phase1\\harness\\repro_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase1" / "harness"))

from determinism import enable_determinism, ensure_cublas_env, flag_snapshot, seed_all  # noqa: E402


def main() -> None:
    ensure_cublas_env()
    seed_all(0)
    controls = enable_determinism()
    import torch

    from src.datasets.kitti2015 import Kitti2015Stereo, normalize
    from src.models.stereonet import StereoNet, StereoNetConfig

    device = "cpu"
    torch.manual_seed(0)
    model = StereoNet(StereoNetConfig()).to(device).eval()
    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    s = ds[0]
    left = torch.from_numpy(normalize(s.left)).to(device)
    right = torch.from_numpy(normalize(s.right)).to(device)
    with torch.no_grad():
        f1 = model(left, right)
        f2 = model(left, right)
    max_diff = float((f1 - f2).abs().max())
    identical = bool(torch.equal(f1, f2))
    res = {"seed": 0, "device": device, "scene": s.name,
           "bitwise_identical": identical, "max_abs_diff": max_diff,
           "out_mean": float(f1.mean()), "controls": controls,
           "flags": flag_snapshot()}
    print("scene             {}".format(s.name))
    print("bitwise_identical {}".format(identical))
    print("max_abs_diff      {:.3e}".format(max_diff))
    print("out_mean          {:.6f}".format(res["out_mean"]))
    out = REPO_ROOT / "phase1" / "harness" / "repro_probe.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote             {}".format(out))


if __name__ == "__main__":
    main()
