"""Stage C2: frozen ARM-P disparity inference (MICROCHIP STEREONET).

Loads the frozen checkpoint (SHA256 asserted BEFORE torch.load), builds the
frozen StereoNetConfig, and runs inference EXACTLY the way the frozen
evaluator stage_c_deploy/dr1_rescale/dr1_eval_40scene.py does:
same normalize(), same net.eval(), same torch.no_grad(), same net(left, right)
call. No rescaling, no architecture change.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

ARMP_REL = "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth"
ARMP_SHA = "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_frozen_net(device: str | None = None,
                    checkpoint: str | Path | None = None,
                    expected_sha256: str | None = None,
                    ) -> tuple[StereoNet, str]:
    """Load the frozen checkpoint after asserting its SHA256. Returns (net, device).

    Called with neither ``checkpoint`` nor ``expected_sha256`` this behaves
    EXACTLY as before: it loads ``REPO / ARMP_REL`` and asserts its sha256
    against the frozen ``ARMP_SHA``. Passed both, it loads the supplied file
    (absolute, or repo-relative) and asserts its sha256 against the supplied
    sha instead. Supplying only one of the two is an error.
    """
    if (checkpoint is None) != (expected_sha256 is None):
        raise ValueError("checkpoint and expected_sha256 must be given together")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint is None:
        ckpt = REPO / ARMP_REL
        want = ARMP_SHA
    else:
        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = REPO / ckpt
        want = expected_sha256
        assert want is not None  # narrowed by the ValueError above
    digest = sha256_file(ckpt)
    assert digest == want, (
        f"checkpoint hash changed: {digest} != {want} -> STOP")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = StereoNet(StereoNetConfig(**CFG))
    net.load_state_dict(blob["model"], strict=True)
    net.eval().to(device)
    return net, device


def infer_disparity_scene(net: StereoNet, device: str, scene_name: str,
                          root: str | Path | None = None) -> np.ndarray:
    """Run inference on one named KITTI hailo_val scene.

    Returns raw disparity (H x W float64, cropped 368x1232) exactly as the
    frozen evaluator produces it: normalize() on 0-255 RGB, NCHW torch tensor,
    net(tl, tr) under torch.no_grad().
    """
    root = Path(root) if root is not None else (REPO / "data" / "kitti2015")
    ds = Kitti2015Stereo(root, split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    idx = list(ds.names).index(scene_name)
    sm = ds[idx]
    tl = torch.from_numpy(normalize(sm.left)).to(device)
    tr = torch.from_numpy(normalize(sm.right)).to(device)
    with torch.no_grad():
        out = net(tl, tr)
    return out[0, 0].detach().cpu().numpy().astype(np.float64)


def demo() -> None:
    net, device = load_frozen_net()
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    name = ds.names[0]
    d = infer_disparity_scene(net, device, name)
    assert d.shape == (368, 1232), d.shape
    assert np.all(np.isfinite(d)), "raw disparity must be finite (final ReLU)"
    assert (d >= 0).all(), "final ReLU means disparity >= 0"
    print(f"armp_depth self-check passed on {name}: "
          f"shape={d.shape} min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f}")


if __name__ == "__main__":
    demo()
