"""Frozen-contract evaluator: thin wrap over existing repo code.

Reuses `src/datasets/kitti2015.py` (slices/crop/scales),
`src/evaluation/metrics.py:52-84` (pooled EPE/D1/RMSE/BAD1-3), and the
provenance pattern of `src/common/experiment.py`. Adds only what was missing:
a single entry point plus a protocol-mixing guard that refuses comparison
against the reference when scene set / pixel count / GT scale do not match
`phase0/docs/BASELINE_CONTRACT.md`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402

# Frozen contract values (phase0/docs/BASELINE_CONTRACT.md + REFERENCE_BASELINE.md).
CONTRACT = {
    "dataset": "kitti2015",
    "split": "hailo_val",
    "scenes": 40,
    "gt_source": "disp_occ_0",
    "gt_scale": 256.0,
    "valid_pixels": 3802797,
    "resolution": [368, 1232],
    "reference": {
        "checkpoint": "reference/onnx/stereonet.onnx",
        "epe": 1.3134471,
        "d1": 8.1543664,
    },
    "pytorch_baseline": {
        "checkpoint": "results/training/convergence_run.pth",
        "epe": 15.3958267,
        "d1": 88.3922807,
    },
}

EXPECTED_STATE_KEYS = 72
EXPECTED_PARAMS = 423586


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_sha(state_dict) -> str:
    import torch

    flats = [v.reshape(-1).float().cpu() for v in state_dict.values()
             if hasattr(v, "is_floating_point") and v.is_floating_point()]
    data = torch.cat(flats).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def strict_load_report(model, state_dict) -> dict:
    """Load with strict=True; report matched/missing/unexpected keys."""
    want = set(model.state_dict().keys())
    got = set(state_dict.keys())
    missing = sorted(want - got)
    unexpected = sorted(got - want)
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "matched": len(want & got),
        "missing": missing,
        "unexpected": unexpected,
        "strict_ok": not missing and not unexpected,
        "expected_keys": EXPECTED_STATE_KEYS,
        "keys_ok": len(want & got) == EXPECTED_STATE_KEYS,
        "params": n_params,
        "params_ok": n_params == EXPECTED_PARAMS,
    }


def record_provenance(seed: int | None = 0, extra: dict | None = None) -> dict:
    def git(*a):
        try:
            r = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=15)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    import torch

    rec = {
        "git_head": git("rev-parse", "HEAD"),
        "git_subject": git("log", "-1", "--format=%s"),
        "git_status": git("status", "--short"),
        "seed": seed,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import onnxruntime as ort

        rec["onnxruntime"] = ort.__version__
    except Exception:
        rec["onnxruntime"] = None
    if extra:
        rec.update(extra)
    return rec


def pooled_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    m = disparity_metrics(np.concatenate(preds), np.concatenate(gts))
    return m.as_dict()


def refuse_unless_contract(n_scenes: int, valid_pixels: int, scale: float,
                            split: str, gt_source: str) -> dict:
    """Protocol-mixing guard. Raises on mismatch; never emits a comparison."""
    detail = {
        "scenes": n_scenes,
        "valid_pixels": valid_pixels,
        "gt_scale": scale,
        "split": split,
        "gt_source": gt_source,
        "contract": {"scenes": 40, "valid_pixels": 3802797,
                     "gt_scale": 256.0, "split": "hailo_val",
                     "gt_source": "disp_occ_0"},
    }
    ok = (n_scenes == 40 and valid_pixels == 3802797 and scale == 256.0
          and split == "hailo_val" and gt_source == "disp_occ_0")
    detail["contract_match"] = ok
    if not ok:
        raise ValueError("protocol mismatch vs frozen contract: " + json.dumps(detail))
    return detail


def score_onnx(limit: int | None = None, provider: str = "cpu") -> dict:
    import onnxruntime as ort

    model_path = REPO_ROOT / CONTRACT["reference"]["checkpoint"]
    providers = ["CPUExecutionProvider"]
    if provider == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts,
                                   providers=providers)
    in_names = [i.name for i in session.get_inputs()]
    out_name = session.get_outputs()[0].name
    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    preds, gts, names = [], [], []
    for i in range(n):
        s = ds[i]
        out = session.run([out_name], {in_names[0]: normalize(s.left),
                                       in_names[1]: normalize(s.right)})[0]
        pred = out[0, 0].astype(np.float64)
        valid = s.disparity > 0
        preds.append(pred[valid].astype(np.float64))
        gts.append(s.disparity[valid].astype(np.float64))
        names.append(s.name)
    m = pooled_metrics(preds, gts)
    guard = None
    if n == 40:
        guard = refuse_unless_contract(n, m["valid_pixels"], 256.0, "hailo_val",
                                       "disp_occ_0")
    return {"checkpoint": CONTRACT["reference"]["checkpoint"],
            "sha256": sha256_file(model_path), "scenes": n, "names": names,
            "metrics": m, "guard": guard,
            "providers": session.get_providers(),
            "provenance": record_provenance()}


def score_checkpoint(ckpt_rel: str = "results/training/convergence_run.pth",
                     device: str | None = None, limit: int | None = None) -> dict:
    import torch

    from src.models.stereonet import StereoNet, StereoNetConfig

    ckpt_path = REPO_ROOT / ckpt_rel
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    config = blob.get("config", {}) if isinstance(blob, dict) else {}
    model = StereoNet(StereoNetConfig())
    compat = strict_load_report(model, state)
    model.eval().to(dev)
    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    preds, gts, names = [], [], []
    with torch.no_grad():
        for i in range(n):
            s = ds[i]
            out = model(torch.from_numpy(normalize(s.left)).to(dev),
                        torch.from_numpy(normalize(s.right)).to(dev))
            pred = out[0, 0].cpu().numpy().astype(np.float64)
            valid = s.disparity > 0
            preds.append(pred[valid])
            gts.append(s.disparity[valid].astype(np.float64))
            names.append(s.name)
    m = pooled_metrics(preds, gts)
    guard = None
    if n == 40:
        guard = refuse_unless_contract(n, m["valid_pixels"], 256.0, "hailo_val",
                                       "disp_occ_0")
    return {"checkpoint": ckpt_rel, "sha256": sha256_file(ckpt_path),
            "weight_sha16": weight_sha(state), "config": config,
            "compat": compat, "device": dev, "scenes": n, "names": names,
            "metrics": m, "guard": guard, "provenance": record_provenance()}
