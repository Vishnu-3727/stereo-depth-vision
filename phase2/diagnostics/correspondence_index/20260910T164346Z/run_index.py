"""EXP-CORRESPONDENCE-INDEX-001 — inference-only candidate-axis re-indexing.

Preregistered in PREREGISTRATION.md. Read-only checkpoints. No training.
Builds each cost volume once, derives all arms via index_select(dim=2),
runs aggregation+standardized regression, reports disparity_initial only.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBLAS_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG

import numpy as np
import torch

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig
from src.datasets.kitti2015 import normalize
from phase2.models import scaled_regression
from phase2.viz import core

OUT = Path(__file__).resolve().parent
FOCUS_SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"

PERMS = {
    "identity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "plus1": [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "plus2": [10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "minus1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
    "random": [7, 2, 10, 0, 5, 11, 1, 8, 4, 9, 6, 3],
}

CHECKPOINTS = {
    "NEG_shift_none": {
        "path": "phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth",
        "shift": "none",
    },
    "POS_6b_seed0": {
        "path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
        "shift": "left",
    },
    "POS_6b_seed1": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
        "shift": "left",
    },
    "POS_6b_seed2": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
        "shift": "left",
    },
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_model(key: str, device: str):
    spec = CHECKPOINTS[key]
    path = REPO_ROOT / spec["path"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rec = ckpt.get("config", {}).get("cost_volume_shift")
    if rec is not None and rec != spec["shift"]:
        raise ValueError(f"{key}: recorded shift {rec} != expected {spec['shift']}")
    model = StereoNet(StereoNetConfig(cost_volume_shift=spec["shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    assert type(model.regression).__name__ == "StandardisedDisparityRegression"
    assert model.cost_volume.shift == spec["shift"]
    return model.eval().to(device), ckpt


@torch.no_grad()
def run_once(model, volume: torch.Tensor, size):
    cost = model.aggregation(volume)
    d = model.regression(cost, size)
    return cost, d


def main() -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log = [f"device={device} torch={torch.__version__}"]
    results = {"perms": PERMS, "focus_scenes": FOCUS_SCENES, "split": SPLIT,
               "checkpoints": {}, "rows": [], "gate": {}}
    order = ["NEG_shift_none", "POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"]
    neg_ok = True
    for key in order:
        if key.startswith("POS") and not neg_ok:
            log.append("SKIP positives: negative gate failed")
            break
        model, ckpt = build_model(key, device)
        path = REPO_ROOT / CHECKPOINTS[key]["path"]
        ckpt_info = {"path": CHECKPOINTS[key]["path"],
                     "sha256": sha256(path),
                     "shift": CHECKPOINTS[key]["shift"],
                     "regression": type(model.regression).__name__}
        results["checkpoints"][key] = ckpt_info
        for si in FOCUS_SCENES:
            scene = core.load_scene(si, split=SPLIT)
            L = torch.from_numpy(normalize(scene.left)).to(device)
            R = torch.from_numpy(normalize(scene.right)).to(device)
            gt = scene.gt_disparity.astype(np.float64)
            mask = (gt > 0) & (gt / 16.0 >= 2.0) & (gt / 16.0 <= 8.0)
            nmask = int(mask.sum())
            with torch.no_grad():
                lf = model.feature_extractor(L)
                rf = model.feature_extractor(R)
                volume = model.cost_volume(lf, rf)
            vinfo = {"shape": list(volume.shape), "dtype": str(volume.dtype),
                     "device": str(volume.device), "contig": volume.is_contiguous()}
            H, W = scene.left.shape[:2]
            size = (H, W)
            outs = {}
            for pname, plist in PERMS.items():
                idx = torch.tensor(plist, dtype=torch.long, device=volume.device)
                vp = volume.index_select(2, idx)
                if pname == "identity":
                    vd = float((vp - volume).abs().max().cpu())
                    vinfo["identity_vs_original_max_abs_diff"] = vd
                    vinfo["identity_shape"] = list(vp.shape)
                    vinfo["identity_contiguous"] = vp.is_contiguous()
                _, d = run_once(model, vp, size)
                outs[pname] = d[0, 0].detach().float().cpu().numpy().astype(np.float64)
            base = outs["identity"]
            for pname in PERMS:
                cur = outs[pname]
                diff = cur - base
                m = mask
                if nmask:
                    vals = cur[m]
                    row = {"checkpoint": key, "scene": scene.name,
                           "scene_index": si, "permutation": pname,
                           "mean": float(vals.mean()), "median": float(np.median(vals)),
                           "std": float(vals.std()),
                           "delta_mean_vs_identity": float(diff[m].mean()),
                           "delta_median_vs_identity": float(np.median(diff[m])),
                           "max_abs_diff_vs_identity_full": float(np.abs(diff).max()),
                           "max_abs_diff_vs_identity_masked": float(np.abs(diff[m]).max()),
                           "valid_pixels": nmask}
                else:
                    row = {"checkpoint": key, "scene": scene.name,
                           "scene_index": si, "permutation": pname,
                           "mean": None, "median": None, "std": None,
                           "delta_mean_vs_identity": None,
                           "delta_median_vs_identity": None,
                           "max_abs_diff_vs_identity_full": float(np.abs(diff).max()),
                           "max_abs_diff_vs_identity_masked": None,
                           "valid_pixels": 0}
                results["rows"].append(row)
            # volume-level identity check per scene (first scene enough, record all)
            idx_id = torch.tensor(PERMS["identity"], dtype=torch.long, device=volume.device)
            for pname in ["plus1", "plus2", "minus1", "random"]:
                idx = torch.tensor(PERMS[pname], dtype=torch.long, device=volume.device)
                vd = float((volume.index_select(2, idx) - volume).abs().max().cpu())
                results.setdefault("volume_diffs", []).append(
                    {"checkpoint": key, "scene": scene.name, "perm": pname,
                     "max_abs_diff_vs_original": vd})
            if key == "NEG_shift_none":
                for pname in PERMS:
                    if pname == "identity":
                        continue
                    fmax = float(np.abs(outs[pname] - base).max())
                    if not (fmax == 0.0):
                        neg_ok = False
                        log.append(f"NEG GATE FAIL scene={scene.name} perm={pname} maxdiff={fmax!r}")
                # also volume-level: degenerate slices identical so any perm diff must be 0
                for pname in ["plus1", "plus2", "minus1", "random"]:
                    idx = torch.tensor(PERMS[pname], dtype=torch.long, device=volume.device)
                    vd = float((volume.index_select(2, idx) - volume).abs().max().cpu())
                    if not (vd == 0.0):
                        neg_ok = False
                        log.append(f"NEG VOLUME GATE FAIL scene={scene.name} perm={pname} vdiff={vd!r}")
            log.append(f"done {key} scene {scene.name} mask={nmask} {vinfo['shape']}")
        del model
    results["gate"]["negative_invariant"] = bool(neg_ok)
    results["wall_clock_s"] = time.time() - t0
    results["determinism"] = {"cudnn.deterministic": True, "cudnn.benchmark": False,
                              "use_deterministic_algorithms": True,
                              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps({"negative_invariant": neg_ok,
                      "n_rows": len(results["rows"]),
                      "wall_clock_s": results["wall_clock_s"]}, indent=2))
    for r in results["rows"]:
        print(f"{r['checkpoint']} sc{r['scene_index']} {r['permutation']:8s} med={r['median']} dmed={r['delta_median_vs_identity']} maxfull={r['max_abs_diff_vs_identity_full']:.6g} n={r['valid_pixels']}")
    return 0 if neg_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
