"""EXP-CORRESPONDENCE-GEOM-001 -- signed odd-part translation response.

Executes EXACTLY the protocol frozen in PREREGISTRATION.md, translation_spec.json
(sha256 4f504740...) and mask_spec.json (sha256 5a2927a7...). Both spec digests are
re-verified over FILE BYTES before the first model instantiation.

Inference only. Read-only checkpoints. No training: this module imports no
optimizer, constructs none, and never calls .backward() or .step().

Crop-based translation: no fill, no padding, no interpolation. Every condition
produces a 1136 x 272 crop. The axis is a function argument, not a branch, so
the horizontal and vertical arms traverse the identical code path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.datasets.kitti2015 import normalize                           # noqa: E402
from phase2.models import scaled_regression                            # noqa: E402
from phase2.viz import core                                            # noqa: E402

OUT = Path(__file__).resolve().parent
W, H = 1232, 368
M = 48                                   # crop margin == max|Delta|
CROP_W, CROP_H = W - 2 * M, H - 2 * M    # 1136 x 272
MASK_Y0, MASK_Y1, MASK_X0, MASK_X1 = 112, 256, 112, 1120      # ORIGINAL coords
DELTAS = [-48, -32, -16, 0, 16, 32, 48]
ODD_DELTAS = [16, 32, 48]
AXES = ["horizontal", "vertical"]
SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"
STRIDE = 16
ANCHOR = -1.0 / STRIDE

CHECKPOINTS = {
    "NEG_shift_none": {"path": "phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth",
                       "shift": "none", "seed": None,
                       "sha256": "d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1"},
    "POS_6b_seed0": {"path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
                     "shift": "left", "seed": 0,
                     "sha256": "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a"},
    "POS_6b_seed1": {"path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
                     "shift": "left", "seed": 1,
                     "sha256": "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf"},
    "POS_6b_seed2": {"path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
                     "shift": "left", "seed": 2,
                     "sha256": "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69"},
}
ORDER = ["NEG_shift_none", "POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"]


class HardStop(RuntimeError):
    """PREREGISTRATION.md section 10."""


def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


# ------------------------------------------------------- frozen crop primitives
def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    """Integer slice only. No interpolation, no fill, no padding."""
    if y0 < 0 or x0 < 0 or y0 + CROP_H > H or x0 + CROP_W > W:
        raise HardStop("crop window (%d,%d) leaves the image -> would require fill" % (y0, x0))
    out = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if out.shape[:2] != (CROP_H, CROP_W):
        raise HardStop("crop produced %s, expected %s" % (out.shape[:2], (CROP_H, CROP_W)))
    return out


def right_origin(axis: str, d: int) -> tuple[int, int]:
    """Frozen right-crop origin. The axis is an argument; both arms share this path."""
    if axis == "horizontal":
        return M, M - d
    if axis == "vertical":
        return M - d, M
    raise HardStop("unknown axis %r" % axis)


def build_model(key: str, device: str):
    spec = CHECKPOINTS[key]
    path = REPO_ROOT / spec["path"]
    actual = sha256_bytes(path)
    if actual != spec["sha256"]:
        raise HardStop("%s sha256 %s != frozen %s" % (key, actual, spec["sha256"]))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rec = ckpt.get("config", {}).get("cost_volume_shift")
    if rec is not None and rec != spec["shift"]:
        raise HardStop("%s recorded shift %r != %r" % (key, rec, spec["shift"]))
    model = StereoNet(StereoNetConfig(cost_volume_shift=spec["shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    if type(model.regression).__name__ != "StandardisedDisparityRegression":
        raise HardStop("%s regression %s" % (key, type(model.regression).__name__))
    if sum(p.numel() for p in model.regression.parameters()) != 0:
        raise HardStop("%s readout has parameters" % key)
    if model.cost_volume.shift != spec["shift"] or model.config.num_disparities != 12:
        raise HardStop("%s config mismatch" % key)
    return model.eval().to(device), actual


@torch.no_grad()
def disparity_initial(model, left_np, right_np, device):
    L = torch.from_numpy(normalize(left_np)).to(device)
    R = torch.from_numpy(normalize(right_np)).to(device)
    lf = model.feature_extractor(L)
    rf = model.feature_extractor(R)
    vol = model.cost_volume(lf, rf)
    d = model.regression(model.aggregation(vol), (CROP_H, CROP_W))
    return d[0, 0].detach().float().cpu().numpy().astype(np.float64), tuple(vol.shape)


def main() -> int:
    t0 = time.time()

    # ---- freeze verification BEFORE the first model instantiation --------
    pre = (OUT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    frozen = {}
    for line in (OUT / "frozen.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, n = line.split()
            frozen[n] = h
    for n, h in frozen.items():
        a = sha256_bytes(OUT / n)
        if a != h:
            raise HardStop("%s file-bytes sha256 %s != frozen %s" % (n, a, h))
        if h not in pre:
            raise HardStop("%s digest not recorded in PREREGISTRATION.md" % n)
    if (OUT / "RESULTS.md").exists():
        raise HardStop("RESULTS.md already exists before execution")
    tspec = json.loads((OUT / "translation_spec.json").read_text(encoding="utf-8"))
    mspec = json.loads((OUT / "mask_spec.json").read_text(encoding="utf-8"))
    if tspec["deltas_px"] != DELTAS or tspec["axes"] != AXES:
        raise HardStop("translation spec mismatch")
    if sorted(DELTAS) != sorted(-d for d in DELTAS):
        raise HardStop("delta set is not symmetric")
    if any(d % STRIDE for d in DELTAS):
        raise HardStop("a delta is not a multiple of the stride")
    if max(abs(d) for d in DELTAS) != M:
        raise HardStop("max|delta| != crop margin")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log = ["EXP-CORRESPONDENCE-GEOM-001",
           "device=%s torch=%s numpy=%s" % (device, torch.__version__, np.__version__),
           "freeze verified: translation_spec %s | mask_spec %s"
           % (frozen["translation_spec.json"][:16], frozen["mask_spec.json"][:16])]
    print("\n".join(log))

    results = {
        "experiment": "EXP-CORRESPONDENCE-GEOM-001",
        "kind": "inference-only; crop-based translation; no fill/padding/interpolation",
        "translation_spec_sha256": frozen["translation_spec.json"],
        "mask_spec_sha256": frozen["mask_spec.json"],
        "deltas_px": DELTAS, "odd_deltas": ODD_DELTAS, "axes": AXES,
        "crop_size": [CROP_W, CROP_H], "anchor_slope_cand_per_px": ANCHOR,
        "scenes": SCENES, "split": SPLIT,
        "mask_rule": "GT>0 AND %d<=y<%d AND %d<=x<%d (original coords)"
                     % (MASK_Y0, MASK_Y1, MASK_X0, MASK_X1),
        "primary_statistic": "S_axis = sum Odd(d)*d / sum d^2; Odd(d)=[m(+d)-m(-d)]/2; "
                             "m=median(disparity_initial[mask])",
        "decision_rule": "C1 S_h<0; C2 Odd_h(16)>Odd_h(32)>Odd_h(48); C3 |S_h|>|S_v|; "
                         "C4 S_h < S_h(NEG,same scene); PASS iff all four at 12/12",
        "p_value_used": False,
        "checkpoints": {}, "cells": [], "units": [], "sanity": {},
    }

    # ---- frozen mask, computed once per scene from GT only ---------------
    scene_cache, masks = {}, {}
    for si in SCENES:
        sc = core.load_scene(si, split=SPLIT)
        if sc.left.shape[:2] != (H, W) or sc.right.shape[:2] != (H, W):
            raise HardStop("scene %s is %s, expected %s" % (sc.name, sc.left.shape[:2], (H, W)))
        gt = sc.gt_disparity.astype(np.float64)
        m_orig = np.zeros((H, W), dtype=bool)
        m_orig[MASK_Y0:MASK_Y1, MASK_X0:MASK_X1] = True
        m_orig &= (gt > 0)
        n = int(m_orig.sum())
        exp = mspec["per_scene"][sc.name]["retained_pixels"]
        if n != exp:
            raise HardStop("mask count %d != frozen %d for %s" % (n, exp, sc.name))
        masks[si] = m_orig[M:M + CROP_H, M:M + CROP_W]      # crop frame
        if int(masks[si].sum()) != n:
            raise HardStop("mask lost pixels under the crop mapping for %s" % sc.name)
        scene_cache[si] = sc
        log.append("mask %s retained=%d (frozen %d) OK" % (sc.name, n, exp))

    n_forward = 0
    sanity = {"crop_shapes": set(), "volume_shapes": set(), "interpolation_used": False,
              "fill_used": False, "agg_hash_stable": True}

    for key in ORDER:
        model, digest = build_model(key, device)
        agg_hash_before = hashlib.sha256("".join(
            tensor_sha256(p) for _, p in model.named_parameters()).encode()).hexdigest()
        results["checkpoints"][key] = {"path": CHECKPOINTS[key]["path"], "sha256": digest,
                                       "shift": CHECKPOINTS[key]["shift"],
                                       "seed": CHECKPOINTS[key]["seed"],
                                       "weights_sha256_before": agg_hash_before}
        for si in SCENES:
            sc = scene_cache[si]
            mask = masks[si]
            left_c = crop(sc.left, M, M)                      # FIXED left crop
            sanity["crop_shapes"].add(left_c.shape[:2])
            m_axis = {}
            for axis in AXES:
                for d in DELTAS:
                    if d == 0 and axis == "vertical" and ("horizontal", 0) in m_axis:
                        m_axis[(axis, 0)] = m_axis[("horizontal", 0)]      # shared, not re-run
                        continue
                    y0, x0 = right_origin(axis, d)
                    right_c = crop(sc.right, y0, x0)
                    sanity["crop_shapes"].add(right_c.shape[:2])
                    arr, vshape = disparity_initial(model, left_c, right_c, device)
                    n_forward += 1
                    sanity["volume_shapes"].add(vshape)
                    if arr.shape != (CROP_H, CROP_W):
                        raise HardStop("disparity_initial %s != crop %s"
                                       % (arr.shape, (CROP_H, CROP_W)))
                    m_axis[(axis, d)] = float(np.median(arr[mask]))
            agg_hash_after = hashlib.sha256("".join(
                tensor_sha256(p) for _, p in model.named_parameters()).encode()).hexdigest()
            if agg_hash_after != agg_hash_before:
                sanity["agg_hash_stable"] = False
                raise HardStop("model weights changed during %s/%s" % (key, sc.name))

            cell = {"checkpoint": key, "seed": CHECKPOINTS[key]["seed"],
                    "scene": sc.name, "scene_index": si, "retained_pixels": int(mask.sum()),
                    "m": {a: {str(d): m_axis[(a, d)] for d in DELTAS} for a in AXES}}
            for a in AXES:
                odd = {str(d): (m_axis[(a, d)] - m_axis[(a, -d)]) / 2.0 for d in ODD_DELTAS}
                even = {str(d): (m_axis[(a, d)] + m_axis[(a, -d)]) / 2.0 for d in ODD_DELTAS}
                num = sum(odd[str(d)] * d for d in ODD_DELTAS)
                den = sum(d * d for d in ODD_DELTAS)
                cell["odd_" + a] = odd
                cell["even_" + a] = even
                cell["S_" + a] = float(num / den)
                cell["F_" + a] = float((num / den) / ANCHOR)
            results["cells"].append(cell)
            log.append("done %-14s %-13s S_h=%+.6f S_v=%+.6f  odd_h=%s"
                       % (key, sc.name, cell["S_horizontal"], cell["S_vertical"],
                          " ".join("%+.4f" % cell["odd_horizontal"][str(d)] for d in ODD_DELTAS)))
            print(log[-1])
        del model

    # ---- decision rule --------------------------------------------------
    neg_by_scene = {c["scene"]: c for c in results["cells"] if c["checkpoint"] == "NEG_shift_none"}
    for c in results["cells"]:
        if c["checkpoint"] == "NEG_shift_none":
            continue
        negS = neg_by_scene[c["scene"]]["S_horizontal"]
        o = c["odd_horizontal"]
        C1 = bool(c["S_horizontal"] < 0)
        C2 = bool(o["16"] > o["32"] > o["48"])
        C3 = bool(abs(c["S_horizontal"]) > abs(c["S_vertical"]))
        C4 = bool(c["S_horizontal"] < negS)
        results["units"].append({
            "checkpoint": c["checkpoint"], "seed": c["seed"], "scene": c["scene"],
            "S_h": c["S_horizontal"], "S_v": c["S_vertical"], "F_h": c["F_horizontal"],
            "odd_h_16": o["16"], "odd_h_32": o["32"], "odd_h_48": o["48"],
            "S_h_NEG": negS, "C1": C1, "C2": C2, "C3": C3, "C4": C4,
            "PASS": bool(C1 and C2 and C3 and C4),
            "failure_class": (None if (C1 and C2 and C3 and C4) else
                              ("sign" if not C1 else "monotonicity" if not C2 else
                               "axis-specificity" if not C3 else "trained-vs-search-free")),
        })

    n_units = len(results["units"])
    n_pass = sum(u["PASS"] for u in results["units"])
    global_pass = bool(n_units == 12 and n_pass == 12)

    sanity["crop_shapes"] = sorted(str(s) for s in sanity["crop_shapes"])
    sanity["volume_shapes"] = sorted(str(s) for s in sanity["volume_shapes"])
    sanity["forward_passes"] = n_forward
    sanity["forward_passes_expected"] = 4 * 4 * 13
    sanity["forward_passes_match"] = bool(n_forward == 4 * 4 * 13)
    sanity["all_crops_identical"] = bool(len(sanity["crop_shapes"]) == 1)
    sanity["training_occurred"] = False
    sanity["deltas_symmetric"] = True
    results["sanity"] = sanity
    results["summary"] = {
        "positive_units": n_units, "units_pass": n_pass,
        "C1_pass": sum(u["C1"] for u in results["units"]),
        "C2_pass": sum(u["C2"] for u in results["units"]),
        "C3_pass": sum(u["C3"] for u in results["units"]),
        "C4_pass": sum(u["C4"] for u in results["units"]),
        "global_pass": global_pass,
        "failure_classes": {k: sum(1 for u in results["units"] if u["failure_class"] == k)
                            for k in ("sign", "monotonicity", "axis-specificity",
                                      "trained-vs-search-free")},
    }
    results["VERDICT"] = ("GEOMETRIC-CORRESPONDENCE-EVIDENCE" if global_pass
                          else "GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED")
    results["wall_clock_s"] = time.time() - t0
    results["determinism"] = {"cudnn.deterministic": True, "cudnn.benchmark": False,
                              "use_deterministic_algorithms": True,
                              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                              "no_grad": True, "eval": True, "dtype": "fp32"}
    results["protocol_deviations"] = "none"

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.append("forwards=%d (expected %d)" % (n_forward, 4 * 4 * 13))
    log.append("units PASS %d/%d  C1=%d C2=%d C3=%d C4=%d"
               % (n_pass, n_units, results["summary"]["C1_pass"], results["summary"]["C2_pass"],
                  results["summary"]["C3_pass"], results["summary"]["C4_pass"]))
    log.append("VERDICT: " + results["VERDICT"])
    log.append("wall_clock_s=%.2f" % results["wall_clock_s"])
    (OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8", newline="")
    print("\nVERDICT:", results["VERDICT"])
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
