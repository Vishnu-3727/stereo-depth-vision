"""EXP-CORRESPONDENCE-GEOM-002 -- synthetic fronto-parallel self-pair sweep.

Executes EXACTLY the protocol frozen in PREREGISTRATION.md,
construction_spec.json (sha256 a9e8f250...) and mask_spec.json (sha256 b274ce03...).
Both digests are re-verified over FILE BYTES before the first model instantiation.

Both images are crops of the SAME source image (scene.left); the scene's own
right image is never used. Integer slicing only: no fill, no padding, no
interpolation. The imposed disparity is exactly t px, uniform over the crop.

Inference only. Read-only checkpoints. No training: this module imports no
optimizer, constructs none, and never calls .backward() or .step().
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
W, H, STRIDE = 1232, 368, 16
TMAX = 96
CROP_W, CROP_H = W - TMAX, H - TMAX          # 1136 x 272
BORDER = 64
T_LEVELS = [0, 16, 32, 48, 64, 80, 96]
T_ODD = [16, 32, 48, 64, 80, 96]             # levels entering the slope
AXES = ["horizontal", "vertical"]
EXPECTED_MASK = 145152

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
    """PREREGISTRATION.md section 11."""


def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def arr_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------- frozen crop construction
def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    """Integer slice only. No interpolation, no fill, no padding."""
    if y0 < 0 or x0 < 0 or y0 + CROP_H > H or x0 + CROP_W > W:
        raise HardStop("crop window (%d,%d) leaves the source -> would need fill" % (y0, x0))
    out = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if out.shape[:2] != (CROP_H, CROP_W):
        raise HardStop("crop produced %s, expected %s" % (out.shape[:2], (CROP_H, CROP_W)))
    return out


def right_origin(axis: str, t: int) -> tuple[int, int]:
    """Frozen right-crop origin: a_R = a_L + t. The axis is an argument, not a branch."""
    if axis == "horizontal":
        return 0, t
    if axis == "vertical":
        return t, 0
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
def disparity_initial(model, left_np, right_np, device, want_volume=False):
    L = torch.from_numpy(normalize(left_np)).to(device)
    R = torch.from_numpy(normalize(right_np)).to(device)
    lf = model.feature_extractor(L)
    rf = model.feature_extractor(R)
    vol = model.cost_volume(lf, rf)
    d = model.regression(model.aggregation(vol), (CROP_H, CROP_W))
    arr = d[0, 0].detach().float().cpu().numpy().astype(np.float64)
    slice_spread = None
    if want_volume:
        base = vol[:, :, 0:1]
        slice_spread = float((vol - base).abs().max().cpu())
    return arr, tuple(vol.shape), slice_spread


def main() -> int:
    t0 = time.time()

    # ---- freeze verification BEFORE the first model instantiation ----------
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
    cspec = json.loads((OUT / "construction_spec.json").read_text(encoding="utf-8"))
    mspec = json.loads((OUT / "mask_spec.json").read_text(encoding="utf-8"))
    if cspec["t_levels_px"] != T_LEVELS or cspec["axes"] != AXES:
        raise HardStop("construction spec mismatch")
    if any(t % STRIDE for t in T_LEVELS):
        raise HardStop("a t level is not a multiple of the stride")
    if max(T_LEVELS) != TMAX:
        raise HardStop("t_max != crop margin")
    SCENES = mspec["scenes"]
    if set(SCENES) & set(mspec["geom001_focus_excluded"]):
        raise HardStop("scene set overlaps the GEOM-001 focus set")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log = ["EXP-CORRESPONDENCE-GEOM-002",
           "device=%s torch=%s numpy=%s" % (device, torch.__version__, np.__version__),
           "freeze verified: construction %s | mask %s"
           % (frozen["construction_spec.json"][:16], frozen["mask_spec.json"][:16]),
           "scenes %r %r" % (SCENES, mspec["scene_names"])]
    print("\n".join(log))

    results = {
        "experiment": "EXP-CORRESPONDENCE-GEOM-002",
        "kind": "inference-only; synthetic fronto-parallel self-pair; no fill/padding/interpolation",
        "construction_spec_sha256": frozen["construction_spec.json"],
        "mask_spec_sha256": frozen["mask_spec.json"],
        "t_levels_px": T_LEVELS, "t_slope_levels": T_ODD, "axes": AXES,
        "crop_size": [CROP_W, CROP_H], "anchor_slope_cand_per_cand": 1.0,
        "scenes": SCENES, "scene_names": mspec["scene_names"], "split": mspec["split"],
        "mask_rule": mspec["definition_crop_coords"],
        "primary_statistic": "Slope_axis = sum_t (m(t)-m(0))*(t/16) / sum_t (t/16)^2",
        "decision_rule": "C1 Slope_h>0; C2 m_h(t) strictly increasing over all 7 t; "
                         "C3 |Slope_h|>|Slope_v|; C4 Slope_h > Slope_h(NEG,same scene); "
                         "PASS iff all four at 12/12",
        "p_value_used": False,
        "checkpoints": {}, "cells": [], "units": [], "sanity": {}, "wiring_check": {},
    }

    # mask in crop coordinates -- purely geometric, no GT
    mask = np.zeros((CROP_H, CROP_W), dtype=bool)
    mask[BORDER:CROP_H - BORDER, BORDER:CROP_W - BORDER] = True
    if int(mask.sum()) != EXPECTED_MASK:
        raise HardStop("mask count %d != frozen %d" % (int(mask.sum()), EXPECTED_MASK))
    log.append("mask retained=%d (frozen %d) OK" % (int(mask.sum()), EXPECTED_MASK))

    scene_cache = {}
    for si in SCENES:
        sc = core.load_scene(si, split=mspec["split"])
        if sc.left.shape[:2] != (H, W):
            raise HardStop("scene %s is %s, expected %s" % (sc.name, sc.left.shape[:2], (H, W)))
        scene_cache[si] = sc

    n_forward = 0
    sanity = {"crop_shapes": set(), "volume_shapes": set(), "left_crop_hashes": set(),
              "interpolation_used": False, "fill_used": False, "weights_stable": True,
              "neg_volume_slice_spread_max": 0.0}

    for key in ORDER:
        model, digest = build_model(key, device)
        wh_before = hashlib.sha256("".join(
            tensor_sha256(p) for _, p in model.named_parameters()).encode()).hexdigest()
        results["checkpoints"][key] = {"path": CHECKPOINTS[key]["path"], "sha256": digest,
                                       "shift": CHECKPOINTS[key]["shift"],
                                       "seed": CHECKPOINTS[key]["seed"],
                                       "weights_sha256": wh_before}
        is_neg = key == "NEG_shift_none"

        for si in SCENES:
            sc = scene_cache[si]
            left_c = crop(sc.left, 0, 0)                      # FIXED left crop
            sanity["crop_shapes"].add(left_c.shape[:2])
            sanity["left_crop_hashes"].add(arr_sha256(left_c))
            m_axis = {}
            for axis in AXES:
                for t in T_LEVELS:
                    if t == 0 and axis == "vertical":
                        m_axis[(axis, 0)] = m_axis[("horizontal", 0)]   # shared, not re-run
                        continue
                    y0, x0 = right_origin(axis, t)
                    right_c = crop(sc.left, y0, x0)           # SAME source image
                    sanity["crop_shapes"].add(right_c.shape[:2])
                    want_vol = is_neg and axis == "horizontal" and t == 0
                    arr, vshape, spread = disparity_initial(model, left_c, right_c,
                                                            device, want_volume=is_neg)
                    n_forward += 1
                    sanity["volume_shapes"].add(vshape)
                    if is_neg and spread is not None:
                        sanity["neg_volume_slice_spread_max"] = max(
                            sanity["neg_volume_slice_spread_max"], spread)
                    if arr.shape != (CROP_H, CROP_W):
                        raise HardStop("disparity_initial %s != crop %s"
                                       % (arr.shape, (CROP_H, CROP_W)))
                    m_axis[(axis, t)] = float(np.median(arr[mask]))
                    if want_vol:
                        results["wiring_check"][sc.name] = {
                            "NEG_t0_disparity_initial_median": m_axis[(axis, 0)],
                            "NEG_t0_disparity_initial_min": float(arr.min()),
                            "NEG_t0_disparity_initial_max": float(arr.max()),
                            "expected_exactly": 5.5}
            wh_after = hashlib.sha256("".join(
                tensor_sha256(p) for _, p in model.named_parameters()).encode()).hexdigest()
            if wh_after != wh_before:
                sanity["weights_stable"] = False
                raise HardStop("model weights changed during %s/%s" % (key, sc.name))

            cell = {"checkpoint": key, "seed": CHECKPOINTS[key]["seed"],
                    "scene": sc.name, "scene_index": si, "retained_pixels": int(mask.sum()),
                    "m": {a: {str(t): m_axis[(a, t)] for t in T_LEVELS} for a in AXES}}
            den = sum((t / STRIDE) ** 2 for t in T_ODD)
            for a in AXES:
                base = m_axis[(a, 0)]
                num = sum((m_axis[(a, t)] - base) * (t / STRIDE) for t in T_ODD)
                cell["Slope_" + a] = float(num / den)
                cell["delta_" + a] = {str(t): m_axis[(a, t)] - base for t in T_LEVELS}
            results["cells"].append(cell)
            log.append("done %-14s %-13s Slope_h=%+.5f Slope_v=%+.5f  m_h=%s"
                       % (key, sc.name, cell["Slope_horizontal"], cell["Slope_vertical"],
                          " ".join("%.3f" % m_axis[("horizontal", t)] for t in T_LEVELS)))
            print(log[-1])
        del model

    # ---- decision rule ---------------------------------------------------
    neg_by_scene = {c["scene"]: c for c in results["cells"] if c["checkpoint"] == "NEG_shift_none"}
    for c in results["cells"]:
        if c["checkpoint"] == "NEG_shift_none":
            continue
        negS = neg_by_scene[c["scene"]]["Slope_horizontal"]
        mh = [c["m"]["horizontal"][str(t)] for t in T_LEVELS]
        C1 = bool(c["Slope_horizontal"] > 0)
        C2 = bool(all(mh[i] < mh[i + 1] for i in range(len(mh) - 1)))
        C3 = bool(abs(c["Slope_horizontal"]) > abs(c["Slope_vertical"]))
        C4 = bool(c["Slope_horizontal"] > negS)
        results["units"].append({
            "checkpoint": c["checkpoint"], "seed": c["seed"], "scene": c["scene"],
            "Slope_h": c["Slope_horizontal"], "Slope_v": c["Slope_vertical"],
            "Slope_h_NEG": negS,
            "m_h": {str(t): c["m"]["horizontal"][str(t)] for t in T_LEVELS},
            "C1": C1, "C2": C2, "C3": C3, "C4": C4,
            "PASS": bool(C1 and C2 and C3 and C4),
            "failure_class": (None if (C1 and C2 and C3 and C4) else
                              ("flat-or-wrong-sign" if not C1 else
                               "non-monotone" if not C2 else
                               "non-specific" if not C3 else "search-free-equivalent")),
        })

    n_units = len(results["units"])
    n_pass = sum(u["PASS"] for u in results["units"])
    global_pass = bool(n_units == 12 and n_pass == 12)

    sanity["crop_shapes"] = sorted(str(s) for s in sanity["crop_shapes"])
    sanity["volume_shapes"] = sorted(str(s) for s in sanity["volume_shapes"])
    sanity["left_crop_distinct_hashes"] = len(sanity.pop("left_crop_hashes"))
    sanity["forward_passes"] = n_forward
    sanity["forward_passes_expected"] = 4 * len(SCENES) * 13
    sanity["forward_passes_match"] = bool(n_forward == 4 * len(SCENES) * 13)
    sanity["all_crops_identical_shape"] = bool(len(sanity["crop_shapes"]) == 1)
    sanity["training_occurred"] = False
    results["sanity"] = sanity
    results["summary"] = {
        "positive_units": n_units, "units_pass": n_pass,
        "C1_pass": sum(u["C1"] for u in results["units"]),
        "C2_pass": sum(u["C2"] for u in results["units"]),
        "C3_pass": sum(u["C3"] for u in results["units"]),
        "C4_pass": sum(u["C4"] for u in results["units"]),
        "global_pass": global_pass,
        "failure_classes": {k: sum(1 for u in results["units"] if u["failure_class"] == k)
                            for k in ("flat-or-wrong-sign", "non-monotone",
                                      "non-specific", "search-free-equivalent")},
    }
    results["VERDICT"] = ("SYNTHETIC-CORRESPONDENCE-EVIDENCE" if global_pass
                          else "SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED")
    results["wall_clock_s"] = time.time() - t0
    results["determinism"] = {"cudnn.deterministic": True, "cudnn.benchmark": False,
                              "use_deterministic_algorithms": True,
                              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                              "no_grad": True, "eval": True, "dtype": "fp32"}
    results["protocol_deviations"] = "none"

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.append("forwards=%d (expected %d)" % (n_forward, sanity["forward_passes_expected"]))
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
