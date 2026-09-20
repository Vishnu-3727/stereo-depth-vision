"""EXP-CORRESPONDENCE-ARCH-RATE-001 -- how often do UNTRAINED aggregation weights
reproduce the synthetic horizontal correspondence signature?

Executes EXACTLY the protocol frozen in PREREGISTRATION.md and the three
*_spec.json files. All three digests are re-verified over FILE BYTES before the
first model instantiation.

THE RESULT IS A RATE. There is no pass/fail rule anywhere in this file.

NO TRAINED AGGREGATION IS EVER EXECUTED. The trained feature extractor is loaded
because the protocol requires it frozen -- it supplies the cost volume. Every
forward pass uses a freshly seeded RANDOM aggregation, asserted at run time.

Inference only. No optimizer is imported or constructed; no .backward(); no .step().
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
from src.models.stereonet.aggregation import Aggregation               # noqa: E402
from src.datasets.kitti2015 import normalize                           # noqa: E402
from phase2.models import scaled_regression                            # noqa: E402
from phase2.viz import core                                            # noqa: E402

OUT = Path(__file__).resolve().parent
W, H, STRIDE = 1232, 368, 16
TMAX = 96
CROP_W, CROP_H = W - TMAX, H - TMAX
BORDER = 64
EXPECTED_MASK = 145152
T_LEVELS = [0, 16, 32, 48, 64, 80, 96]
T_FIT = [16, 32, 48, 64, 80, 96]
AXES = ["horizontal", "vertical"]
N_SEEDS = 32
AGG_PARAMS = 111585
RAIL_LOW, RAIL_HIGH, FLAT_RANGE = 1.0, 10.0, 1.0      # grid-derived, frozen

EXTRACTORS = {
    "POS_6b_seed0": {"path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
                     "sha256": "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a"},
    "POS_6b_seed1": {"path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
                     "sha256": "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf"},
    "POS_6b_seed2": {"path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
                     "sha256": "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69"},
}
ORDER = ["POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"]


class HardStop(RuntimeError):
    """PREREGISTRATION.md section 11. Every one of these HALTS execution."""


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


def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    if y0 < 0 or x0 < 0 or y0 + CROP_H > H or x0 + CROP_W > W:
        raise HardStop("crop window (%d,%d) leaves the source -> would need fill" % (y0, x0))
    out = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if out.shape[:2] != (CROP_H, CROP_W):
        raise HardStop("crop produced %s" % (out.shape[:2],))
    return out


def right_origin(axis: str, t: int) -> tuple[int, int]:
    """Frozen: a_R = a_L + t. The axis is an argument, not a branch."""
    if axis == "horizontal":
        return 0, t
    if axis == "vertical":
        return t, 0
    raise HardStop("unknown axis %r" % axis)


def free_slope(m: dict) -> float:
    """FREE-INTERCEPT OLS slope of m(t) on t/16 over T_FIT. NOT through the origin."""
    x = np.array([t / STRIDE for t in T_FIT], dtype=np.float64)
    y = np.array([m[t] for t in T_FIT], dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    return float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))


def classify(mh: dict, mv: dict) -> tuple[str, dict]:
    """Frozen ordered taxonomy. First match wins. CLIPPED is evaluated SECOND,
    before every ordinary failure class, so a railed response can never be
    folded into ordinary failure."""
    vals_h = [mh[t] for t in T_LEVELS]
    vals_v = [mv[t] for t in T_LEVELS]
    if not (all(np.isfinite(vals_h)) and all(np.isfinite(vals_v))):
        raise HardStop("non-finite response -> implementation fault")
    lo, hi = min(vals_h), max(vals_h)
    a_h, a_v = free_slope(mh), free_slope(mv)
    diffs = [vals_h[i + 1] - vals_h[i] for i in range(len(vals_h) - 1)]
    Q1 = bool(a_h > 0)
    Q2 = bool(all(d > 0 for d in diffs))
    Q3 = bool(abs(a_h) > abs(a_v))
    facts = {"alpha_h": a_h, "alpha_v": a_v, "min_h": lo, "max_h": hi,
             "range_h": hi - lo, "first_diffs": diffs, "Q1": Q1, "Q2": Q2, "Q3": Q3}
    # --- frozen precedence -------------------------------------------------
    if lo <= RAIL_LOW or hi >= RAIL_HIGH:
        return "CLIPPED", facts
    if (hi - lo) < FLAT_RANGE:
        return "FLAT", facts
    if a_h < 0:
        return "ANTI_CORRELATED", facts
    if not Q2:
        return "NON_MONOTONE", facts
    if not Q3:
        return "NON_SPECIFIC", facts
    return "QUALIFYING", facts


def main() -> int:
    t0 = time.time()

    # ---- freeze verification BEFORE the first model instantiation ---------
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
    rspec = json.loads((OUT / "randomisation_spec.json").read_text(encoding="utf-8"))
    kspec = json.loads((OUT / "classification_spec.json").read_text(encoding="utf-8"))
    if cspec["t_levels_px"] != T_LEVELS or cspec["axes"] != AXES:
        raise HardStop("construction spec mismatch")
    if rspec["seeds"] != list(range(N_SEEDS)):
        raise HardStop("seed list mismatch")
    if kspec["classification_order"] != ["NONFINITE", "CLIPPED", "FLAT", "ANTI_CORRELATED",
                                         "NON_MONOTONE", "NON_SPECIFIC", "QUALIFYING"]:
        raise HardStop("classification order mismatch")
    SCENES = kspec["scenes"]
    used = set(kspec["prior_scene_sets_excluded"]["GEOM-001"]) | \
        set(kspec["prior_scene_sets_excluded"]["GEOM-002"])
    if set(SCENES) & used:
        raise HardStop("scene set overlaps a prior correspondence experiment")
    if any(t % STRIDE for t in T_LEVELS):
        raise HardStop("a t level is not a multiple of the stride")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log = ["EXP-CORRESPONDENCE-ARCH-RATE-001",
           "device=%s torch=%s numpy=%s" % (device, torch.__version__, np.__version__),
           "freeze verified: construction %s | randomisation %s | classification %s"
           % (frozen["construction_spec.json"][:12], frozen["randomisation_spec.json"][:12],
              frozen["classification_spec.json"][:12]),
           "scenes %r %r  seeds 0..%d" % (SCENES, kspec["scene_names"], N_SEEDS - 1)]
    print("\n".join(log))

    # ---- build the 32 random aggregations ONCE ---------------------------
    randoms, seed_hashes = {}, {}
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        agg = Aggregation(in_channels=32, channels=32, num_layers=4)
        if sum(p.numel() for p in agg.parameters()) != AGG_PARAMS:
            raise HardStop("seed %d aggregation parameter count mismatch" % s)
        seed_hashes[s] = {n: tensor_sha256(p) for n, p in agg.named_parameters()}
        randoms[s] = agg.eval().to(device)
    for name in seed_hashes[0]:
        seen = {}
        for s in range(N_SEEDS):
            h = seed_hashes[s][name]
            if h in seen:
                raise HardStop("identical tensor %s across seeds %d,%d" % (name, seen[h], s))
            seen[h] = s
    log.append("built %d random aggregations, all tensor hashes distinct" % N_SEEDS)

    # ---- mask: purely geometric, no GT -----------------------------------
    mask = np.zeros((CROP_H, CROP_W), dtype=bool)
    mask[BORDER:CROP_H - BORDER, BORDER:CROP_W - BORDER] = True
    if int(mask.sum()) != EXPECTED_MASK:
        raise HardStop("mask count %d != frozen %d" % (int(mask.sum()), EXPECTED_MASK))

    results = {
        "experiment": "EXP-CORRESPONDENCE-ARCH-RATE-001",
        "kind": "descriptive measurement; the result is a RATE, not a verdict",
        "construction_spec_sha256": frozen["construction_spec.json"],
        "randomisation_spec_sha256": frozen["randomisation_spec.json"],
        "classification_spec_sha256": frozen["classification_spec.json"],
        "t_levels_px": T_LEVELS, "t_fit_levels_px": T_FIT, "axes": AXES,
        "n_seeds": N_SEEDS, "rate_resolution": 1.0 / N_SEEDS,
        "scenes": SCENES, "scene_names": kspec["scene_names"],
        "extractors": ORDER, "crop_size": [CROP_W, CROP_H],
        "mask_rule": kspec["mask"]["definition_crop_coords"],
        "statistic": "alpha = FREE-INTERCEPT OLS slope of m(t) on t/16 over T_FIT",
        "classification_order": kspec["classification_order"],
        "thresholds": {"RAIL_LOW": RAIL_LOW, "RAIL_HIGH": RAIL_HIGH, "FLAT_RANGE": FLAT_RANGE,
                       "provenance": "candidate grid spacing (1 candidate); NOT from any observation"},
        "trained_aggregation_executed": False,
        "pass_fail_rule": None,
        "checkpoints": {}, "units": [], "sanity": {},
    }

    n_forward = 0
    n_volumes = 0
    sanity = {"crop_shapes": set(), "volume_shapes": set(),
              "left_crop_hashes_per_scene": {}, "agg_is_random_asserted": 0}

    for ekey in ORDER:
        spec = EXTRACTORS[ekey]
        path = REPO_ROOT / spec["path"]
        actual = sha256_bytes(path)
        if actual != spec["sha256"]:
            raise HardStop("%s sha256 mismatch" % ekey)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = StereoNet(StereoNetConfig(cost_volume_shift="left"))
        scaled_regression.apply_to(model)
        model.load_state_dict(ckpt["model"])
        if sum(p.numel() for p in model.regression.parameters()) != 0:
            raise HardStop("%s readout has parameters" % ekey)
        if model.cost_volume.shift != "left" or model.config.num_disparities != 12:
            raise HardStop("%s config mismatch" % ekey)
        trained_agg_hash = hashlib.sha256("".join(
            tensor_sha256(p) for _, p in model.aggregation.named_parameters()).encode()).hexdigest()
        model = model.eval().to(device)
        results["checkpoints"][ekey] = {"path": spec["path"], "sha256": actual,
                                        "role": "FEATURE EXTRACTOR ONLY",
                                        "trained_aggregation_sha256": trained_agg_hash,
                                        "trained_aggregation_executed": False}

        for si in SCENES:
            sc = core.load_scene(si, split=kspec["split"])
            if sc.left.shape[:2] != (H, W):
                raise HardStop("scene %s is %s" % (sc.name, sc.left.shape[:2]))
            left_c = crop(sc.left, 0, 0)
            sanity["crop_shapes"].add(left_c.shape[:2])
            sanity["left_crop_hashes_per_scene"].setdefault(sc.name, set()).add(arr_sha256(left_c))
            L = torch.from_numpy(normalize(left_c)).to(device)

            # ---- cache the 13 cost volumes; they do NOT depend on the seed ----
            vols = {}
            with torch.no_grad():
                lf = model.feature_extractor(L)
                for axis in AXES:
                    for t in T_LEVELS:
                        if t == 0 and axis == "vertical":
                            vols[(axis, 0)] = vols[("horizontal", 0)]
                            continue
                        y0, x0 = right_origin(axis, t)
                        right_c = crop(sc.left, y0, x0)
                        sanity["crop_shapes"].add(right_c.shape[:2])
                        Rt = torch.from_numpy(normalize(right_c)).to(device)
                        rf = model.feature_extractor(Rt)
                        v = model.cost_volume(lf, rf)
                        if tuple(v.shape) != (1, 32, 12, 17, 71):
                            raise HardStop("unexpected volume shape %s" % (tuple(v.shape),))
                        sanity["volume_shapes"].add(tuple(v.shape))
                        vols[(axis, t)] = v
                        n_volumes += 1

            for s in range(N_SEEDS):
                model.aggregation = randoms[s]                       # RANDOM, never trained
                if hashlib.sha256("".join(tensor_sha256(p) for _, p in
                                          model.aggregation.named_parameters()).encode()
                                  ).hexdigest() == trained_agg_hash:
                    raise HardStop("trained aggregation would have been executed")
                sanity["agg_is_random_asserted"] += 1
                mh, mv = {}, {}
                with torch.no_grad():
                    for axis in AXES:
                        for t in T_LEVELS:
                            d = model.regression(model.aggregation(vols[(axis, t)]),
                                                 (CROP_H, CROP_W))
                            a = d[0, 0].detach().float().cpu().numpy().astype(np.float64)
                            n_forward += 1
                            val = float(np.median(a[mask]))
                            (mh if axis == "horizontal" else mv)[t] = val
                cls, facts = classify(mh, mv)
                results["units"].append({
                    "extractor": ekey, "scene": sc.name, "scene_index": si, "seed": s,
                    "m_h": {str(t): mh[t] for t in T_LEVELS},
                    "m_v": {str(t): mv[t] for t in T_LEVELS},
                    "alpha_h": facts["alpha_h"], "alpha_v": facts["alpha_v"],
                    "min_h": facts["min_h"], "max_h": facts["max_h"],
                    "range_h": facts["range_h"], "first_diffs": facts["first_diffs"],
                    "Q1": facts["Q1"], "Q2": facts["Q2"], "Q3": facts["Q3"],
                    "classification": cls,
                })
            log.append("done %-14s %-13s  %d seeds" % (ekey, sc.name, N_SEEDS))
            print(log[-1])
        del model

    # ---- the RATE (no pass/fail anywhere) --------------------------------
    classes = ["CLIPPED", "FLAT", "ANTI_CORRELATED", "NON_MONOTONE",
               "NON_SPECIFIC", "QUALIFYING"]
    tally = {c: sum(1 for u in results["units"] if u["classification"] == c) for c in classes}
    n_units = len(results["units"])
    n_qual = tally["QUALIFYING"]

    by_ext = {e: {"n": 0, "qualifying": 0} for e in ORDER}
    by_scene = {n: {"n": 0, "qualifying": 0} for n in kspec["scene_names"]}
    for u in results["units"]:
        by_ext[u["extractor"]]["n"] += 1
        by_scene[u["scene"]]["n"] += 1
        if u["classification"] == "QUALIFYING":
            by_ext[u["extractor"]]["qualifying"] += 1
            by_scene[u["scene"]]["qualifying"] += 1
    for d in (by_ext, by_scene):
        for k in d:
            d[k]["rate"] = d[k]["qualifying"] / d[k]["n"] if d[k]["n"] else None

    ah = np.array([u["alpha_h"] for u in results["units"]])
    av = np.array([u["alpha_v"] for u in results["units"]])

    sanity["crop_shapes"] = sorted(str(x) for x in sanity["crop_shapes"])
    sanity["volume_shapes"] = sorted(str(x) for x in sanity["volume_shapes"])
    sanity["left_crop_distinct_per_scene"] = {k: len(v) for k, v in
                                              sanity.pop("left_crop_hashes_per_scene").items()}
    sanity["forward_passes"] = n_forward
    sanity["forward_passes_expected"] = n_units * len(T_LEVELS) * 2 - n_units  # t=0 shared
    sanity["cost_volumes_built"] = n_volumes
    sanity["training_occurred"] = False
    sanity["trained_aggregation_executed"] = False
    results["sanity"] = sanity
    results["result"] = {
        "n_units": n_units,
        "QUALIFYING_count": n_qual,
        "QUALIFYING_rate": n_qual / n_units,
        "rate_resolution": 1.0 / N_SEEDS,
        "tally": tally,
        "by_extractor": by_ext, "by_scene": by_scene,
        "alpha_h": {"min": float(ah.min()), "median": float(np.median(ah)),
                    "mean": float(ah.mean()), "max": float(ah.max()),
                    "std": float(ah.std(ddof=1))},
        "alpha_v": {"min": float(av.min()), "median": float(np.median(av)),
                    "mean": float(av.mean()), "max": float(av.max()),
                    "std": float(av.std(ddof=1))},
        "note": "THIS IS A RATE, NOT A VERDICT. No pass/fail rule exists.",
    }
    results["wall_clock_s"] = time.time() - t0
    results["determinism"] = {"cudnn.deterministic": True, "cudnn.benchmark": False,
                              "use_deterministic_algorithms": True,
                              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                              "no_grad": True, "eval": True, "dtype": "fp32"}
    results["protocol_deviations"] = "none"

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.append("volumes=%d arms=%d" % (n_volumes, n_forward))
    log.append("QUALIFYING rate = %d/%d = %.4f" % (n_qual, n_units, n_qual / n_units))
    log.append("tally: " + json.dumps(tally))
    log.append("wall_clock_s=%.2f" % results["wall_clock_s"])
    (OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8", newline="")
    print("\nQUALIFYING rate = %d/%d = %.4f" % (n_qual, n_units, n_qual / n_units))
    print(json.dumps(tally, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
