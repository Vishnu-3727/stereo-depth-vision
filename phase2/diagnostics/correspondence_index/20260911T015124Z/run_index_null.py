"""EXP-CORRESPONDENCE-INDEX-002 -- empirical permutation null for candidate-axis
sensitivity.

Preregistered in PREREGISTRATION.md, frozen before this file ran. Permutations
frozen in permutations.json (sha256 recorded in the preregistration) before the
preregistration was finalised.

Inference only. Read-only checkpoints. No training: this module imports no
optimizer, constructs none, and never calls .backward() or .step().

Derived from the INDEX-001 harness (same model construction, same cost-volume
access point, same index_select mechanism, same mask, same np.median
aggregation), extended with the empirical null.
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

CUBLAS_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig   # noqa: E402
from src.datasets.kitti2015 import normalize                  # noqa: E402
from phase2.models import scaled_regression                   # noqa: E402
from phase2.viz import core                                   # noqa: E402

OUT = Path(__file__).resolve().parent
FOCUS_SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"
N_CANDIDATES = 12

# ---------------------------------------------------------------- preregistered
ALPHA = 0.01                 # PREREGISTRATION.md section 10.1
PCT_LOW, PCT_HIGH = 1.0, 99.0

CHECKPOINTS = {
    "NEG_shift_none": {
        "path": "phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth",
        "shift": "none", "seed": None,
        "sha256": "d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1",
    },
    "POS_6b_seed0": {
        "path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
        "shift": "left", "seed": 0,
        "sha256": "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a",
    },
    "POS_6b_seed1": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
        "shift": "left", "seed": 1,
        "sha256": "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf",
    },
    "POS_6b_seed2": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
        "shift": "left", "seed": 2,
        "sha256": "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69",
    },
}
ORDER = ["NEG_shift_none", "POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"]


class HardStop(RuntimeError):
    """Raised for any PREREGISTRATION.md section 16 condition."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_frozen_permutations() -> dict:
    body = (OUT / "permutations.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = (OUT / "permutations.sha256").read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise HardStop("permutations.json hash %s != frozen %s" % (digest, recorded))
    pre = (OUT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    if digest not in pre:
        raise HardStop("permutations.json hash is not the one recorded in PREREGISTRATION.md")
    return json.loads(body)


def build_model(key: str, device: str):
    spec = CHECKPOINTS[key]
    path = REPO_ROOT / spec["path"]
    actual = sha256_file(path)
    if actual != spec["sha256"]:                      # section 16: checkpoint differs
        raise HardStop("%s sha256 %s != preregistered %s" % (key, actual, spec["sha256"]))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rec = ckpt.get("config", {}).get("cost_volume_shift")
    if rec is not None and rec != spec["shift"]:
        raise HardStop("%s: recorded shift %r != expected %r" % (key, rec, spec["shift"]))
    model = StereoNet(StereoNetConfig(cost_volume_shift=spec["shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    if type(model.regression).__name__ != "StandardisedDisparityRegression":
        raise HardStop("%s: regression is %s" % (key, type(model.regression).__name__))
    if model.cost_volume.shift != spec["shift"]:
        raise HardStop("%s: cost_volume.shift mismatch" % key)
    if model.config.num_disparities != N_CANDIDATES:
        raise HardStop("%s: num_disparities %d" % (key, model.config.num_disparities))
    return model.eval().to(device), actual


@torch.no_grad()
def arm(model, volume: torch.Tensor, idx: torch.Tensor, size) -> np.ndarray:
    """One permutation arm: index_select -> aggregation -> standardised regression.

    Identity goes through this identical path (PREREGISTRATION.md section A9).
    Refinement is never invoked.
    """
    vp = volume.index_select(2, idx)
    cost = model.aggregation(vp)
    d = model.regression(cost, size)
    return d[0, 0].detach().float().cpu().numpy().astype(np.float64)


def median_masked(a: np.ndarray, mask: np.ndarray) -> float:
    """d(x) exactly as INDEX-001 computed it."""
    return float(np.median(a[mask]))


def empirical_null_test(s_ord: float, s_null: np.ndarray) -> dict:
    """PREREGISTRATION.md section 10.1, verbatim. No parametric assumption."""
    m = int(s_null.size)
    ge = int(np.sum(s_null >= s_ord))
    le = int(np.sum(s_null <= s_ord))
    p_ge = (1.0 + ge) / (m + 1.0)
    p_le = (1.0 + le) / (m + 1.0)
    p_two = float(min(1.0, 2.0 * min(p_ge, p_le)))
    p01, p99 = (float(x) for x in np.percentile(s_null, [PCT_LOW, PCT_HIGH], method="linear"))
    outside = bool(s_ord < p01 or s_ord > p99)
    return {
        "m": m,
        "n_null_ge_ordered": ge,
        "n_null_le_ordered": le,
        "p_one_sided_ge": float(p_ge),
        "p_one_sided_le": float(p_le),
        "p_two_sided": p_two,
        "null_p01": p01,
        "null_p99": p99,
        "outside_central_99": outside,
        # rank of the ordered value among the null, 1 = smallest
        "empirical_rank": int(np.sum(s_null < s_ord)) + 1,
        "rank_of": m + 1,
        "null_pass": bool(p_two <= ALPHA and outside),
    }


def main() -> int:
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frozen = load_frozen_permutations()
    ordered_arms = frozen["ordered_arms"]
    randoms = frozen["random_permutations"]
    triples = frozen["null_triples"]
    M = frozen["m_triples"]
    if len(randoms) != 3 * M:
        raise HardStop("permutation count %d != 3*M" % len(randoms))

    log = ["EXP-CORRESPONDENCE-INDEX-002",
           "device=%s torch=%s numpy=%s" % (device, torch.__version__, np.__version__),
           "frozen seed=%d  n_random=%d  M=%d" % (frozen["frozen_seed"], len(randoms), M)]
    print("\n".join(log))

    results = {
        "experiment": "EXP-CORRESPONDENCE-INDEX-002",
        "title": "Empirical Permutation Null for Candidate-Axis Sensitivity",
        "kind": "inference-only; no training; no image or feature modification",
        "claim_level_targeted": "C -- candidate-coordinate sensitivity",
        "claim_levels_NOT_established": ["D geometric correspondence",
                                         "E genuine disparity search"],
        "frozen_seed": frozen["frozen_seed"],
        "permutations_sha256": hashlib.sha256(
            (OUT / "permutations.json").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        "m_triples": M,
        "n_random_permutations": len(randoms),
        "ordered_arms": ordered_arms,
        "alpha": ALPHA,
        "percentiles": [PCT_LOW, PCT_HIGH],
        "mask_rule": "GT valid and GT/16.0 in [2.0, 8.0]",
        "primary_statistic": "S_order = (d(+2) - d(identity)) + (d(+1) - d(-1)); "
                             "d(x) = np.median(disparity_initial[mask])",
        "secondary_statistic": "S_abs = |d(+2)-d(id)| + |d(+1)-d(id)| + |d(-1)-d(id)|",
        "endpoint": "disparity_initial only; refinement never invoked",
        "focus_scenes": FOCUS_SCENES,
        "split": SPLIT,
        "checkpoints": {},
        "units": [],
        "gate": {},
        "screening_applied": False,
        "null_draws_removed": 0,
    }

    ord_names = ["identity", "m_plus_1", "m_plus_2", "m_minus_1"]
    neg_ok = True
    neg_detail = {"volume_max_abs_diff": 0.0, "disparity_max_abs_diff": 0.0,
                  "identity_vs_original_max_abs_diff": 0.0,
                  "arms_checked": 0, "violations": []}

    for key in ORDER:
        if key.startswith("POS") and not neg_ok:
            log.append("HARD STOP: negative control failed; positive checkpoints not run")
            break
        model, digest = build_model(key, device)
        results["checkpoints"][key] = {
            "path": CHECKPOINTS[key]["path"], "sha256": digest,
            "shift": CHECKPOINTS[key]["shift"], "seed": CHECKPOINTS[key]["seed"],
            "regression": type(model.regression).__name__,
            "num_disparities": model.config.num_disparities,
        }
        is_neg = key == "NEG_shift_none"

        for si in FOCUS_SCENES:
            scene = core.load_scene(si, split=SPLIT)
            L = torch.from_numpy(normalize(scene.left)).to(device)
            R = torch.from_numpy(normalize(scene.right)).to(device)
            gt = scene.gt_disparity.astype(np.float64)
            mask = (gt > 0) & (gt / 16.0 >= 2.0) & (gt / 16.0 <= 8.0)
            nmask = int(mask.sum())
            if nmask == 0:
                raise HardStop("empty mask for scene %s" % scene.name)

            with torch.no_grad():
                lf = model.feature_extractor(L)
                rf = model.feature_extractor(R)
                volume = model.cost_volume(lf, rf)          # intervention point
            if tuple(volume.shape) != (1, 32, 12, 23, 77):
                raise HardStop("unexpected volume shape %s" % (tuple(volume.shape),))
            H, W = scene.left.shape[:2]
            size = (H, W)

            # ---- ordered arms (identity through the same index_select path) ----
            d_ord, arr_id = {}, None
            for nm in ord_names:
                idx = torch.tensor(ordered_arms[nm], dtype=torch.long, device=volume.device)
                a = arm(model, volume, idx, size)
                if nm == "identity":
                    arr_id = a
                    vid = volume.index_select(2, idx)
                    dv = float((vid - volume).abs().max().cpu())
                    neg_detail["identity_vs_original_max_abs_diff"] = max(
                        neg_detail["identity_vs_original_max_abs_diff"], dv)
                    if dv != 0.0:
                        raise HardStop("identity index_select != original (%r) scene %s"
                                       % (dv, scene.name))
                d_ord[nm] = median_masked(a, mask)
                if is_neg:
                    idxo = torch.tensor(ordered_arms[nm], dtype=torch.long, device=volume.device)
                    vd = float((volume.index_select(2, idxo) - volume).abs().max().cpu())
                    dd = float(np.abs(a - arr_id).max())
                    neg_detail["volume_max_abs_diff"] = max(neg_detail["volume_max_abs_diff"], vd)
                    neg_detail["disparity_max_abs_diff"] = max(neg_detail["disparity_max_abs_diff"], dd)
                    neg_detail["arms_checked"] += 1
                    if vd != 0.0 or dd != 0.0:
                        neg_ok = False
                        neg_detail["violations"].append(
                            {"scene": scene.name, "arm": nm, "volume": vd, "disparity": dd})

            # ---- random null arms ----
            d_rand = np.empty(len(randoms), dtype=np.float64)
            for i, p in enumerate(randoms):
                idx = torch.tensor(p, dtype=torch.long, device=volume.device)
                a = arm(model, volume, idx, size)
                d_rand[i] = median_masked(a, mask)
                if is_neg:
                    vd = float((volume.index_select(2, idx) - volume).abs().max().cpu())
                    dd = float(np.abs(a - arr_id).max())
                    neg_detail["volume_max_abs_diff"] = max(neg_detail["volume_max_abs_diff"], vd)
                    neg_detail["disparity_max_abs_diff"] = max(neg_detail["disparity_max_abs_diff"], dd)
                    neg_detail["arms_checked"] += 1
                    if vd != 0.0 or dd != 0.0:
                        neg_ok = False
                        neg_detail["violations"].append(
                            {"scene": scene.name, "arm": "random[%d]" % i,
                             "volume": vd, "disparity": dd})

            # ---- preregistered statistics ----
            s_ord = ((d_ord["m_plus_2"] - d_ord["identity"])
                     + (d_ord["m_plus_1"] - d_ord["m_minus_1"]))
            s_abs = (abs(d_ord["m_plus_2"] - d_ord["identity"])
                     + abs(d_ord["m_plus_1"] - d_ord["identity"])
                     + abs(d_ord["m_minus_1"] - d_ord["identity"]))
            did = d_ord["identity"]
            s_null = np.array([(d_rand[a_] - did) + (d_rand[b_] - d_rand[c_])
                               for a_, b_, c_ in triples], dtype=np.float64)
            s_abs_null = np.array([abs(d_rand[a_] - did) + abs(d_rand[b_] - did)
                                   + abs(d_rand[c_] - did) for a_, b_, c_ in triples],
                                  dtype=np.float64)

            test = empirical_null_test(s_ord, s_null)
            order_pass = bool(d_ord["m_plus_2"] > d_ord["m_plus_1"]
                              > d_ord["identity"] > d_ord["m_minus_1"])

            unit = {
                "checkpoint": key, "seed": CHECKPOINTS[key]["seed"],
                "scene": scene.name, "scene_index": si, "valid_pixels": nmask,
                "is_negative_control": is_neg,
                "d_minus1": d_ord["m_minus_1"], "d_identity": d_ord["identity"],
                "d_plus1": d_ord["m_plus_1"], "d_plus2": d_ord["m_plus_2"],
                "delta_minus1": d_ord["m_minus_1"] - did,
                "delta_plus1": d_ord["m_plus_1"] - did,
                "delta_plus2": d_ord["m_plus_2"] - did,
                "S_order": float(s_ord), "S_abs": float(s_abs),
                "null": {
                    "n": int(s_null.size),
                    "mean": float(s_null.mean()), "median": float(np.median(s_null)),
                    "std": float(s_null.std(ddof=1)) if s_null.size > 1 else 0.0,
                    "std_population": float(s_null.std()),
                    "min": float(s_null.min()), "max": float(s_null.max()),
                    "p01": test["null_p01"], "p99": test["null_p99"],
                    "degenerate": bool(float(s_null.std()) == 0.0),
                },
                "null_abs": {
                    "mean": float(s_abs_null.mean()),
                    "median": float(np.median(s_abs_null)),
                    "std": float(s_abs_null.std(ddof=1)) if s_abs_null.size > 1 else 0.0,
                    "p01": float(np.percentile(s_abs_null, PCT_LOW, method="linear")),
                    "p99": float(np.percentile(s_abs_null, PCT_HIGH, method="linear")),
                },
                "test": test,
                "order_pass": order_pass,
                "d_random_median": float(np.median(d_rand)),
                "d_random_mean": float(d_rand.mean()),
                "d_random_min": float(d_rand.min()),
                "d_random_max": float(d_rand.max()),
            }
            results["units"].append(unit)
            log.append("done %s %s mask=%d S_order=%+.4f p2=%.6f rank=%d/%d "
                       "null[%.4f,%.4f] order=%s"
                       % (key, scene.name, nmask, s_ord, test["p_two_sided"],
                          test["empirical_rank"], test["rank_of"],
                          test["null_p01"], test["null_p99"], order_pass))
            print(log[-1])

        del model
        if is_neg:
            if not neg_ok:
                break

    # ------------------------------------------------------------------ verdict
    gate_pass = bool(neg_ok
                     and neg_detail["volume_max_abs_diff"] == 0.0
                     and neg_detail["disparity_max_abs_diff"] == 0.0
                     and neg_detail["identity_vs_original_max_abs_diff"] == 0.0)
    results["gate"] = {
        "algebraic_claim": "shift=none gives V(k)=left-right for all k, so V[:,:,pi] == V for any pi",
        "arms_checked": neg_detail["arms_checked"],
        "volume_max_abs_diff": neg_detail["volume_max_abs_diff"],
        "disparity_initial_max_abs_diff": neg_detail["disparity_max_abs_diff"],
        "identity_index_select_vs_original_max_abs_diff":
            neg_detail["identity_vs_original_max_abs_diff"],
        "violations": neg_detail["violations"],
        "PASS": gate_pass,
    }

    pos = [u for u in results["units"] if not u["is_negative_control"]]
    degenerate = [u for u in pos if u["null"]["degenerate"]]
    n_null_pass = sum(1 for u in pos if u["test"]["null_pass"])
    n_order_pass = sum(1 for u in pos if u["order_pass"])
    global_null = bool(pos and n_null_pass == len(pos))
    global_order = bool(pos and n_order_pass == len(pos))

    if not gate_pass:
        verdict = "HARD-STOP-NEGATIVE-CONTROL-FAILED"
    elif degenerate:
        verdict = "NULL-DESIGN-INCONCLUSIVE"
    elif global_null and global_order:
        verdict = "CANDIDATE-COORDINATE-SENSITIVITY-ESTABLISHED"
    else:
        verdict = "CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED"

    results["summary"] = {
        "positive_units": len(pos),
        "null_pass_units": n_null_pass,
        "null_pass_fraction": (n_null_pass / len(pos)) if pos else None,
        "order_pass_units": n_order_pass,
        "order_pass_fraction": (n_order_pass / len(pos)) if pos else None,
        "evaluation_level": "all positive (checkpoint, scene) units must pass both",
        "global_null_pass": global_null,
        "global_order_pass": global_order,
        "negative_gate_pass": gate_pass,
        "degenerate_null_units": len(degenerate),
        "min_attainable_two_sided_p": 2.0 / (M + 1.0),
    }
    results["VERDICT"] = verdict
    results["wall_clock_s"] = time.time() - t0
    results["determinism"] = {
        "cudnn.deterministic": True, "cudnn.benchmark": False,
        "use_deterministic_algorithms": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "no_grad": True, "eval": True, "dtype": "fp32",
    }

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.append("VERDICT: " + verdict)
    log.append("wall_clock_s=%.2f" % results["wall_clock_s"])
    (OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\nVERDICT:", verdict)
    print(json.dumps(results["summary"], indent=2))
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
