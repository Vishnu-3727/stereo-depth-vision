"""EXP-CORRESPONDENCE-ARCH-001 -- architecture-only control.

Preregistered in PREREGISTRATION.md, frozen before this file ran. Random
aggregation initialisations and the generator orbit frozen in
random_initialization_metadata.json / generator.json (hashes recorded in the
preregistration) before the preregistration was finalised.

Inference only. Read-only checkpoints. No training: this module imports no
optimizer, constructs none, and never calls .backward() or .step().

Only the aggregation module's 111,585 parameters differ between conditions. The
feature extractor, cost-volume construction, cost-volume tensor, mask, readout
and standardisation are held identical, and the pre-aggregation cost volume is
sha256-verified identical across every weight-set at every unit.
"""
from __future__ import annotations

import hashlib
import itertools
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

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.models.stereonet.aggregation import Aggregation               # noqa: E402
from src.datasets.kitti2015 import normalize                           # noqa: E402
from phase2.models import scaled_regression                            # noqa: E402
from phase2.viz import core                                            # noqa: E402

OUT = Path(__file__).resolve().parent
FOCUS_SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"
N_CANDIDATES = 12
EXPECTED_PARAM_COUNT = 111585
SEEDS = list(range(16))

CHECKPOINTS = {
    "NEG_shift_none": {
        "path": "phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth",
        "shift": "none", "seed": None,
        "sha256": "d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1"},
    "POS_6b_seed0": {
        "path": "phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
        "shift": "left", "seed": 0,
        "sha256": "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a"},
    "POS_6b_seed1": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
        "shift": "left", "seed": 1,
        "sha256": "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf"},
    "POS_6b_seed2": {
        "path": "phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
        "shift": "left", "seed": 2,
        "sha256": "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69"},
}
ORDER = ["NEG_shift_none", "POS_6b_seed0", "POS_6b_seed1", "POS_6b_seed2"]


class HardStop(RuntimeError):
    """PREREGISTRATION.md section 16."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_frozen():
    init_body = (OUT / "random_initialization_metadata.json").read_text(encoding="utf-8")
    gen_body = (OUT / "generator.json").read_text(encoding="utf-8")
    init_d = hashlib.sha256(init_body.encode("utf-8")).hexdigest()
    gen_d = hashlib.sha256(gen_body.encode("utf-8")).hexdigest()
    pre = (OUT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    for name, d in (("random_initialization_metadata.json", init_d), ("generator.json", gen_d)):
        if d not in pre:
            raise HardStop("%s hash %s is not the one recorded in PREREGISTRATION.md" % (name, d))
    return json.loads(init_body), json.loads(gen_body), init_d, gen_d


def build_model(key: str, device: str):
    spec = CHECKPOINTS[key]
    path = REPO_ROOT / spec["path"]
    actual = sha256_file(path)
    if actual != spec["sha256"]:                                   # K6
        raise HardStop("%s sha256 %s != preregistered %s" % (key, actual, spec["sha256"]))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rec = ckpt.get("config", {}).get("cost_volume_shift")
    if rec is not None and rec != spec["shift"]:
        raise HardStop("%s recorded shift %r != expected %r" % (key, rec, spec["shift"]))
    model = StereoNet(StereoNetConfig(cost_volume_shift=spec["shift"]))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    if type(model.regression).__name__ != "StandardisedDisparityRegression":
        raise HardStop("%s regression is %s" % (key, type(model.regression).__name__))
    if sum(p.numel() for p in model.regression.parameters()) != 0:  # readout must be param-free
        raise HardStop("%s readout has parameters" % key)
    if model.cost_volume.shift != spec["shift"] or model.config.num_disparities != N_CANDIDATES:
        raise HardStop("%s config mismatch" % key)
    if sum(p.numel() for p in model.aggregation.parameters()) != EXPECTED_PARAM_COUNT:  # K8
        raise HardStop("%s aggregation parameter count mismatch" % key)
    return model.eval().to(device), actual


def build_random_aggregations(frozen_init, device):
    """Reconstruct the frozen 16 random aggregations and verify every tensor hash
    against the pre-execution freeze (K5)."""
    mods, meta = {}, {e["seed"]: e for e in frozen_init["initialisations"]}
    for s in SEEDS:
        torch.manual_seed(s)
        agg = Aggregation(in_channels=32, channels=32, num_layers=4)
        if sum(p.numel() for p in agg.parameters()) != EXPECTED_PARAM_COUNT:     # K8
            raise HardStop("random seed %d parameter count mismatch" % s)
        recorded = {t["name"]: t["sha256"] for t in meta[s]["tensors"]}
        for name, p in agg.named_parameters():
            h = tensor_sha256(p)
            if h != recorded[name]:
                raise HardStop("random seed %d tensor %s hash %s != frozen %s"
                               % (s, name, h, recorded[name]))
        mods[s] = agg.eval().to(device)
    return mods


@torch.no_grad()
def orbit_for(model, volume, size, mask, perms):
    """Full 12-position orbit. Identity is NOT special-cased: every position,
    including m=0, goes through the identical index_select path."""
    s = np.empty(N_CANDIDATES, dtype=np.float64)
    arrs = {}
    ident_vol_diff = None
    for m in range(N_CANDIDATES):
        idx = torch.tensor(perms[str(m)], dtype=torch.long, device=volume.device)
        vp = volume.index_select(2, idx)
        if m == 0:
            ident_vol_diff = float((vp - volume).abs().max().cpu())              # K3
        d = model.regression(model.aggregation(vp), size)
        a = d[0, 0].detach().float().cpu().numpy().astype(np.float64)
        arrs[m] = a
        s[m] = float(np.median(a[mask]))
    return s, arrs, ident_vol_diff


def zscore(s: np.ndarray):
    """Preregistered: population std; zero variance -> z = 0 and degenerate."""
    sd = float(s.std())
    if sd == 0.0:
        return np.zeros_like(s), True
    return (s - s.mean()) / sd, False


def rho(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(np.sum(z1 * z2) / N_CANDIDATES)


def circ_tv(s: np.ndarray) -> float:
    order = np.argsort(s, kind="stable")
    r = np.empty(N_CANDIDATES)
    r[order] = np.arange(1, N_CANDIDATES + 1)
    for v in np.unique(s):
        i = np.where(s == v)[0]
        if i.size > 1:
            r[i] = r[i].mean()
    return float(sum(abs(r[(m + 1) % N_CANDIDATES] - r[m]) for m in range(N_CANDIDATES)))


def main() -> int:
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)                                      # K4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frozen_init, gen, init_d, gen_d = load_frozen()
    perms = gen["permutations_by_m"]
    for m in range(N_CANDIDATES):                                                 # K2
        if perms[str(m)] != [(k - m) % N_CANDIDATES for k in range(N_CANDIDATES)]:
            raise HardStop("generator permutation m=%d does not match (k-m) mod 12" % m)
    if frozen_init["seeds"] != SEEDS:
        raise HardStop("frozen seed list %r != %r" % (frozen_init["seeds"], SEEDS))

    log = ["EXP-CORRESPONDENCE-ARCH-001",
           "device=%s torch=%s numpy=%s" % (device, torch.__version__, np.__version__),
           "seeds=%r  random_init sha256=%s  generator sha256=%s" % (SEEDS, init_d, gen_d)]
    print("\n".join(log))

    randoms = build_random_aggregations(frozen_init, device)
    log.append("random aggregations reconstructed and hash-verified: %d" % len(randoms))

    results = {
        "experiment": "EXP-CORRESPONDENCE-ARCH-001",
        "title": "Architecture-only control for candidate-axis response structure",
        "kind": "inference-only; no training; only aggregation weights differ",
        "random_initialization_metadata_sha256": init_d,
        "generator_sha256": gen_d,
        "seeds": SEEDS, "n_candidates": N_CANDIDATES,
        "focus_scenes": FOCUS_SCENES, "split": SPLIT,
        "mask_rule": "GT valid and GT/16.0 in [2.0, 8.0]",
        "primary_statistic": "orbit-shape agreement rho = (1/12) sum_m z1_m z2_m, "
                             "z = within-unit z-score (population std) of s_m = median(disparity_initial[mask])",
        "decision_rule": "INSIDE: every rho in TR within [min RR, max RR]; "
                         "SEPARATED: max(TR) < min(RR); CASE A = INSIDE 12/12; "
                         "CASE B = SEPARATED 12/12; CASE C = otherwise",
        "p_value_used": False,
        "checkpoints": {}, "units": [], "gate": {},
    }

    neg_ok = True
    neg = {"arms": 0, "volume_max_abs_diff": 0.0, "disparity_max_abs_diff": 0.0,
           "identity_vs_original_max_abs_diff": 0.0, "violations": []}

    for key in ORDER:
        if key.startswith("POS") and not neg_ok:
            log.append("HARD STOP: negative control failed; positives not run")
            break
        model, digest = build_model(key, device)
        trained_agg = model.aggregation
        results["checkpoints"][key] = {
            "path": CHECKPOINTS[key]["path"], "sha256": digest,
            "shift": CHECKPOINTS[key]["shift"], "seed": CHECKPOINTS[key]["seed"],
            "regression": type(model.regression).__name__,
            "aggregation_param_count": EXPECTED_PARAM_COUNT,
            "trained_aggregation_sha256": hashlib.sha256("".join(
                tensor_sha256(p) for _, p in trained_agg.named_parameters()).encode()).hexdigest(),
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

            model.aggregation = trained_agg
            with torch.no_grad():
                lf = model.feature_extractor(L)
                rf = model.feature_extractor(R)
                volume = model.cost_volume(lf, rf)                 # intervention point
            if tuple(volume.shape) != (1, 32, 12, 23, 77):
                raise HardStop("unexpected volume shape %s" % (tuple(volume.shape),))
            vol_hash = tensor_sha256(volume)                        # K1
            H, W = scene.left.shape[:2]
            size = (H, W)

            orbits, degen, hashes = {}, {}, {}
            for wname, agg in [("trained", trained_agg)] + [("R%d" % s, randoms[s]) for s in SEEDS]:
                model.aggregation = agg
                s_vec, arrs, ivd = orbit_for(model, volume, size, mask, perms)
                hashes[wname] = tensor_sha256(volume)               # K1 re-check per weight-set
                if hashes[wname] != vol_hash:
                    raise HardStop("cost volume changed between weight-sets at %s/%s"
                                   % (key, scene.name))
                neg["identity_vs_original_max_abs_diff"] = max(
                    neg["identity_vs_original_max_abs_diff"], abs(ivd))
                if ivd != 0.0:                                      # K3
                    raise HardStop("identity index_select != original (%r) %s/%s/%s"
                                   % (ivd, key, scene.name, wname))
                z, dg = zscore(s_vec)
                orbits[wname] = {"s": s_vec, "z": z}
                degen[wname] = dg
                if is_neg:
                    base = arrs[0]
                    for m in range(N_CANDIDATES):
                        idx = torch.tensor(perms[str(m)], dtype=torch.long, device=volume.device)
                        vd = float((volume.index_select(2, idx) - volume).abs().max().cpu())
                        dd = float(np.abs(arrs[m] - base).max())
                        neg["arms"] += 1
                        neg["volume_max_abs_diff"] = max(neg["volume_max_abs_diff"], vd)
                        neg["disparity_max_abs_diff"] = max(neg["disparity_max_abs_diff"], dd)
                        if vd != 0.0 or dd != 0.0:
                            neg_ok = False
                            neg["violations"].append({"scene": scene.name, "weight_set": wname,
                                                      "m": m, "volume": vd, "disparity": dd})

            any_degen = any(degen.values())
            TR, RR = [], []
            if not any_degen:
                for s in SEEDS:
                    TR.append({"random_seed": s,
                               "rho": rho(orbits["trained"]["z"], orbits["R%d" % s]["z"])})
                for a, b in itertools.combinations(SEEDS, 2):
                    RR.append({"seed_a": a, "seed_b": b,
                               "rho": rho(orbits["R%d" % a]["z"], orbits["R%d" % b]["z"])})
            tr = np.array([x["rho"] for x in TR]) if TR else np.array([])
            rr = np.array([x["rho"] for x in RR]) if RR else np.array([])

            if any_degen:
                inside = separated = reverse = None
                cls = "DEGENERATE-NOT-CLASSIFIED"
            else:
                inside = bool(np.all((tr >= rr.min()) & (tr <= rr.max())))
                separated = bool(tr.max() < rr.min())
                reverse = bool(tr.min() > rr.max())
                cls = "SEPARATED" if separated else ("INSIDE" if inside else "CASE-C")

            unit = {
                "checkpoint": key, "checkpoint_seed": CHECKPOINTS[key]["seed"],
                "scene": scene.name, "scene_index": si, "valid_pixels": nmask,
                "is_negative_control": is_neg, "cost_volume_sha256": vol_hash,
                "cost_volume_identical_across_weight_sets": len(set(hashes.values())) == 1,
                "orbits": {w: {"s": list(orbits[w]["s"]),
                               "z": list(orbits[w]["z"]),
                               "degenerate": degen[w],
                               "range": float(orbits[w]["s"].max() - orbits[w]["s"].min()),
                               "s0": float(orbits[w]["s"][0]),
                               "circular_rank_TV": circ_tv(orbits[w]["s"]),
                               "four_arm_ordering_s11_s0_s1_s2": bool(
                                   orbits[w]["s"][11] < orbits[w]["s"][0]
                                   < orbits[w]["s"][1] < orbits[w]["s"][2])}
                           for w in orbits},
                "TR": TR, "RR": RR,
                "TR_stats": ({"n": int(tr.size), "min": float(tr.min()), "median": float(np.median(tr)),
                              "mean": float(tr.mean()), "max": float(tr.max())} if tr.size else None),
                "RR_stats": ({"n": int(rr.size), "min": float(rr.min()), "median": float(np.median(rr)),
                              "mean": float(rr.mean()), "max": float(rr.max())} if rr.size else None),
                "INSIDE": inside, "SEPARATED": separated, "reverse_separation": reverse,
                "classification": cls,
            }
            results["units"].append(unit)
            log.append("done %-14s %-13s mask=%d  cls=%s  TR[%s]  RR[%s]"
                       % (key, scene.name, nmask, cls,
                          "%.4f..%.4f" % (tr.min(), tr.max()) if tr.size else "degenerate",
                          "%.4f..%.4f" % (rr.min(), rr.max()) if rr.size else "degenerate"))
            print(log[-1])

        model.aggregation = trained_agg
        del model
        if is_neg and not neg_ok:
            break

    gate_pass = bool(neg_ok and neg["volume_max_abs_diff"] == 0.0
                     and neg["disparity_max_abs_diff"] == 0.0
                     and neg["identity_vs_original_max_abs_diff"] == 0.0)
    results["gate"] = {
        "algebraic_claim": "shift=none gives V(k) constant in k, so V[:,:,pi] == V for any pi",
        "arms_checked": neg["arms"],
        "volume_max_abs_diff": neg["volume_max_abs_diff"],
        "disparity_initial_max_abs_diff": neg["disparity_max_abs_diff"],
        "identity_index_select_vs_original_max_abs_diff": neg["identity_vs_original_max_abs_diff"],
        "violations": neg["violations"], "PASS": gate_pass,
    }

    pos = [u for u in results["units"] if not u["is_negative_control"]]
    n_inside = sum(1 for u in pos if u["INSIDE"])
    n_sep = sum(1 for u in pos if u["SEPARATED"])
    if not gate_pass:
        verdict = "HARD-STOP-NEGATIVE-CONTROL-FAILED"
    elif pos and n_inside == len(pos):
        verdict = "CASE-A-ARCHITECTURE-INDUCED"
    elif pos and n_sep == len(pos):
        verdict = "CASE-B-WEIGHT-DEPENDENT"
    else:
        verdict = "CASE-C-ARCHITECTURE-CONTROL-INCONCLUSIVE"

    results["summary"] = {
        "positive_units": len(pos), "INSIDE_units": n_inside, "SEPARATED_units": n_sep,
        "CASE_C_units": len(pos) - n_inside - n_sep,
        "cost_volume_identity_all_units": all(u["cost_volume_identical_across_weight_sets"]
                                              for u in results["units"]),
        "negative_gate_pass": gate_pass,
        "weight_sets_per_unit": 1 + len(SEEDS),
        "arms_total": sum(1 for _ in results["units"]) * (1 + len(SEEDS)) * N_CANDIDATES,
    }
    results["VERDICT"] = verdict
    results["wall_clock_s"] = time.time() - t0
    results["determinism"] = {"cudnn.deterministic": True, "cudnn.benchmark": False,
                              "use_deterministic_algorithms": True,
                              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                              "no_grad": True, "eval": True, "dtype": "fp32"}

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.append("VERDICT: " + verdict)
    log.append("wall_clock_s=%.2f" % results["wall_clock_s"])
    (OUT / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\nVERDICT:", verdict)
    print(json.dumps(results["summary"], indent=2))
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
