#!/usr/bin/env python
"""Gradient probe for the M1 term (0.1*CE) on REAL training data.

Purpose: before spending GPU on the F1M pilot, decide whether the M1 term
(``0.1*ce``, see the patched ``bundle_f1m/scripts/finetune_pilot.py``) delivers
usable gradient on real data, given aggregated_cost std ~5e14 on synthetic
input and Adam eps = 1e-8.

What it does:
  - Builds the model and loads weights EXACTLY as bundle_f1m finetune_pilot
    does (config parsed from the bundle file, strict load). Two subjects:
      (A) ARM-P Stage-1 init (bundle_f1m/checkpoints, sha 3ae6fb3b...)
      (B) E3 seed0 final (stage_e_recipe/kaggle/e3_output/seed0/..., sha 82e58bc4...)
  - Data: 8 real training batches of batch size 2 built with the SAME dataset
    class, crop, augmentation and seed-0 generator as finetune_pilot.py uses
    (hailo_calib / training split under data/kitti2015). The dataset class is
    reused by importing the bundle's finetune_pilot module (not reimplemented).
  - GPU (cuda) if available, model.train() mode, same determinism settings as
    finetune_pilot (seed_all(0); finetune_pilot does NOT enable deterministic
    algorithms, so neither do we).
  - Per batch per subject: forward with return_stages, compute L_sl1
    (masked_smooth_l1 as in the step) and L_m1 = 0.1*ce (exact patched lines),
    backward each separately (zero grads between). Per module group
    (feature_extractor, cost_volume, excitation, aggregation, refinement):
    grad L2 norm, max |g|, median |g|, fraction of elements with |g| < 1e-8
    (Adam eps), ratio ||g_m1|| / ||g_sl1||, and the Adam-first-step diagnostic
    median(|g|/(|g|+eps)).
  - Records aggregated_cost std across the disparity axis (median over pixels)
    and the ce value per batch.
  - Writes grad_probe_m1.json (provenance: utc, git head, shas, device, torch
    version) and prints a compact table.

Memory: < 4 GB host RAM (8 small batches materialised once, ~60 MB); CUDA is
freed between subjects.

    python stage_f/followup/kaggle/grad_probe_m1.py
"""
from __future__ import annotations

import ast
import datetime
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle_f1m"
PATCHED_FILE = BUNDLE / "scripts" / "finetune_pilot.py"
REPO_ROOT = HERE.parents[2]  # kaggle -> followup -> stage_f -> repo root

EXPECTED_INIT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
E3_PREFIX = "82e58bc4"

N_BATCHES = 8
ADAM_EPS_PROBE = 1e-8  # cross-checked against the real optimizer object at runtime

GROUP_PREFIXES = ("feature_extractor", "cost_volume", "excitation",
                  "aggregation", "refinement")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_arm_v_config() -> dict:
    """ARM_V_CONFIG from the bundle copy, parsed AST literal (not re-typed)."""
    tree = ast.parse(PATCHED_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ARM_V_CONFIG"
                for t in node.targets):
            if isinstance(node.value, ast.Dict):
                return ast.literal_eval(node.value)
            if (isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict"
                    and not node.value.args):
                return {kw.arg: ast.literal_eval(kw.value)
                        for kw in node.value.keywords}
    raise SystemExit("PROBE FAIL: ARM_V_CONFIG not found in finetune_pilot.py")


def check_m1_lines_present() -> None:
    """The exact patched M1 lines must be in the bundle file (modulo F alias)."""
    text = PATCHED_FILE.read_text(encoding="utf-8")
    for line in (
        '            _m1_cost = _st["aggregated_cost"]',
        "            loss = loss + 0.1 * _m1_ce",
        "            _m1_ce = _m1_ce_map[_m1_valid].mean() if int(_m1_valid.sum()) > 0 else _m1_up.sum() * 0.0",
    ):
        assert line in text, f"PROBE FAIL: M1 line missing from patched file: {line!r}"
    assert "import torch.nn.functional as F" in text


def import_bundle() -> dict:
    """Import bundle code (bundle first on sys.path) and prove provenance."""
    sys.path.insert(0, str(BUNDLE))
    import torch  # noqa: E402

    import src as _src  # noqa: E402
    assert Path(_src.__file__).resolve().is_relative_to(BUNDLE.resolve()), \
        f"imported src is not the bundle copy: {_src.__file__}"
    from phase1.harness import determinism as det  # noqa: E402
    assert Path(det.__file__).resolve().is_relative_to(BUNDLE.resolve()), \
        f"imported determinism is not the bundle copy: {det.__file__}"
    from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402
    from src.losses.disparity import masked_smooth_l1  # noqa: E402
    from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
    import src.datasets.kitti2015 as _kds  # noqa: E402
    assert Path(_kds.__file__).resolve().is_relative_to(BUNDLE.resolve())

    # Reuse the dataset/augmentation class from the bundle's finetune_pilot
    # module itself (no reimplementation). Its `from ... import ...` lines hit
    # the already-imported bundle modules above via sys.modules.
    spec = importlib.util.spec_from_file_location(
        "bundle_finetune_pilot", PATCHED_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert Path(getattr(mod, "__file__", str(PATCHED_FILE))).resolve() == \
        PATCHED_FILE.resolve()
    for name in ("ScaledCroppedKitti", "CROP_H", "CROP_W", "BATCH",
                 "SEED", "SCALE_LO", "SCALE_HI"):
        assert hasattr(mod, name), f"PROBE FAIL: finetune_pilot.{name} missing"
    return {"torch": torch, "det": det, "Kitti2015Stereo": Kitti2015Stereo,
            "masked_smooth_l1": masked_smooth_l1, "StereoNet": StereoNet,
            "StereoNetConfig": StereoNetConfig, "fp": mod}


def group_of(name: str) -> str:
    for g in GROUP_PREFIXES:
        if name.startswith(g + "."):
            return g
    return "other"


def grad_stats(model, eps: float) -> dict:
    """Per-group gradient stats from the current .grad buffers (CPU math)."""
    import torch
    out: dict = {}
    buckets: dict[str, list] = {}
    for n, p in model.named_parameters():
        buckets.setdefault(group_of(n), []).append((n, p))
    for g, items in buckets.items():
        flats = [p.grad.detach().flatten().abs().float().cpu()
                 for _, p in items if p.grad is not None]
        n_params = sum(p.numel() for _, p in items)
        n_with_grad = sum(p.numel() for _, p in items if p.grad is not None)
        if not flats:
            out[g] = {"n_params": n_params, "n_with_grad": 0,
                      "l2": None, "max_abs": None, "median_abs": None,
                      "frac_below_eps": None, "adam_eff_median": None}
            continue
        allg = torch.cat(flats)
        l2 = float(sum(float((p.grad.detach().float() ** 2).sum())
                       for _, p in items if p.grad is not None) ** 0.5)
        out[g] = {
            "n_params": n_params, "n_with_grad": n_with_grad,
            "l2": l2,
            "max_abs": float(allg.max()),
            "median_abs": float(allg.median()),
            "frac_below_eps": float((allg < eps).float().mean()),
            "adam_eff_median": float((allg / (allg + eps)).median()),
        }
    # Document parameter-less groups explicitly (task: "cost_volume/excitation
    # if any params" -- here there are none: CostVolume has no weights when
    # groups=0, and excitation is None since cost_volume_excitation=False).
    for g in ("cost_volume", "excitation"):
        if g not in out:
            mod = getattr(model, g, None)
            out[g] = {"n_params": 0, "n_with_grad": 0,
                      "l2": None, "max_abs": None, "median_abs": None,
                      "frac_below_eps": None, "adam_eff_median": None,
                      "note": "no parameters"
                      + (" (module None: cost_volume_excitation=False)"
                         if mod is None else "")}
    return out


def main() -> None:
    t_start = time.time()
    check_m1_lines_present()
    cfg_dict = read_arm_v_config()
    assert cfg_dict["regression_normalize"] is True, cfg_dict
    B = import_bundle()
    torch, det = B["torch"], B["det"]

    SEED = int(B["fp"].SEED) if isinstance(getattr(B["fp"], "SEED", 0), int) \
        else 0
    assert SEED == 0, SEED
    assert (B["fp"].CROP_H, B["fp"].CROP_W) == (256, 512)
    assert int(B["fp"].BATCH) == 2

    # Same determinism settings as finetune_pilot: seed_all(SEED) only.
    det.seed_all(SEED)
    det_flags = det.flag_snapshot()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = B["StereoNetConfig"](**cfg_dict)
    assert config.num_disparities == 24 and config.feature_stride == 8, config
    assert config.max_disparity_px == 184, config.max_disparity_px

    # Adam eps actually used in finetune_pilot.py: the file constructs
    # torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999)) with no
    # eps argument, so it must be the torch default. Verify on a live object.
    _probe_model = B["StereoNet"](config)
    _opt = torch.optim.Adam(_probe_model.parameters(), lr=1e-3,
                            betas=(0.9, 0.999))
    eps_actual = float(_opt.param_groups[0]["eps"])
    assert "eps" not in [ln for ln in PATCHED_FILE.read_text().splitlines()
                         if "torch.optim.Adam(" in ln][0], \
        "finetune_pilot passes eps explicitly; update the probe"
    assert eps_actual == ADAM_EPS_PROBE == 1e-8, eps_actual
    del _probe_model, _opt
    eps = eps_actual

    # Data: same split / class / augmentation / seed-0 generator as training.
    data_root = REPO_ROOT / "data" / "kitti2015"
    train_base = B["Kitti2015Stereo"](data_root, split="hailo_calib")
    assert len(train_base) == 160, len(train_base)
    assert train_base.names[0] == "000000_10.png", train_base.names[0]
    assert train_base.names[-1] == "000159_10.png", train_base.names[-1]
    train = B["fp"].ScaledCroppedKitti(train_base, seed=SEED)
    from torch.utils.data import DataLoader
    loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=0,
                        generator=det.make_generator(SEED),
                        worker_init_fn=det.worker_init_fn)
    batches = []
    for i, (left, right, disp) in enumerate(loader):
        if i >= N_BATCHES:
            break
        batches.append((left.clone(), right.clone(), disp.clone()))
    assert len(batches) == N_BATCHES, len(batches)
    # Provenance: same augmentation constants as the training file.
    aug_prov = {"class": "ScaledCroppedKitti",
                "module_file": str(PATCHED_FILE),
                "seed": SEED, "crop": [256, 512], "batch": 2,
                "scale": [float(B["fp"].SCALE_LO), float(B["fp"].SCALE_HI)],
                "shuffle": True, "num_workers": 0,
                "generator": f"make_generator({SEED})"}

    subjects = [
        ("A_armp_stage1",
         BUNDLE / "checkpoints" / "armp_stage1_best.pth", EXPECTED_INIT_SHA),
        ("B_e3seed0_final",
         REPO_ROOT / "stage_e_recipe" / "kaggle" / "e3_output" / "seed0"
         / "e3_seed0_final.pth", None),
    ]
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=15).stdout.strip() or None
    except Exception:
        git_head = None

    results: dict = {}
    for subj, ckpt, expect_sha in subjects:
        ckpt = Path(ckpt)
        assert ckpt.is_file(), f"missing checkpoint: {ckpt}"
        sha = sha256(ckpt)
        if expect_sha is not None:
            assert sha == expect_sha, f"{subj} sha mismatch: {sha}"
        else:
            assert sha.startswith(E3_PREFIX), f"{subj} sha prefix: {sha}"

        model = B["StereoNet"](config).to(device)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(sd, strict=True)  # strict, as finetune_pilot
        model.train()
        n_params = sum(p.numel() for p in model.parameters())

        subj_t0 = time.time()
        per_batch = []
        for bi, (left0, right0, disp0) in enumerate(batches):
            left, right, disp = (left0.to(device), right0.to(device),
                                 disp0.to(device))
            F = torch.nn.functional  # noqa: N806 (same alias as patched file)

            # --- L_sl1 branch (own forward; graph freed after backward) ---
            model.zero_grad(set_to_none=True)
            out, _st = model(left, right, return_stages=True)
            cost = _st["aggregated_cost"]
            cost_std_med = float(cost.std(dim=1).median())
            loss_sl1, n = B["masked_smooth_l1"](
                out, disp, max_disparity=float(config.max_disparity_px))
            v_sl1 = float(loss_sl1.detach())
            loss_sl1.backward()
            stats_sl1 = grad_stats(model, eps)

            # --- L_m1 branch: exact patched lines, L_m1 = 0.1*ce ---
            model.zero_grad(set_to_none=True)
            out2, _st2 = model(left, right, return_stages=True)
            _m1_cost = _st2["aggregated_cost"]
            _m1_up = F.interpolate(_m1_cost, size=disp.shape[-2:],
                                   mode="bilinear", align_corners=True)
            _m1_norm = ((_m1_up - _m1_up.mean(1, keepdim=True))
                        / (_m1_up.std(1, keepdim=True) + 1e-6))
            _m1_logp = F.log_softmax(-_m1_norm, dim=1)
            _m1_valid = ((disp > 0) & (disp < float(config.max_disparity_px)))
            _m1_t = disp / float(config.feature_stride)
            _m1_k = torch.arange(
                config.num_disparities, dtype=_m1_up.dtype,
                device=_m1_up.device).view(1, -1, 1, 1)
            _m1_q = torch.softmax(-torch.abs(_m1_k - _m1_t) / 0.5, dim=1)
            _m1_ce_map = -(_m1_q * _m1_logp).sum(1, keepdim=True)
            _m1_ce = (_m1_ce_map[_m1_valid].mean()
                      if int(_m1_valid.sum()) > 0 else _m1_up.sum() * 0.0)
            v_ce = float(_m1_ce.detach())
            loss_m1 = 0.1 * _m1_ce
            v_m1 = float(loss_m1.detach())
            loss_m1.backward()
            stats_m1 = grad_stats(model, eps)
            model.zero_grad(set_to_none=True)

            per_batch.append({
                "batch": bi, "valid_pixels": int(n),
                "sl1": v_sl1, "ce": v_ce, "L_m1": v_m1,
                "cost_std_med": cost_std_med,
                "sl1_groups": stats_sl1, "m1_groups": stats_m1,
            })
            print(f"[{subj} b{bi}] sl1={v_sl1:.4f} ce={v_ce:.4f} "
                  f"L_m1={v_m1:.4f} cost_std_med={cost_std_med:.3e} "
                  f"n={n}", flush=True)

        # Aggregate across batches (mean of per-batch stats).
        groups: dict = {}
        for g in list(per_batch[0]["sl1_groups"].keys()):
            agg = {"n_params": per_batch[0]["sl1_groups"][g]["n_params"]}
            for tag, key in (("sl1", "sl1_groups"), ("m1", "m1_groups")):
                vals = [b[key][g] for b in per_batch]
                if vals[0]["l2"] is None:
                    agg[tag] = {"l2_mean": None, "max_mean": None,
                                "median_mean": None, "frac_below_eps_mean": None,
                                "adam_eff_mean": None}
                else:
                    agg[tag] = {
                        "l2_mean": sum(v["l2"] for v in vals) / len(vals),
                        "max_mean": sum(v["max_abs"] for v in vals) / len(vals),
                        "median_mean": sum(v["median_abs"] for v in vals)
                        / len(vals),
                        "frac_below_eps_mean": sum(v["frac_below_eps"]
                                                   for v in vals) / len(vals),
                        "adam_eff_mean": sum(v["adam_eff_median"]
                                             for v in vals) / len(vals),
                    }
            s, m = agg["sl1"]["l2_mean"], agg["m1"]["l2_mean"]
            agg["ratio_m1_over_sl1"] = (m / s if s not in (None, 0)
                                        and m is not None else None)
            if m is None and s is not None:
                agg["note_m1_none"] = ("M1 loss is upstream of this group "
                                       "(not an ancestor of aggregated_cost), "
                                       "so no gradient path exists")
            if s is None and m is None and agg["n_params"] == 0:
                agg["note_m1_none"] = agg.get("note_m1_none", "") or \
                    "group has no parameters"
            groups[g] = agg

        results[subj] = {
            "ckpt": str(ckpt), "sha256": sha, "n_params": n_params,
            "seconds": time.time() - subj_t0,
            "ce_mean": sum(b["ce"] for b in per_batch) / len(per_batch),
            "sl1_mean": sum(b["sl1"] for b in per_batch) / len(per_batch),
            "cost_std_med_mean": sum(b["cost_std_med"] for b in per_batch)
            / len(per_batch),
            "groups": groups, "per_batch": per_batch,
        }
        # Free CUDA between subjects.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = {
        "provenance": {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_head": git_head,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "adam_eps_actual": eps,
            "adam_spec": "torch.optim.Adam(lr=1e-3, betas=(0.9, 0.999)), "
                         "no eps arg in finetune_pilot.py -> default 1e-8",
            "config": cfg_dict,
            "data": aug_prov,
            "n_batches": N_BATCHES,
            "determinism": det_flags,
            "subjects": {k: {"ckpt": v["ckpt"], "sha256": v["sha256"]}
                         for k, v in results.items()},
        },
        "subjects": results,
        "seconds_total": time.time() - t_start,
    }
    out_path = HERE / "grad_probe_m1.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    for subj, r in results.items():
        print(f"\n=== {subj}  ce_mean={r['ce_mean']:.4f} "
              f"sl1_mean={r['sl1_mean']:.4f} "
              f"cost_std_med={r['cost_std_med_mean']:.3e} "
              f"({r['seconds']:.0f}s) ===")
        print(f"{'group':16s} {'||g_sl1||':>10s} {'||g_m1||':>10s} "
              f"{'ratio':>9s} {'frac<eps_sl1':>12s} {'frac<eps_m1':>11s} "
              f"{'adamSl1':>8s} {'adamM1':>8s}")
        for g, a in r["groups"].items():
            s, m = a["sl1"], a["m1"]
            fmt = lambda x: "n/a" if x is None else (  # noqa: E731
                f"{x:.3e}" if isinstance(x, float) else str(x))
            print(f"{g:16s} {fmt(s['l2_mean']):>10s} {fmt(m['l2_mean']):>10s} "
                  f"{fmt(a['ratio_m1_over_sl1']):>9s} "
                  f"{fmt(s['frac_below_eps_mean']):>12s} "
                  f"{fmt(m['frac_below_eps_mean']):>11s} "
                  f"{fmt(s['adam_eff_mean']):>8s} {fmt(m['adam_eff_mean']):>8s}")
    print(f"\nwrote {out_path} ({r['seconds']:.0f}s subjects, "
          f"{record['seconds_total']:.0f}s total)")


if __name__ == "__main__":
    main()
