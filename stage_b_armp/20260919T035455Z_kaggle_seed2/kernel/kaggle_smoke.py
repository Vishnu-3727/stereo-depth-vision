"""Kaggle GPU smoke kernel — ARM-P Tier-2 seed-2 portability gate, PHASES 3-7.

Runs, in order, P3 (dependency audit), P4 (data access), P5 (checkpoint),
P6 (one-batch smoke), P7 (emit kaggle_environment.json). Exactly one batch;
NO epochs; NO training. Writes go only to /kaggle/working. The research
checkpoint is opened read-only and never modified.

Usage on Kaggle (GPU kernel, both datasets attached):
    python /kaggle/input/<code-bundle>/kaggle_smoke.py

Output: /kaggle/working/kaggle_environment.json (the kernel artefact).
Any FAIL writes the partial artefact first, then raises — the kernel must
go red, never silently green.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as pkgmeta
import importlib.util as importutil
import json
import os
import platform
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
ENV_JSON = WORKING / "kaggle_environment.json"
SMOKE_CKPT = WORKING / "smoke_ckpt.pth"

EXPECTED_PARAMS = 397954
EXPECTED_KEYS = 70
EXPECTED_CKPT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
CFG = dict(downsample_levels=3, num_disparities=24,
           cost_volume_shift="right", regression_normalize=True)
EXPECTED_POPULATION = 3802797

CLOSURE_MODULES = [
    "src.datasets.kitti2015",
    "src.evaluation.metrics",
    "src.losses.disparity",
    "src.models.stereonet",
    "src.models.stereonet.aggregation",
    "src.models.stereonet.blocks",
    "src.models.stereonet.cost_volume",
    "src.models.stereonet.excitation",
    "src.models.stereonet.feature_extractor",
    "src.models.stereonet.refinement",
    "src.models.stereonet.regression",
    "phase1.harness.determinism",
    "phase1.harness.frozen_eval",
]

result: dict = {"phases": {}, "status": "RUNNING"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def emit(status: str) -> None:
    result["status"] = status
    WORKING.mkdir(parents=True, exist_ok=True)
    ENV_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", ENV_JSON, "status=", status, flush=True)


def pkg_version(dist: str):
    try:
        return pkgmeta.version(dist)
    except Exception:
        return None


def phase3() -> dict:
    """P3: dependency audit + import-only test of the full closure + CUDA probe."""
    import torch  # noqa: E402
    try:
        import torchvision  # noqa: E402
        tv = torchvision.__version__
    except Exception as exc:
        tv = "NOT IMPORTABLE (%s: %s)" % (type(exc).__name__, exc)
    import numpy  # noqa: E402
    import cv2  # noqa: E402
    import PIL  # noqa: E402
    from PIL import Image  # noqa: E402

    imported = {}
    for mod in CLOSURE_MODULES:
        try:
            importlib.import_module(mod)
            imported[mod] = "OK"
        except Exception as exc:
            imported[mod] = "FAIL %s: %s" % (type(exc).__name__, exc)
    cudnn_v = None
    try:
        cudnn_v = torch.backends.cudnn.version()
    except Exception:
        cudnn_v = None
    gpu_name, capability = None, None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu_name = torch.cuda.get_device_name(0)
        try:
            capability = list(torch.cuda.get_device_capability(0))
        except Exception:
            capability = None
    rep = {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": tv,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_v,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else PIL.__version__,
        "pip_dist_versions": {
            "torch": pkg_version("torch"),
            "torchvision": pkg_version("torchvision"),
            "numpy": pkg_version("numpy"),
            "opencv-python": pkg_version("opencv-python"),
            "opencv-python-headless": pkg_version("opencv-python-headless"),
            "Pillow": pkg_version("Pillow"),
        },
        "closure_import_test": imported,
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_name": gpu_name,
        "cuda_capability": capability,
    }
    bad = [m for m, s in imported.items() if s != "OK"]
    if bad:
        raise RuntimeError("P3 FAIL: closure import failures: " + json.dumps(bad))
    if not torch.cuda.is_available():
        raise RuntimeError("P3 FAIL: torch.cuda.is_available() is False on a GPU kernel")
    return rep


def phase4(repo: Path) -> dict:
    """P4: data access — hailo_val population must be EXACTLY 3802797, else FAIL."""
    from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402
    data_root = repo / "data" / "kitti2015"
    val = Kitti2015Stereo(data_root, split="hailo_val")
    cal = Kitti2015Stereo(data_root, split="hailo_calib")
    assert len(val) == 40, "hailo_val scenes: %d" % len(val)
    assert len(cal) == 160, "hailo_calib scenes: %d" % len(cal)
    assert val.names[0] == "000160_10.png" and val.names[-1] == "000199_10.png", val.names[:1] + val.names[-1:]
    assert cal.names[0] == "000000_10.png" and cal.names[-1] == "000159_10.png", cal.names[:1] + cal.names[-1:]
    assert not (set(val.names) & set(cal.names)), "train/eval overlap!"
    assert val.disparity_scale == 256.0, val.disparity_scale
    assert val.gt_dir.name == "disp_occ_0", val.gt_dir.name
    total, h0, w0, dshape = 0, None, None, None
    for i in range(len(val)):
        s = val[i]
        if h0 is None:
            h0, w0, dshape = s.left.shape[0], s.left.shape[1], s.disparity.shape
        assert s.left.shape == (368, 1232, 3), s.left.shape
        assert s.right.shape == (368, 1232, 3), s.right.shape
        assert s.disparity.shape == (368, 1232), s.disparity.shape
        total += int((s.disparity > 0).sum())  # valid-pixel semantics: disparity > 0
    rep = {
        "hailo_val_scenes": len(val),
        "hailo_val_first": val.names[0], "hailo_val_last": val.names[-1],
        "hailo_calib_scenes": len(cal),
        "hailo_calib_first": cal.names[0], "hailo_calib_last": cal.names[-1],
        "overlap": 0,
        "image_dims": [h0, w0, 3], "disparity_dims": list(dshape),
        "disparity_scale": val.disparity_scale,
        "gt_source": val.gt_dir.name,
        "valid_semantics": "disparity > 0",
        "valid_pixels_counted": total,
        "valid_pixels_expected": EXPECTED_POPULATION,
    }
    if total != EXPECTED_POPULATION:
        raise RuntimeError("P4 FAIL: evaluation population %d != %d"
                           % (total, EXPECTED_POPULATION))
    return rep


def phase5(repo: Path) -> dict:
    """P5: strict checkpoint load + architecture/ Provenance guards."""
    import torch  # noqa: E402
    from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
    ckpt = repo / "checkpoints" / "armp_stage1_best.pth"
    on_kaggle_sha = sha256_file(ckpt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    config = StereoNetConfig(**CFG)
    model = StereoNet(config)
    want, got = set(model.state_dict().keys()), set(sd.keys())
    missing, unexpected = sorted(want - got), sorted(got - want)
    model.load_state_dict(sd, strict=True)  # raises on any mismatch
    params = sum(p.numel() for p in model.parameters())
    rep = {
        "checkpoint": str(ckpt),
        "sha256_on_kaggle": on_kaggle_sha,
        "sha256_expected": EXPECTED_CKPT_SHA,
        "sha256_match": on_kaggle_sha == EXPECTED_CKPT_SHA,
        "strict_missing": missing, "strict_unexpected": unexpected,
        "params": params, "params_expected": EXPECTED_PARAMS,
        "n_keys": len(sd), "n_keys_expected": EXPECTED_KEYS,
        "downsample_levels": config.downsample_levels,
        "num_disparities": config.num_disparities,
        "cost_volume_shift": config.cost_volume_shift,
        "regression_normalize": config.regression_normalize,
    }
    failures = []
    if not rep["sha256_match"]:
        failures.append("sha256 mismatch")
    if missing or unexpected:
        failures.append("strict-load key mismatch")
    if params != EXPECTED_PARAMS:
        failures.append("param count %d" % params)
    if len(sd) != EXPECTED_KEYS:
        failures.append("key count %d" % len(sd))
    if failures:
        raise RuntimeError("P5 FAIL: " + "; ".join(failures))
    return rep


def phase6(repo: Path) -> dict:
    """P6: exactly one batch — read, H2D, forward, loss, backward, step, save, reload."""
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from phase1.harness.frozen_eval import strict_load_report  # noqa: E402
    from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
    from src.losses.disparity import masked_smooth_l1  # noqa: E402
    from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
    if not torch.cuda.is_available():
        raise RuntimeError("P6 FAIL: no CUDA device")
    device = torch.device("cuda")
    t0 = time.time()
    ds = Kitti2015Stereo(repo / "data" / "kitti2015", split="hailo_calib")
    s = ds[0]
    left = torch.from_numpy(normalize(s.left)).to(device)
    right = torch.from_numpy(normalize(s.right)).to(device)
    # normalize() already returns (1,3,H,W); disparity needs both channel and
    # batch axes to match the DataLoader-batched frozen recipe (B,1,H,W).
    disp = torch.from_numpy(np.ascontiguousarray(s.disparity[None, None], dtype=np.float32)).to(device)
    config = StereoNetConfig(**CFG)
    model = StereoNet(config).to(device)
    model.train()
    blob = torch.load(repo / "checkpoints" / "armp_stage1_best.pth",
                      map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd, strict=True)
    out = model(left, right)
    loss, n = masked_smooth_l1(out, disp, max_disparity=float(config.max_disparity_px))
    loss.backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    optimizer.step()
    torch.save({"model": model.state_dict()}, SMOKE_CKPT)
    reloaded = torch.load(SMOKE_CKPT, map_location="cpu", weights_only=False)
    fresh = StereoNet(StereoNetConfig(**CFG))  # frozen-eval model construction
    compat = strict_load_report(fresh, reloaded["model"])
    wall_s = time.time() - t0
    rep = {
        "sample": s.name,
        "left_shape": list(left.shape), "right_shape": list(right.shape),
        "disp_shape": list(disp.shape), "pred_shape": list(out.shape),
        "device": str(device),
        "loss": float(loss), "valid_pixels_in_batch": int(n),
        "backward_grads_present": bool(grads_ok),
        "optimizer": "Adam lr=1e-3 betas=(0.9, 0.999)",
        "optimizer_steps": 1, "epochs": 0,
        "checkpoint_written": str(SMOKE_CKPT),
        "reload_strict_missing": compat["missing"],
        "reload_strict_unexpected": compat["unexpected"],
        "params": int(compat["params"]),
        "wall_s": wall_s,
    }
    if not grads_ok or compat["missing"] or compat["unexpected"]:
        raise RuntimeError("P6 FAIL: " + json.dumps({k: rep[k] for k in
                              ("backward_grads_present", "reload_strict_missing",
                               "reload_strict_unexpected")}))
    return rep


def main() -> None:
    t_all = time.time()
    result.update({
        "experiment": "ARM-P Tier-2 seed-2 Kaggle portability smoke (phases 3-7; NO training)",
        "seed": 2,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_identifiers": {},
        "source_hashes": {},
        "tensor_shapes": {},
    })
    # Kaggle script kernels execute ONLY code_file as /kaggle/src/script.py;
    # sibling files pushed alongside it are NOT present at runtime. The
    # stdlib-only wrapper therefore loads from the attached bundle dataset
    # mount (recursive search under /kaggle/input, marker mount preferred),
    # falling back to next-to-this-file for local runs. The /kaggle/input
    # listing is recorded for mount diagnostics.
    _inp = Path("/kaggle/input")
    _listing, _found = [], []
    if _inp.is_dir():
        for _dp, _dn, _fn in os.walk(_inp):
            _rel = os.path.relpath(_dp, _inp)
            _depth = 0 if _rel == "." else _rel.count(os.sep) + 1
            if _depth <= 2:
                _listing.append(_rel + "/: " + ",".join(sorted(_dn)[:10])
                                + " | " + ",".join(sorted(_fn)[:10]))
            if "kaggle_bootstrap.py" in _fn:
                _found.append(Path(_dp) / "kaggle_bootstrap.py")
    _found.sort(key=lambda p: (not (p.parent / "BUNDLE_MARKER.json").is_file(),
                               str(p)))
    _found.append(Path(__file__).resolve().parent / "kaggle_bootstrap.py")
    result["kaggle_input_listing"] = _listing
    result["bootstrap_candidates"] = [str(p) for p in _found]
    kaggle_bootstrap = None
    for _p in _found:
        if _p.is_file():
            _spec = importutil.spec_from_file_location("kaggle_bootstrap", _p)
            kaggle_bootstrap = importutil.module_from_spec(_spec)
            _spec.loader.exec_module(kaggle_bootstrap)
            result["bootstrap_loaded_from"] = str(_p)
            break
    if kaggle_bootstrap is None:
        raise ModuleNotFoundError(
            "kaggle_bootstrap.py not found (searched /kaggle/input "
            "recursively + script dir); /kaggle/input listing: "
            + json.dumps(_listing))
    asm = kaggle_bootstrap.assemble()
    repo = Path(asm["repo"])
    result["dataset_identifiers"] = {"bundle_mount": asm["bundle"], "data_mount": asm["data"]}
    result["repo"] = asm["repo"]
    result["scripts_dir"] = asm["scripts"]
    # Nesting-depth contract (finetune_pilot.py:39 parents[3], run_arm.py
    # PILOT/REPO): scripts MUST live at <root>/stage_b_armp/seed2/scripts/
    # and phase1/runs/arm_v/arm_v_best.pth MUST be bundled (integrity_guard).
    _sdir = Path(asm["scripts"])
    _fp = _sdir / "finetune_pilot.py"
    _depth_ok = _fp.resolve().parents[3] == repo.resolve()
    _armv = repo / "phase1" / "runs" / "arm_v" / "arm_v_best.pth"
    result["nesting"] = {
        "scripts_dir": str(_sdir),
        "scripts_present": sorted(p.name for p in _sdir.glob("*.py")) if _sdir.is_dir() else [],
        "parents3_is_repo": bool(_depth_ok),
        "arm_v_best_present": bool(_armv.is_file()),
        "arm_v_best_bytes": int(_armv.stat().st_size) if _armv.is_file() else 0,
    }
    if not _depth_ok:
        raise RuntimeError("nesting depth wrong: parents[3]=%s != repo=%s"
                           % (_fp.resolve().parents[3], repo.resolve()))
    if not _armv.is_file():
        raise RuntimeError("missing bundled architecture reference: " + str(_armv))
    print("nesting OK", flush=True)
    result["bundle_marker"] = kaggle_bootstrap.marker(asm["bundle"])
    sys.path.insert(0, str(repo))

    result["phases"]["P3_dependency_audit"] = phase3()
    print("P3 OK", flush=True)
    result["phases"]["P4_data_access"] = phase4(repo)
    print("P4 OK", flush=True)
    result["phases"]["P5_checkpoint"] = phase5(repo)
    print("P5 OK", flush=True)
    result["phases"]["P6_one_batch"] = phase6(repo)
    print("P6 OK", flush=True)

    p6 = result["phases"]["P6_one_batch"]
    result["tensor_shapes"] = {"left": p6["left_shape"], "pred": p6["pred_shape"]}
    result["source_hashes"] = {
        f: sha256_file(repo / f) for f in
        ["stage_b_armp/seed2/scripts/finetune_pilot.py", "stage_b_armp/seed2/scripts/run_pilot.py",
         "stage_b_armp/seed2/scripts/run_arm.py",
         "stage_b_armp/seed2/scripts/eval_tier2.py", "src/datasets/kitti2015.py",
         "src/evaluation/metrics.py", "src/losses/disparity.py",
         "phase1/harness/determinism.py", "phase1/harness/frozen_eval.py"]
    }
    for mod in ["src.models.stereonet." + m for m in
                ("__init__", "aggregation", "blocks", "cost_volume", "excitation",
                 "feature_extractor", "refinement", "regression", "stereonet")]:
        result["source_hashes"][mod.replace(".", "/") + ".py"] = \
            sha256_file(repo / (mod.replace(".", "/") + ".py"))
    result["checkpoint_sha256_on_kaggle"] = result["phases"]["P5_checkpoint"]["sha256_on_kaggle"]
    result["smoke_test_result"] = "PASS"
    result["total_wall_s"] = time.time() - t_all
    emit("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        result["smoke_test_result"] = "FAIL"
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        try:
            emit("FAIL")
        except Exception:
            pass
        raise
