"""Kaggle training kernel — ARM-P Tier-2 seed-2, CONTROL arm.

arm=control, init=random (PyTorch defaults, original P2A recipe).
Seed 2, 200 epochs. ONLY INITIALIZATION DIFFERS from the armp kernel.

Invokes scripts/run_arm.py UNMODIFIED as a fresh subprocess:
    run_arm.py --arm control --init random --outdir <dir> --timeout_s 39600
--timeout_s 39600 (11 h) is REQUIRED: run_arm.py STOPs on PROJECTED wall
over the 7200 s default, and a T4 is slower than the local card.

Outputs live under /kaggle/working so checkpoints persist. Subprocess
output streams to both the kernel log (flushed) and outdir/stdout.log.
"""
from __future__ import annotations

import hashlib
import importlib.util as importutil
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ARM = "control"
INIT = "random"  # CONTROL is RANDOM INITIALIZATION under the original P2A recipe
TIMEOUT_S = 39600.0
SUBPROCESS_TIMEOUT_S = 41400.0  # above run_arm's own budget so its STOP fires first

WORKING = Path("/kaggle/working")
OUTDIR = WORKING / "runs" / (ARM + "_seed2")
PROVENANCE = WORKING / ("provenance_" + ARM + "_seed2.json")

EXPECTED_PARAMS = 397954
EXPECTED_KEYS = 70
EXPECTED_CKPT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
CFG = dict(downsample_levels=3, num_disparities=24,
           cost_volume_shift="right", regression_normalize=True)
EXPECTED_POPULATION = 3802797


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_bootstrap():
    _inp = Path("/kaggle/input")
    _found = []
    if _inp.is_dir():
        for _dp, _dn, _fn in os.walk(_inp):
            if "kaggle_bootstrap.py" in _fn:
                _found.append(Path(_dp) / "kaggle_bootstrap.py")
    _found.sort(key=lambda p: (not (p.parent / "BUNDLE_MARKER.json").is_file(),
                               str(p)))
    _found.append(Path(__file__).resolve().parent / "kaggle_bootstrap.py")
    for _p in _found:
        if _p.is_file():
            _spec = importutil.spec_from_file_location("kaggle_bootstrap", _p)
            mod = importutil.module_from_spec(_spec)
            _spec.loader.exec_module(mod)
            print("bootstrap loaded from %s" % _p, flush=True)
            return mod
    raise ModuleNotFoundError("kaggle_bootstrap.py not found under /kaggle/input")


def preflight(repo: Path) -> dict:
    """Cheap guards before the 200-epoch run: strict load + data contract."""
    import torch
    from src.datasets.kitti2015 import Kitti2015Stereo
    from src.models.stereonet import StereoNet, StereoNetConfig
    ckpt = repo / "checkpoints" / "armp_stage1_best.pth"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    config = StereoNetConfig(**CFG)
    model = StereoNet(config)
    want, got = set(model.state_dict().keys()), set(sd.keys())
    missing, unexpected = sorted(want - got), sorted(got - want)
    model.load_state_dict(sd, strict=True)
    params = sum(p.numel() for p in model.parameters())
    assert sha256_file(ckpt) == EXPECTED_CKPT_SHA, "ckpt sha"
    assert not missing and not unexpected, (missing, unexpected)
    assert params == EXPECTED_PARAMS, params
    assert len(sd) == EXPECTED_KEYS, len(sd)
    assert all(getattr(config, k) == v for k, v in CFG.items()), "cfg"
    data_root = repo / "data" / "kitti2015"
    val = Kitti2015Stereo(data_root, split="hailo_val")
    cal = Kitti2015Stereo(data_root, split="hailo_calib")
    assert len(val) == 40 and len(cal) == 160, (len(val), len(cal))
    assert not (set(val.names) & set(cal.names)), "overlap!"
    total = sum(int((val[i].disparity > 0).sum()) for i in range(len(val)))
    assert total == EXPECTED_POPULATION, total
    armv = repo / "phase1" / "runs" / "arm_v" / "arm_v_best.pth"
    assert armv.is_file(), "arm_v_best missing"
    return {"strict_load": "OK", "params": params, "keys": len(sd),
            "data": {"calib": 160, "val": 40, "overlap": 0,
                     "valid_pixels": total}}


def main() -> int:
    utc_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_start = time.time()
    kb = load_bootstrap()
    asm = kb.assemble()
    repo = Path(asm["repo"])
    sys.path.insert(0, str(repo))
    print("assembled %s" % json.dumps(asm), flush=True)

    guard = preflight(repo)
    print("preflight OK %s" % json.dumps(guard), flush=True)

    import torch
    prov: dict = {
        "arm": ARM, "seed": 2, "init": INIT,
        "init_sha256": "random",
        "utc_start": utc_start,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "epochs": 200, "timeout_s": TIMEOUT_S,
        "bundle_marker": kb.marker(asm["bundle"]),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    script = kb.script_dir(repo) / "run_arm.py"
    cmd = [sys.executable, str(script), "--arm", ARM, "--init", INIT,
           "--outdir", str(OUTDIR), "--timeout_s", str(int(TIMEOUT_S))]
    print("RUN %s" % json.dumps(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(repo), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    with open(OUTDIR / "kernel_stdout.log", "w", encoding="utf-8") as fh:
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
    rc = proc.wait(timeout=SUBPROCESS_TIMEOUT_S - (time.time() - t_start))
    wall = time.time() - t_start
    prov["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prov["wall_s"] = wall
    prov["run_arm_returncode"] = rc
    if rc == 0:
        rec = json.loads((OUTDIR / "record.json").read_text(encoding="utf-8"))
        ig = json.loads((OUTDIR / "integrity_guard.json").read_text(encoding="utf-8"))
        prov.update({
            "best_epoch": rec.get("train_monitor_best_epoch"),
            "best_10scene_epe": rec.get("train_monitor_best_epe_10scene"),
            "best_sha256": sha256_file(OUTDIR / "p2a_best.pth"),
            "final_sha256": sha256_file(OUTDIR / "p2a_final.pth"),
            "guard_all_ok": bool(ig.get("all_ok")),
            "guard_param_count": ig.get("param_count"),
        })
    PROVENANCE.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print("provenance %s rc=%d wall=%.1f" % (PROVENANCE, rc, wall), flush=True)
    if rc != 0:
        raise SystemExit("TRAINING kernel FAIL: run_arm.py rc=%d (see STOP.json)" % rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
