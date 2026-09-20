"""Kaggle entrypoint gate (FIX 3) — ARM-P Tier-2 seed-2.

Runs the ACTUAL training script `finetune_pilot.py` end to end for 2
epochs (P2A_EPOCHS=2, legitimate: finetune_pilot.py:63 reads P2A_EPOCHS),
for --init random AND --init <bundled armp checkpoint>.

Deliberately NOT routed through run_arm.py: run_arm.py forces
P2A_EPOCHS=200 and its post-checks demand 200 rows, so a short run would
trip a false STOP.

Proves: REPO_ROOT resolves (nested stage_b_armp/seed2 layout), all
imports succeed, integrity_guard PASSES against the bundled
phase1/runs/arm_v/arm_v_best.pth, the dataset asserts pass (160/40, no
overlap), the training loop runs, and checkpoints are written.

Output: /kaggle/working/entrycheck_result.json (PASS/FAIL). Any FAIL
raises — the kernel must go red, never silently green.
"""
from __future__ import annotations

import hashlib
import importlib.util as importutil
import json
import os
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
RESULT = WORKING / "entrycheck_result.json"

EXPECTED_PARAMS = 397954
EXPECTED_KEYS = 70
EXPECTED_CKPT_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
EXPECTED_ARMV_SHA = "f2dcf8a22e6f1711b67a70837066f84e68550d53bb4891d3b8d46bd50b452cc4"
CFG = dict(downsample_levels=3, num_disparities=24,
           cost_volume_shift="right", regression_normalize=True)
EXPECTED_POPULATION = 3802797
PER_RUN_TIMEOUT_S = 5400.0

result: dict = {"status": "RUNNING", "runs": {}}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def emit(status: str) -> None:
    result["status"] = status
    WORKING.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", RESULT, "status=", status, flush=True)


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
    result["bootstrap_candidates"] = [str(p) for p in _found]
    for _p in _found:
        if _p.is_file():
            _spec = importutil.spec_from_file_location("kaggle_bootstrap", _p)
            mod = importutil.module_from_spec(_spec)
            _spec.loader.exec_module(mod)
            result["bootstrap_loaded_from"] = str(_p)
            return mod
    raise ModuleNotFoundError("kaggle_bootstrap.py not found under /kaggle/input")


def preflight(repo: Path) -> dict:
    """Cheap guards: strict load (397954 params, 70 keys, cfg) + data contract."""
    import torch
    from src.datasets.kitti2015 import Kitti2015Stereo
    from src.models.stereonet import StereoNet, StereoNetConfig
    ckpt = repo / "checkpoints" / "armp_stage1_best.pth"
    on_sha = sha256_file(ckpt)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    config = StereoNetConfig(**CFG)
    model = StereoNet(config)
    want, got = set(model.state_dict().keys()), set(sd.keys())
    missing, unexpected = sorted(want - got), sorted(got - want)
    model.load_state_dict(sd, strict=True)
    params = sum(p.numel() for p in model.parameters())
    armv = repo / "phase1" / "runs" / "arm_v" / "arm_v_best.pth"
    rep = {
        "ckpt_sha_match": on_sha == EXPECTED_CKPT_SHA,
        "strict_missing": missing, "strict_unexpected": unexpected,
        "params": params, "n_keys": len(sd),
        "arm_v_best_present": armv.is_file(),
        "arm_v_best_sha_match": sha256_file(armv) == EXPECTED_ARMV_SHA if armv.is_file() else False,
    }
    data_root = repo / "data" / "kitti2015"
    val = Kitti2015Stereo(data_root, split="hailo_val")
    cal = Kitti2015Stereo(data_root, split="hailo_calib")
    assert len(val) == 40 and len(cal) == 160, (len(val), len(cal))
    assert not (set(val.names) & set(cal.names)), "train/eval overlap!"
    total = 0
    for i in range(len(val)):
        s = val[i]
        total += int((s.disparity > 0).sum())
    rep["data"] = {"calib": len(cal), "val": len(val), "overlap": 0,
                   "valid_pixels": total, "expected": EXPECTED_POPULATION}
    fails = []
    if not rep["ckpt_sha_match"]:
        fails.append("ckpt sha")
    if missing or unexpected:
        fails.append("strict-load keys")
    if params != EXPECTED_PARAMS:
        fails.append("params %d" % params)
    if len(sd) != EXPECTED_KEYS:
        fails.append("keys %d" % len(sd))
    if not rep["arm_v_best_sha_match"]:
        fails.append("arm_v_best sha")
    if total != EXPECTED_POPULATION:
        fails.append("population %d" % total)
    if fails:
        raise RuntimeError("preflight FAIL: " + "; ".join(fails))
    return rep


def run_entrypoint(kb, repo: Path, init: str, tag: str) -> dict:
    """Run the ACTUAL finetune_pilot.py for 2 epochs. Returns the run record."""
    outdir = WORKING / "entrycheck" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cp = kb.run_script(
        repo, "finetune_pilot.py",
        ["--init", init, "--arm", "entrycheck"],
        env_extra={"P2A_SEED": "2", "P2A_OUT_DIR": str(outdir),
                   "P2A_EPOCHS": "2"},
        timeout_s=PER_RUN_TIMEOUT_S)
    (outdir / "stdout.log").write_text(cp.stdout, encoding="utf-8")
    print(cp.stdout, flush=True)
    wall = time.time() - t0
    rec = {"init": init, "returncode": cp.returncode, "wall_s": wall,
           "outdir": str(outdir)}
    if cp.returncode != 0:
        rec["error"] = "nonzero exit; tail: " + cp.stdout[-2000:]
        return rec
    guard = json.loads((outdir / "integrity_guard.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in
            (outdir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    prec = json.loads((outdir / "p2a_record.json").read_text(encoding="utf-8"))
    rec.update({
        "guard_all_ok": bool(guard.get("all_ok")),
        "guard_param_count": guard.get("param_count"),
        "epochs_logged": len(rows),
        "epochs_run": prec.get("epochs_run"),
        "best_present": (outdir / "p2a_best.pth").is_file(),
        "final_present": (outdir / "p2a_final.pth").is_file(),
        "best_sha256": prec.get("best_sha256"),
        "final_sha256": prec.get("final_sha256"),
    })
    ok = (rec["guard_all_ok"] and rec["epochs_logged"] == 2
          and rec["epochs_run"] == 2
          and rec["best_present"] and rec["final_present"])
    rec["ok"] = bool(ok)
    return rec


def main() -> None:
    t_all = time.time()
    result.update({"experiment": "ARM-P Tier-2 seed-2 entrypoint gate (FIX 3)",
                   "seed": 2,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    kb = load_bootstrap()
    asm = kb.assemble()
    repo = Path(asm["repo"])
    result["repo"] = asm["repo"]
    result["scripts"] = asm["scripts"]
    result["bundle_marker"] = kb.marker(asm["bundle"])
    sys.path.insert(0, str(repo))

    result["preflight"] = preflight(repo)
    print("preflight OK", flush=True)

    inits = [("random", "init_random"),
             (str(repo / "checkpoints" / "armp_stage1_best.pth"), "init_armp")]
    for init, tag in inits:
        rec = run_entrypoint(kb, repo, init, tag)
        result["runs"][tag] = rec
        print(tag, "OK" if rec.get("ok") else "FAIL", flush=True)
    bad = [t for t, r in result["runs"].items() if not r.get("ok")]
    if bad:
        raise RuntimeError("entrypoint FAIL: " + json.dumps(bad))
    result["total_wall_s"] = time.time() - t_all
    emit("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        try:
            emit("FAIL")
        except Exception:
            pass
        raise
