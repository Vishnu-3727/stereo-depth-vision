#!/usr/bin/env python
"""Stage E — E4 finetune run, ONE seed, on Kaggle T4.

Trains the BYTE-IDENTICAL E0 recipe (batch 2, 200 epochs, T_max 200, KITTI
hailo_calib only) from the E4 pretrain's best-by-pretrain-val init instead of
the Stage-1 init. The init arrives as its own dataset
(vishnuvardhanksece/stage-e-e4-init); this kernel resolves it by file name
under /kaggle/input, verifies its sha256 against the value templated in by
push_e4.py at push time, and passes the mount path as --init.

SEED and EXPECTED_INIT_SHA are templated in by push_e4.py before the kernel
is pushed, because Kaggle script kernels take no arguments.

Scoring reuses eval_tier2.py's `score()` unmodified (which itself imports the
frozen evaluator's pooled_metrics and contract guard). eval_tier2's own main()
is not called: it targets a control arm that Stage E does not have.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEED = __SEED__          # templated by push_e4.py
EXP = "e4"
EPOCHS = 200             # E4 finetune is byte-identical to E0: 200, not 400
TIMEOUT_S = 18000.0      # 5 h; measured E0 rate is ~1.03 h/seed
EXPECTED_INIT_SHA = "__INIT_SHA__"  # templated by push_e4.py at push time
INIT_FILE_NAME = "e4_pretrain_best.pth"
EXPECTED_PARAMS = 397954
OUT = Path(f"/kaggle/working/{EXP}_seed{SEED}.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_init() -> Path:
    """Resolve the E4 init checkpoint from its dataset mount.

    The init dataset contains exactly one file named e4_pretrain_best.pth.
    Zero or two-plus hits fail loudly rather than picking one silently.
    """
    hits: list[Path] = []
    base = Path("/kaggle/input")
    if base.is_dir():
        for dp, _dn, fn in os.walk(base):
            if INIT_FILE_NAME in fn:
                candidate = Path(dp) / INIT_FILE_NAME
                if candidate.is_file():
                    hits.append(candidate)
    if len(hits) != 1:
        raise FileNotFoundError(
            f"E4 FINETUNE ABORT: expected exactly 1 {INIT_FILE_NAME} under "
            f"/kaggle/input, found {len(hits)}: {hits}")
    return hits[0]


def load_bootstrap():
    hits = []
    base = Path("/kaggle/input")
    if base.is_dir():
        for dp, _dn, fn in os.walk(base):
            if "BUNDLE_MARKER.json" in fn and "kaggle_bootstrap.py" in fn:
                hits.append(Path(dp))
    # The init dataset mount carries no BUNDLE_MARKER.json, so it can never
    # collide here; exactly one bundle mount (stage-e-e4ft) is expected.
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 bundle mount, found {len(hits)}: {hits}")
    spec = importlib.util.spec_from_file_location(
        "kaggle_bootstrap", hits[0] / "kaggle_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hits[0]


def main() -> None:
    t_all = time.time()
    rec: dict = {"experiment": f"STAGE E - {EXP.upper()} FINETUNE", "seed": SEED,
                 "epochs": EPOCHS,
                 "intervention": "E4 broader-pretrain init; E0 recipe",
                 "initialization": "E4 pretrain best-by-pretrain-val"}

    boot, bundle = load_bootstrap()
    paths = boot.assemble()
    repo = Path(paths["repo"])
    sys.path.insert(0, str(repo))
    scripts = boot.script_dir(repo)
    pilot = scripts.parent
    outdir = pilot / "armp"

    # Pre-run assertions. Any failure stops before training.
    init = find_init()
    init_sha = sha256(init)
    assert init_sha == EXPECTED_INIT_SHA, f"init hash {init_sha}"
    rec["init_sha256"] = init_sha
    rec["init_source"] = str(init)

    import torch
    rec["environment"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
    }
    print(json.dumps(rec["environment"]), flush=True)

    # --- training -------------------------------------------------------
    cmd = [sys.executable, str(scripts / "run_arm.py"),
           "--arm", "armp", "--init", str(init), "--outdir", str(outdir),
           "--seed", str(SEED), "--epochs", str(EPOCHS),
           "--timeout_s", str(TIMEOUT_S)]
    print("+ " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(repo))
    rec["train_wall_s"] = round(time.time() - t0, 1)
    rec["run_arm_returncode"] = p.returncode

    stop = outdir / "STOP.json"
    if stop.is_file():
        rec["STOP"] = json.loads(stop.read_text())
        rec["status"] = "STOPPED"
        OUT.write_text(json.dumps(rec, indent=2))
        raise SystemExit("E4 STOPPED: " + json.dumps(rec["STOP"]))
    if p.returncode != 0:
        rec["status"] = "FAILED"
        OUT.write_text(json.dumps(rec, indent=2))
        raise SystemExit(f"run_arm.py exited {p.returncode}")

    train_record = outdir / "record.json"
    if train_record.is_file():
        rec["train_record"] = json.loads(train_record.read_text())

    # --- scoring --------------------------------------------------------
    # Import eval_tier2 for its score() only; its main() targets a control arm
    # Stage E does not have.
    spec = importlib.util.spec_from_file_location(
        "eval_tier2", scripts / "eval_tier2.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    from phase1.harness.frozen_eval import sha256_file, strict_load_report, weight_sha
    from src.models.stereonet import StereoNet, StereoNetConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec["checkpoints"] = {}
    for tag in ("best", "final"):
        ck = outdir / f"p2a_{tag}.pth"
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        state = blob["model"]
        model = StereoNet(StereoNetConfig(**ev.CFG))
        compat = strict_load_report(model, state)
        params = sum(p_.numel() for p_ in model.parameters())
        if compat["missing"] or compat["unexpected"] or params != EXPECTED_PARAMS:
            raise SystemExit(f"ABORT eval: guard failed {tag}: {json.dumps(compat)}")
        model.eval().to(device)
        entry = {"sha256": sha256_file(ck), "weight_sha16": weight_sha(state),
                 "params": params, "n_keys": len(state), "strict_load": compat,
                 "hailo_val": ev.score(model, "hailo_val", device)}
        rec["checkpoints"][tag] = entry
        v = entry["hailo_val"]
        print(f"armp/{tag} EPE {v['metrics']['epe']:.7f} D1 {v['metrics']['d1']:.4f}% "
              f"contract {v['guard']['contract_match']}", flush=True)
        del model
        torch.cuda.empty_cache()

        # Ship the checkpoint out with the kernel so it is never lost.
        import shutil
        shutil.copy2(ck, Path("/kaggle/working") / f"{EXP}_seed{SEED}_{tag}.pth")

    for extra in ("epoch_log.jsonl", "stdout.log", "integrity_guard.json"):
        src = outdir / extra
        if src.is_file():
            import shutil
            shutil.copy2(src, Path("/kaggle/working") / f"{EXP}_seed{SEED}_{extra}")

    rec["status"] = "COMPLETE"
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(rec, indent=2))
    b = rec["checkpoints"]["best"]["hailo_val"]["metrics"]
    f = rec["checkpoints"]["final"]["hailo_val"]["metrics"]
    print(f"\n{EXP.upper()} seed {SEED} COMPLETE in {rec['wall_s']}s")
    print(f"  best  EPE {b['epe']:.7f}  D1 {b['d1']:.4f}%")
    print(f"  final EPE {f['epe']:.7f}  D1 {f['d1']:.4f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
