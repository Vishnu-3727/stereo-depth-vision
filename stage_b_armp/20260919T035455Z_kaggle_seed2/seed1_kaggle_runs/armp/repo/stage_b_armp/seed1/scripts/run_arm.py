"""TIER2-PILOT arm runner. Runs ONE arm of finetune_pilot.py UNCHANGED as a fresh
subprocess (fresh optimizer/scheduler by construction), tees stdout to
stdout.log, enforces STOP conditions live, then writes epoch_log.jsonl
(enriched per-epoch log) and record.json atomically.

Usage (invoked by run_pilot.py, not by hand):
    <python> run_arm.py --arm control --init random --outdir <dir> --timeout_s 7200
    <python> run_arm.py --arm armp --init <ckpt.pth> --outdir <dir> --timeout_s 7200

STOP conditions (halt arm, write STOP.json, exit 3; never silently repair):
  NaN/Inf loss | zero valid pixels | param count != 397954 | strict-load
  failure | wrong-checkpoint resume | wall > timeout | projected wall > timeout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
REPO = PILOT.parents[1]
SCRIPT = PILOT / "scripts" / "finetune_pilot.py"
EXPECTED_PARAMS = 397954
EXPECTED_SHA_ARMP_STAGE1 = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"

EPOCH_RE = re.compile(r"^epoch\s+(\d+)\s+loss\s+([0-9eE+\-.,infINFnanNAN]+)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def stop(outdir: Path, reason: str, detail: dict) -> int:
    payload = {"arm": outdir.name, "status": "STOPPED", "reason": reason,
               "detail": detail, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_text(outdir / "STOP.json", json.dumps(payload, indent=2))
    print("STOP[%s]: %s %s" % (outdir.name, reason, json.dumps(detail)), flush=True)
    return 3


def looks_nonfinite(token: str) -> bool:
    t = token.strip().lower()
    if t in ("nan", "+nan", "-nan", "inf", "+inf", "-inf"):
        return True
    try:
        return not math.isfinite(float(t))
    except ValueError:
        return True  # unparseable loss token -> treat as stop, never repair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timeout_s", type=float, default=7200.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stdout_log = outdir / "stdout.log"
    arm_start = time.time()
    pid_self = os.getpid()

    env = dict(os.environ)
    env["P2A_SEED"] = "1"
    env["P2A_OUT_DIR"] = str(outdir)
    env["P2A_EPOCHS"] = "200"

    cmd = [sys.executable, str(SCRIPT), "--init", args.init, "--arm", args.arm]
    print("RUN arm=%s init=%s pid=%d cmd=%s" % (args.arm, args.init, pid_self, cmd), flush=True)

    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    epoch_times: dict[int, float] = {}
    nan_hit = None
    abort_line = None
    with open(stdout_log, "w", encoding="utf-8") as fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.time()
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
            m = EPOCH_RE.match(line.strip())
            if m:
                ep = int(m.group(1))
                epoch_times[ep] = now
                if looks_nonfinite(m.group(2)):
                    nan_hit = {"epoch": ep, "loss_token": m.group(2)}
                    proc.kill()
                    break
                lo, hi = line.lower(), line.strip().lower()
                if "nan" in lo or "inf" in lo:
                    nan_hit = {"epoch": ep, "line": line.strip()}
                    proc.kill()
                    break
                done = len(epoch_times)
                if done >= 5:
                    proj = (now - arm_start) / done * 200.0
                    if proj > args.timeout_s:
                        proc.wait(timeout=60)
                        return stop(outdir, "projected_wall_exceeds_2h",
                                    {"elapsed_s": now - arm_start, "epochs_done": done,
                                     "projected_s": proj, "timeout_s": args.timeout_s})
            s = line.strip()
            if s.startswith("ABORT") or "Traceback" in s:
                abort_line = s[:500]
            if (now - arm_start) > args.timeout_s:
                proc.kill()
                proc.wait(timeout=60)
                return stop(outdir, "wall_exceeds_2h",
                            {"elapsed_s": now - arm_start, "timeout_s": args.timeout_s})
    rc = proc.wait()
    wall_s = time.time() - arm_start
    if nan_hit is not None:
        return stop(outdir, "nan_inf_loss", nan_hit)
    if rc != 0:
        return stop(outdir, "nonzero_exit_strict_load_or_crash",
                    {"returncode": rc, "abort_hint": abort_line})

    # ---- post-hoc guards (all read-only checks on the finished arm) ----
    log_path = outdir / "training_log.jsonl"
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 200:
        return stop(outdir, "epochs_incomplete", {"rows": len(rows)})
    for r in rows:
        if not math.isfinite(r["mean_loss"]):
            return stop(outdir, "nan_inf_loss_in_log", {"epoch": r["epoch"]})
        if r.get("valid_pixels", 0) <= 0:
            return stop(outdir, "zero_valid_pixels", {"epoch": r["epoch"]})
        for k in ("val_epe", "val_d1"):
            if k in r and not math.isfinite(r[k]):
                return stop(outdir, "nan_inf_val_metric", {"epoch": r["epoch"], "key": k})

    guard = json.loads((outdir / "integrity_guard.json").read_text(encoding="utf-8"))
    if guard.get("param_count") != EXPECTED_PARAMS or not guard.get("all_ok"):
        return stop(outdir, "param_or_architecture_changed", {"guard": guard})

    rec = json.loads((outdir / "p2a_record.json").read_text(encoding="utf-8"))
    if rec.get("epochs_run") != 200:
        return stop(outdir, "epochs_incomplete", {"epochs_run": rec.get("epochs_run")})
    init_rec = rec.get("config", {}).get("init_record", {})
    if args.init == "random":
        if init_rec.get("source") != "random":
            return stop(outdir, "wrong_checkpoint_resume", {"init_record": init_rec})
        init_sha = "random"
    else:
        if init_rec.get("sha256") != EXPECTED_SHA_ARMP_STAGE1:
            return stop(outdir, "wrong_checkpoint_resume", {"init_record": init_rec})
        init_sha = init_rec["sha256"]

    best_ckpt = outdir / "p2a_best.pth"
    final_ckpt = outdir / "p2a_final.pth"
    best_sha = sha256_file(best_ckpt)
    final_sha = sha256_file(final_ckpt)
    atomic_write_text(outdir / "p2a_best.sha256", best_sha + "  p2a_best.pth\n")
    atomic_write_text(outdir / "p2a_final.sha256", final_sha + "  p2a_final.pth\n")
    if rec.get("best_sha256") != best_sha or rec.get("final_sha256") != final_sha:
        return stop(outdir, "checkpoint_sha_mismatch",
                    {"record_best": rec.get("best_sha256"), "actual_best": best_sha,
                     "record_final": rec.get("final_sha256"), "actual_final": final_sha})

    # ---- enriched per-epoch log: required columns + epoch_wall_s ----
    t_prev = arm_start
    enriched = []
    for r in rows:
        ep = r["epoch"]
        t_ep = epoch_times.get(ep)
        if t_ep is not None:
            ewall = t_ep - t_prev
            t_prev = t_ep
        else:
            ewall = None  # honest null; never fabricate timing
        enriched.append({"epoch": ep, "mean_loss": r["mean_loss"], "lr": r["lr"],
                         "val_epe": r.get("val_epe"), "val_d1": r.get("val_d1"),
                         "epoch_wall_s": ewall, "valid_pixels": r["valid_pixels"]})
    atomic_write_text(outdir / "epoch_log.jsonl",
                      "\n".join(json.dumps(x) for x in enriched) + "\n")

    final_lr = rows[-1]["lr"]
    record = {"arm": args.arm, "status": "COMPLETE", "seed": 1,
              "init": args.init, "init_sha256": init_sha,
              "epochs_completed": 200, "training_duration_s": wall_s,
              "train_monitor_best_epe_10scene": rec.get("best_val_epe_10scene"),
              "train_monitor_best_epoch": rec.get("best_epoch"),
              "final_lr": final_lr, "param_count": EXPECTED_PARAMS,
              "best_sha256": best_sha, "final_sha256": final_sha,
              "train_pid": proc.pid, "runner_pid": pid_self,
              "note": ("train_monitor numbers are the 10-scene in-training monitor, NOT the "
                       "frozen contract; frozen EPE/D1 land in pilot_results.json after eval. "
                       "Fresh process => fresh optimizer/scheduler (train_pid recorded).")}
    atomic_write_text(outdir / "record.json", json.dumps(record, indent=2))
    print("ARM %s COMPLETE wall=%.1fs best10=%.4f@%d final_lr=%.2e" % (
        args.arm, wall_s, rec.get("best_val_epe_10scene"), rec.get("best_epoch"), final_lr),
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
