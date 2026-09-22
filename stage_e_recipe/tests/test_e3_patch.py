#!/usr/bin/env python
"""E3 correctness gate: the epoch-guard fix does what it claims.

The E3 runs trained all 400 epochs and then STOPped with
`epochs_incomplete / {rows: 400}`, because run_arm.py's two post-hoc guards
hardcoded the literal 200. e3_patch.py repoints both guards at the arm's own
`--epochs` argument. This test exercises the patched guards — the verbatim
post-hoc block from run_arm.py, exec'd by recover_e3.run_posthoc_guards —
against synthetic arm directories. CPU only, stdlib only, no data, no
checkpoint, no training, no Kaggle: this is guard logic, not a run.

Checks:
  1. the patch applies cleanly (every EDITS `old` occurs exactly once)
  2. the patched guards accept a 400-row log (all guards pass, no stop)
  3. the patched guards reject a 399-row log (epochs_incomplete, rows=399)
  4. the patched guards reject an epochs_run mismatch (epochs_incomplete)
  5. no `!= 200` epoch literal remains in the patched file

    python stage_e_recipe/tests/test_e3_patch.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KAGGLE = REPO / "stage_e_recipe" / "kaggle"
sys.path.insert(0, str(KAGGLE))

import e3_patch  # noqa: E402
import recover_e3  # noqa: E402

RUN_ARM_SRC = (REPO / "stage_b_armp" / "20260919T012646Z_tier2_seed1"
               / "scripts" / "run_arm.py")
EXPECTED_PARAMS = 397954
EXPECTED_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"


def patched_text() -> str:
    return e3_patch.apply(RUN_ARM_SRC.read_text(encoding="utf-8"))


def make_arm(parent: Path, n_rows: int, epochs_run: int) -> Path:
    """Build a synthetic arm dir: valid log rows plus consistent metadata."""
    outdir = parent / "armp"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [{"epoch": ep, "mean_loss": 2.0 / (1 + ep * 0.01),
             "valid_pixels": 5000000, "lr": 1e-3 * (1 - ep / 800.0),
             "val_epe": 1.6, "val_d1": 6.5} for ep in range(n_rows)]
    (outdir / "training_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (outdir / "integrity_guard.json").write_text(json.dumps(
        {"param_count": EXPECTED_PARAMS, "all_ok": True}), encoding="utf-8")
    best = outdir / "p2a_best.pth"
    final = outdir / "p2a_final.pth"
    best.write_bytes(b"fake-best-checkpoint")
    final.write_bytes(b"fake-final-checkpoint")

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    (outdir / "p2a_record.json").write_text(json.dumps({
        "epochs_run": epochs_run,
        "config": {"init_record": {"source": "/kaggle/working/repo/checkpoints/x.pth",
                                   "sha256": EXPECTED_SHA}},
        "best_sha256": sha(best), "final_sha256": sha(final),
        "best_val_epe_10scene": 1.6, "best_epoch": 0,
    }), encoding="utf-8")
    return outdir


def test_patch_applies_cleanly():
    src = RUN_ARM_SRC.read_text(encoding="utf-8")
    for old, new, why in e3_patch.EDITS:
        assert src.count(old) == 1, f"EDITS target not unique: {why}"
    text = e3_patch.apply(src)  # must not raise
    assert "if len(rows) != args.epochs:" in text
    assert 'if rec.get("epochs_run") != args.epochs:' in text


def test_no_epoch_literal_left():
    assert "!= 200" not in patched_text()


def test_accepts_400_rows():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = make_arm(Path(tmp), 400, 400)
        result = recover_e3.run_posthoc_guards(outdir, 400, patched_text())
    assert result["stop"] is None, result["stop"]
    assert all(v == "pass" for v in result["guards"].values()), result["guards"]


def test_rejects_399_rows():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = make_arm(Path(tmp), 399, 399)
        result = recover_e3.run_posthoc_guards(outdir, 400, patched_text())
    assert result["stop"] is not None
    assert result["stop"]["reason"] == "epochs_incomplete"
    assert result["stop"]["detail"] == {"rows": 399}
    assert result["guards"]["rows_count"] == "fail"


def test_rejects_epochs_run_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = make_arm(Path(tmp), 400, 200)
        result = recover_e3.run_posthoc_guards(outdir, 400, patched_text())
    assert result["stop"] is not None
    assert result["stop"]["reason"] == "epochs_incomplete"
    assert result["stop"]["detail"] == {"epochs_run": 200}
    assert result["guards"]["epochs_run"] == "fail"


def test_bundle_e3_ships_patched_guards():
    bundle_run_arm = KAGGLE / "bundle_e3" / "scripts" / "run_arm.py"
    if not bundle_run_arm.is_file():
        return  # bundle not built on this machine; nothing to tie in
    text = bundle_run_arm.read_text(encoding="utf-8")
    assert "if len(rows) != args.epochs:" in text
    assert 'if rec.get("epochs_run") != args.epochs:' in text
    assert "!= 200" not in text


def main() -> None:
    failures = []

    def check(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {e!r}")
            failures.append(name)
        else:
            print(f"PASS  {name}")

    check("patch_applies_cleanly", test_patch_applies_cleanly)
    check("no_epoch_literal_left", test_no_epoch_literal_left)
    check("accepts_400_rows", test_accepts_400_rows)
    check("rejects_399_rows", test_rejects_399_rows)
    check("rejects_epochs_run_mismatch", test_rejects_epochs_run_mismatch)
    check("bundle_e3_ships_patched_guards", test_bundle_e3_ships_patched_guards)

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
