"""E4 trainer contract: Stage-1 equivalence + probe-mode safety, no training run."""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
E4_SCRIPTS = REPO_ROOT / "stage_e_recipe" / "e4_pretrain" / "scripts"
STAGE1_SCRIPTS = REPO_ROOT / "stage_b_armp" / "20260918T062146Z_stage1_pretrain" / "scripts"

sys.path.insert(0, str(E4_SCRIPTS))
sys.path.insert(0, str(STAGE1_SCRIPTS))

import train_armp_stage1 as stage1
import train_e4_pretrain as e4


def test_epochs_is_required():
    # parse_args without --epochs must fail (argparse exits non-zero).
    with pytest.raises(SystemExit) as exc:
        e4.parse_args(["--manifest", "m.json", "--out-dir", "o"])
    assert exc.value.code != 0
    # End-to-end: invoking the script CLI without --epochs exits non-zero
    # (argparse rejects before any training code runs).
    proc = subprocess.run(
        [sys.executable, str(E4_SCRIPTS / "train_e4_pretrain.py"),
         "--manifest", "m.json", "--out-dir", "o"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0


def test_recipe_constants_identical_to_stage1():
    # Compare the ACTUAL module values -- no retyped numeric literals.
    for name in ("SCALE_LO", "SCALE_HI", "CROP_H", "CROP_W",
                 "BATCH", "LR", "MAX_DISP", "P2A_CONFIG", "P2A_PARAMS",
                 "VAL_SCENES", "SEED"):
        assert getattr(e4, name) == getattr(stage1, name), name
    # Same augmentation math object: the wrapper class source carries the
    # logUniform draw, the realised sx factor and the P2A op order.
    src = inspect.getsource(e4.ScaledCroppedFT3D)
    assert "np.log(SCALE_LO)" in src and "np.log(SCALE_HI)" in src
    assert "CROP_W / float(w)" in src
    # Same loss call with the same ceiling variable.
    main_src = inspect.getsource(e4.main)
    assert "masked_smooth_l1(out, disparity" in main_src
    assert "max_disparity=float(config.max_disparity_px)" in main_src


def test_tmax_follows_epochs():
    # Direct: the helper wires T_max to its epochs argument.
    for n in (1, 7, 20):
        opt = torch.optim.Adam(torch.nn.Linear(2, 2).parameters(), lr=0.01)
        sched = e4.build_scheduler(opt, n)
        assert sched.T_max == n
    # Wiring: main() builds the scheduler from the --epochs argument.
    main_src = inspect.getsource(e4.main)
    assert "build_scheduler(optimizer, EPOCHS)" in main_src


def test_probe_mode_writes_no_checkpoints():
    src = inspect.getsource(e4.main)
    # Probe and real record names differ, and the probe name is probe-specific.
    assert e4.RECORD_NAME != e4.PROBE_RECORD_NAME
    assert "probe" in e4.PROBE_RECORD_NAME
    assert "PROBE_RECORD_NAME" in src
    # Every checkpoint save is guarded so probe mode skips it.
    assert "not is_probe" in src
    save_lines = [ln for ln in src.splitlines() if "torch.save" in ln]
    assert save_lines, "expected checkpoint saves in main()"
    # Each torch.save must sit under a non-probe guard: check the enclosing
    # block mentions the probe guard (best-save) or lives after the
    # `if is_probe: ... return` early-exit (final-save).
    assert "if m.epe < best_val_epe and not is_probe:" in src
    probe_exit = src.index("if is_probe:")
    last_save = src.rindex("torch.save")
    assert last_save > probe_exit
    # The probe branch returns before reaching any torch.save.
    probe_block = src[probe_exit:last_save]
    assert "torch.save" not in probe_block
    assert "return" in probe_block
    # No .pth filename is written anywhere inside the probe branch.
    assert ".pth" not in probe_block
