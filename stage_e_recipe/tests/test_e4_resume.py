#!/usr/bin/env python
"""E4 resume gate: a resumed run continues the same stream.

Exercises the exact state round-trip the per-epoch resume.pt checkpoint in
stage_e_recipe/e4_pretrain/scripts/train_e4_pretrain.py performs, on toy
objects. CPU only, no dataset, no GPU, no training run, no network: this is
stream continuity, not a run.

The point is to prove a resumed run continues the same random streams,
optimizer trajectory and LR schedule -- not that the file has the right keys.

Checks:
  1. the atomic save leaves a loadable resume.pt and no .tmp behind
  2. the restored model weights exactly equal the uninterrupted ones
  3. the restored optimizer state (tensors and param-group hypers) equals it
  4. after one more scheduler step on each side, the next LR matches --
     i.e. the cosine schedule continues, not restarts
  5. the next draws from the augmentation np Generator match -- i.e. the
     augmentation stream continues, not replays
  6. the next draws from the global torch-CPU, numpy and python-random
     streams match

    python stage_e_recipe/tests/test_e4_resume.py
    pytest stage_e_recipe/tests/test_e4_resume.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

EPOCHS = 8
N_STEPS_BEFORE_SAVE = 3


class Toy(torch.nn.Module):
    """Tiny stand-in for the model: a few parameters Adam can step over."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 8)
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def make_trio():
    """Fresh toy model + Adam + CosineAnnealingLR, mirroring the script."""
    model = Toy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    return model, opt, sched


def step_once(model, opt, sched, aug_rng, x, y):
    """One deterministic optimizer+scheduler step plus augmentation draws."""
    _s = float(np.exp(aug_rng.uniform(np.log(0.7), np.log(1.7))))
    _j = float(aug_rng.normal(0, 0.1))
    opt.zero_grad(set_to_none=True)
    loss = ((model(x * (1.0 + _j)) - y) ** 2).mean()
    loss.backward()
    opt.step()
    sched.step()
    return _s, _j


def atomic_torch_save(path: Path, payload: dict) -> None:
    """Byte-for-byte the discipline train_e4_pretrain.save_atomic_torch uses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def collect_rng_states() -> dict:
    """Byte-for-byte what train_e4_pretrain.collect_rng_states snapshots."""
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_states(blob: dict) -> None:
    """Byte-for-byte what train_e4_pretrain.restore_rng_states restores."""
    torch.set_rng_state(blob["torch_cpu"])
    if blob.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["torch_cuda"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["python"])


def assert_opt_state_equal(a: dict, b: dict) -> None:
    """Recursive equality over Adam state_dicts (tensors + hypers)."""
    assert a["param_groups"] == b["param_groups"], "param_groups differ"
    assert set(a["state"].keys()) == set(b["state"].keys()), "state keys differ"
    for k in a["state"]:
        assert set(a["state"][k].keys()) == set(b["state"][k].keys()), f"state[{k}] keys differ"
        for kk in a["state"][k]:
            va, vb = a["state"][k][kk], b["state"][k][kk]
            if isinstance(va, torch.Tensor):
                assert isinstance(vb, torch.Tensor) and torch.equal(va, vb), f"state[{k}][{kk}] differs"
            else:
                assert va == vb, f"state[{k}][{kk}] differs: {va!r} vs {vb!r}"


def run_round_trip(tmp_path: Path) -> dict:
    """Drive the uninterrupted-vs-resumed comparison; return what was checked."""
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    model, opt, sched = make_trio()
    aug_rng = np.random.default_rng(0)
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24.0
    y = torch.arange(4, dtype=torch.float32).reshape(4, 1)

    for _ in range(N_STEPS_BEFORE_SAVE):
        step_once(model, opt, sched, aug_rng, x, y)

    # The same set of state objects the resume checkpoint saves.
    payload = {
        "epochs": EPOCHS,
        "last_completed_epoch": N_STEPS_BEFORE_SAVE - 1,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "aug_rng_state": aug_rng.bit_generator.state,
        "rng": collect_rng_states(),
    }
    ckpt = tmp_path / "resume.pt"
    atomic_torch_save(ckpt, payload)

    # Uninterrupted side: continue the original objects one more step.
    ref_aug = aug_rng.random(4)
    ref_torch = torch.randn(4)
    ref_np = np.random.random(4)
    ref_py = [random.random() for _ in range(4)]
    step_once(model, opt, sched, aug_rng, x, y)
    ref_lr = sched.get_last_lr()[0]
    ref_opt_state = opt.state_dict()

    # Resumed side: fresh objects, load everything back, continue one step.
    fresh_model, fresh_opt, fresh_sched = make_trio()
    fresh_aug = np.random.default_rng(12345)  # deliberately NOT 0: load must fix it
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    fresh_model.load_state_dict(blob["model"])
    fresh_opt.load_state_dict(blob["optimizer"])
    fresh_sched.load_state_dict(blob["scheduler"])
    fresh_aug.bit_generator.state = blob["aug_rng_state"]
    restore_rng_states(blob["rng"])
    got_aug = fresh_aug.random(4)
    got_torch = torch.randn(4)
    got_np = np.random.random(4)
    got_py = [random.random() for _ in range(4)]
    step_once(fresh_model, fresh_opt, fresh_sched, fresh_aug, x, y)
    got_lr = fresh_sched.get_last_lr()[0]
    got_opt_state = fresh_opt.state_dict()

    return {
        "ckpt": ckpt,
        "ref_model_state": model.state_dict(),
        "got_model_state": fresh_model.state_dict(),
        "ref_opt_state": ref_opt_state,
        "got_opt_state": got_opt_state,
        "ref_lr": ref_lr,
        "got_lr": got_lr,
        "ref_aug": ref_aug,
        "got_aug": got_aug,
        "ref_torch": ref_torch,
        "got_torch": got_torch,
        "ref_np": ref_np,
        "got_np": got_np,
        "ref_py": ref_py,
        "got_py": got_py,
    }


def test_atomic_save_leaves_no_tmp(tmp_path):
    r = run_round_trip(tmp_path)
    assert r["ckpt"].is_file()
    assert list(tmp_path.glob("*.tmp")) == []
    blob = torch.load(str(r["ckpt"]), map_location="cpu", weights_only=False)
    assert blob["epochs"] == EPOCHS
    assert blob["last_completed_epoch"] == N_STEPS_BEFORE_SAVE - 1


def test_model_weights_round_trip(tmp_path):
    r = run_round_trip(tmp_path)
    assert set(r["got_model_state"].keys()) == set(r["ref_model_state"].keys())
    for k in r["ref_model_state"]:
        assert torch.equal(r["got_model_state"][k], r["ref_model_state"][k]), k


def test_optimizer_state_round_trip(tmp_path):
    r = run_round_trip(tmp_path)
    # after the same one further step on each side, the resumed optimizer
    # must equal the uninterrupted trajectory
    assert_opt_state_equal(r["got_opt_state"], r["ref_opt_state"])


def test_scheduler_lr_continues(tmp_path):
    r = run_round_trip(tmp_path)
    assert r["got_lr"] == r["ref_lr"], f"{r['got_lr']!r} vs {r['ref_lr']!r}"


def test_augmentation_stream_continues(tmp_path):
    r = run_round_trip(tmp_path)
    assert np.array_equal(r["got_aug"], r["ref_aug"]), f"{r['got_aug']!r} vs {r['ref_aug']!r}"


def test_global_rng_streams_continue(tmp_path):
    r = run_round_trip(tmp_path)
    assert torch.equal(r["got_torch"], r["ref_torch"])
    assert np.array_equal(r["got_np"], r["ref_np"])
    assert r["got_py"] == r["ref_py"]


def main() -> None:
    import tempfile

    failures = []

    def check(name, fn):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {e!r}")
            failures.append(name)
        else:
            print(f"PASS  {name}")

    check("atomic_save_leaves_no_tmp", test_atomic_save_leaves_no_tmp)
    check("model_weights_round_trip", test_model_weights_round_trip)
    check("optimizer_state_round_trip", test_optimizer_state_round_trip)
    check("scheduler_lr_continues", test_scheduler_lr_continues)
    check("augmentation_stream_continues", test_augmentation_stream_continues)
    check("global_rng_streams_continue", test_global_rng_streams_continue)

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
