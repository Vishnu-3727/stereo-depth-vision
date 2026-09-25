"""Determinism controls: single importable entry point.

Thin wrapper over the settings Stage B demonstrated
(`phase2/diagnostics/determinism/stageb_deterministic_baseline.py:132-143`).
No new semantics: fixed seeds, deterministic algorithms, cudnn flags, and the
env var they require. DataLoader helpers match the frozen training recipe
(`scripts/exp_train_convergence.py:131`: num_workers=0).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

CUBLAS_CONFIG = ":4096:8"  # per stageb_deterministic_baseline.py:42-44

EXACT_SETTINGS = {
    "torch.use_deterministic_algorithms": True,
    "torch.backends.cudnn.deterministic": True,
    "torch.backends.cudnn.benchmark": False,
    "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
}


def seed_all(seed: int) -> None:
    """Fix python, numpy, torch (CPU+CUDA) seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism() -> dict:
    """Apply the Stage B controls. Raises if CUBLAS config is wrong."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG, (
        "CUBLAS_WORKSPACE_CONFIG must be %r before the CUDA context exists; got %r"
        % (CUBLAS_CONFIG, os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    )
    return dict(EXACT_SETTINGS)


def ensure_cublas_env() -> None:
    """Set CUBLAS_WORKSPACE_CONFIG if absent. Call before CUDA init."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CONFIG:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically (gives base_seed+worker_id)."""
    info = torch.utils.data.get_worker_info()
    base = info.seed if info is not None else 0
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % (2 ** 32))


def make_generator(seed: int) -> torch.Generator:
    """Seeded generator for DataLoader(shuffle=generator)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def flag_snapshot() -> dict:
    """Record the effective deterministic flags (for provenance)."""
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


if __name__ == "__main__":
    ensure_cublas_env()
    seed_all(0)
    print(enable_determinism())
    print(flag_snapshot())
