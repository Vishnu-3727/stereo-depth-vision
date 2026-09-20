"""H1 regression test: the working cost volume must genuinely shift.

Phase 1's central finding was that the frozen baseline's cost volume is
degenerate -- every disparity slice is bit-identical (EXP-010). Phase 2's first
change is to enable the shift that already exists in the shared model code
(``src/models/stereonet/cost_volume.py::shift_left``), behind the
``cost_volume_shift="left"`` flag Phase 1 never turned on.

This test pins two things at once, so Phase 2 can never silently regress into
Phase 1's defect:

1. ``shift="left"`` produces slices that actually differ from each other.
2. ``shift="none"`` still reproduces the frozen degeneracy exactly, unchanged --
   proving this test exercises the real code path Phase 1 measured, not a
   reimplementation of it.

Run: ``python -m pytest phase2/tests/test_h1_cost_volume_shift.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.stereonet.cost_volume import (  # noqa: E402
    build_cost_volume,
    build_cost_volume_reference,
)


def _random_features(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # non-degenerate: right features must vary along width, or a shift of a
    # constant-in-x tensor would trivially "differ" for the wrong reason.
    left = torch.randn(1, 4, 5, 9, generator=g)
    right = torch.randn(1, 4, 5, 9, generator=g)
    return left, right


def test_shift_none_reproduces_frozen_degeneracy() -> None:
    """Phase 1's EXP-010 finding, unregressed: shift="none" gives 12 identical slices."""
    left, right = _random_features()
    volume = build_cost_volume(left, right, num_disparities=12, shift="none")
    slice0 = volume[:, :, 0]
    for k in range(1, 12):
        assert torch.equal(volume[:, :, k], slice0), (
            "shift='none' must stay degenerate -- Phase 1's frozen finding"
        )


def test_shift_left_produces_genuinely_different_slices() -> None:
    """The Phase 2 change: shift="left" must NOT reproduce the degeneracy."""
    left, right = _random_features()
    volume = build_cost_volume(left, right, num_disparities=12, shift="left")
    slice0 = volume[:, :, 0]
    differing = [
        k for k in range(1, 12)
        if not torch.equal(volume[:, :, k], slice0)
    ]
    assert differing == list(range(1, 12)), (
        "shift='left' must make every candidate slice differ from slice 0; "
        "slices that matched: " + str([k for k in range(1, 12) if k not in differing])
    )


def test_shift_left_matches_explicit_indexed_reference() -> None:
    """Vectorised shift='left' agrees with the independently-written, explicitly
    indexed reference implementation -- agreement here is evidence, not a
    tautology, since the two were written separately (mirrors EXP-011's method)."""
    left, right = _random_features()
    vectorised = build_cost_volume(left, right, num_disparities=12, shift="left")
    indexed = build_cost_volume_reference(left, right, num_disparities=12, shift="left")
    assert torch.allclose(vectorised, indexed, atol=1e-6)


def test_shift_left_boundary_is_zero_padded() -> None:
    """Disparity candidate k reads left[x + k]; columns beyond the right edge
    must be treated as zero (the documented boundary policy), not wrapped."""
    left, right = _random_features()
    w = left.shape[-1]
    volume = build_cost_volume(left, right, num_disparities=12, shift="left")
    k = 5
    # column w-1 reads left[:, :, :, w-1+k], out of range -> zero, so
    # cost = 0 - right[..., w-1]
    got = volume[:, :, k, :, w - 1]
    expected = -right[:, :, :, w - 1]
    assert torch.allclose(got, expected, atol=1e-6)


if __name__ == "__main__":
    test_shift_none_reproduces_frozen_degeneracy()
    test_shift_left_produces_genuinely_different_slices()
    test_shift_left_matches_explicit_indexed_reference()
    test_shift_left_boundary_is_zero_padded()
    print("H1 cost-volume shift regression checks passed")
