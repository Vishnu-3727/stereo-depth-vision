"""Checks for the logic E2's verdict depends on.

The sweep itself needs checkpoints; these cover the intervention (that T = 1.0 is
the identity and that T moves sharpness in the direction claimed) and the frozen
bands.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.models.scaled_regression import StandardisedDisparityRegression  # noqa: E402
from phase2.scripts.exp_e2_readout_temperature import (  # noqa: E402
    TEMPERATURES, TemperedStandardisedDisparityRegression, band_for,
)


def _cost():
    torch.manual_seed(0)
    return torch.randn(2, 12, 5, 9)


def test_temperature_one_is_bit_identical_to_the_frozen_h2_readout():
    cost = _cost()
    frozen = StandardisedDisparityRegression()
    tempered = TemperedStandardisedDisparityRegression(1.0)
    with torch.no_grad():
        assert torch.equal(frozen(cost, (10, 18)), tempered(cost, (10, 18)))


def test_temperature_one_is_the_identity_in_the_other_ordering_too():
    cost = _cost()
    frozen = StandardisedDisparityRegression(upsample_first=False)
    tempered = TemperedStandardisedDisparityRegression(1.0, upsample_first=False)
    with torch.no_grad():
        assert torch.equal(frozen(cost, (10, 18)), tempered(cost, (10, 18)))


def test_lower_temperature_sharpens_and_higher_softens():
    """The direction of the intervention, which the interpretation depends on."""
    cost = _cost()

    def entropy(temperature):
        module = TemperedStandardisedDisparityRegression(temperature)
        module.capture = True
        with torch.no_grad():
            module(cost, (5, 9))
        scaled = module.last["softmax_input"]
        p = torch.softmax(-scaled / temperature, dim=1)
        return float(-(p * p.clamp_min(1e-30).log()).sum(dim=1).mean())

    sharp, control, soft = entropy(0.25), entropy(1.0), entropy(4.0)
    assert sharp < control < soft
    assert soft < float(torch.log(torch.tensor(12.0)))      # never exceeds ln 12


def test_the_readout_adds_no_parameters():
    assert list(TemperedStandardisedDisparityRegression(0.5).parameters()) == []


def test_bands_use_the_frozen_thresholds():
    assert band_for(0.0, 0.0) == "NEGLIGIBLE"
    assert band_for(-0.10, -1.0) == "NEGLIGIBLE"          # on the boundary
    assert band_for(-0.11, -1.01) == "MATERIAL IMPROVEMENT"
    assert band_for(+0.11, 0.0) == "MATERIAL DEGRADATION"
    assert band_for(0.0, +1.01) == "MATERIAL DEGRADATION"
    # An improvement on one metric only is not a material improvement.
    assert band_for(-0.50, 0.0) == "MATERIAL DEGRADATION"


def test_the_grid_is_the_frozen_one_and_contains_the_control():
    assert TEMPERATURES == (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
    assert 1.0 in TEMPERATURES
