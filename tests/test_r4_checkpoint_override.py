"""R4 checkpoint override checks (Stage C deploy runtime).

Covers the opt-in override added for R4: `load_frozen_net()` with no
arguments must still resolve the frozen ARMP_REL and pass its sha assert
(default behaviour unchanged), passing the E3 seed-0 final checkpoint plus
its real sha must load that checkpoint instead, and a wrong sha must raise.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "runtime"))

import armp_depth as AD  # noqa: E402
import runtime_path as RP  # noqa: E402

E3_REL = "stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth"
E3_SHA = "82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d"


def test_default_still_resolves_frozen_checkpoint():
    assert AD.ARMP_REL == (
        "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth")
    assert AD.ARMP_SHA == (
        "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454")
    net, _device = AD.load_frozen_net()  # must pass the sha assert or raise
    assert net is not None


def test_override_loads_e3_checkpoint():
    ckpt = REPO / E3_REL
    assert ckpt.exists()
    assert AD.sha256_file(ckpt) == E3_SHA  # real sha, never hardcoded by code
    net, _device = AD.load_frozen_net("cpu", checkpoint=E3_REL,
                                      expected_sha256=E3_SHA)
    assert net is not None


def test_override_wrong_sha_raises():
    with pytest.raises(AssertionError):
        AD.load_frozen_net("cpu", checkpoint=E3_REL,
                           expected_sha256="0" * 64)


def test_override_needs_both_arguments():
    with pytest.raises(ValueError):
        AD.load_frozen_net("cpu", checkpoint=E3_REL)


def test_default_path_asserts_frozen_sha():
    """Default (no --checkpoint) must assert against ARMP_SHA, not itself.

    Proves the freeze guard is live on the default path two ways, without
    copying the 64-char literal into this test twice (it is read once from
    AD.ARMP_SHA):
    1. load_net with override=False calls AD.load_frozen_net with NO
       checkpoint arguments, so armp_depth asserts the frozen constant.
    2. a corrupted sha on the default path raises: monkeypatching
       sha256_file to a wrong digest makes the default load fail, which a
       vacuous self-assert could never do.
    """
    ckpt, _label = RP.resolve_checkpoint(None)
    assert ckpt == REPO / AD.ARMP_REL
    ckpt_sha = AD.sha256_file(ckpt)
    assert ckpt_sha == AD.ARMP_SHA  # measured once, compared to the constant
    with mock.patch.object(AD, "load_frozen_net",
                           return_value=(object(), "cpu")) as m:
        RP.load_net("cpu", ckpt, ckpt_sha, override=False)
        m.assert_called_once_with("cpu")
    with mock.patch.object(AD, "load_frozen_net",
                           return_value=(object(), "cpu")) as m:
        RP.load_net("cpu", ckpt, ckpt_sha, override=True)
        m.assert_called_once_with("cpu", checkpoint=ckpt,
                                  expected_sha256=ckpt_sha)
    with mock.patch.object(AD, "sha256_file", return_value="0" * 64):
        with pytest.raises(AssertionError):
            AD.load_frozen_net("cpu")
