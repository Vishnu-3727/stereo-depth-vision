"""STAGE_E_KAGGLE_USER override for stage_e_recipe/kaggle/push_run.py."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_e_recipe" / "kaggle"))

import push_run


def test_unset_env_resolves_to_default(monkeypatch):
    monkeypatch.delenv("STAGE_E_KAGGLE_USER", raising=False)
    importlib.reload(push_run)
    assert push_run.resolve_user() == "vishnu3727"
    assert push_run.USER == "vishnu3727"
    assert push_run.DATA_SLUG == "vishnu3727/kitti2015-tier2-seed2-subset"
    assert push_run.BUNDLES["e0"] == "vishnu3727/stage-e-recipe-bundle"
    assert push_run.slug(0, "e0") == "vishnu3727/stage-e-e0-seed0"


def test_set_env_overrides(monkeypatch):
    monkeypatch.setenv("STAGE_E_KAGGLE_USER", "newaccount123")
    importlib.reload(push_run)
    assert push_run.resolve_user() == "newaccount123"
    assert push_run.USER == "newaccount123"
    assert push_run.DATA_SLUG == "newaccount123/kitti2015-tier2-seed2-subset"
    assert push_run.slug(0, "e0") == "newaccount123/stage-e-e0-seed0"
    # cleanup: restore default for any later tests in the session
    monkeypatch.delenv("STAGE_E_KAGGLE_USER", raising=False)
    importlib.reload(push_run)
    assert push_run.USER == "vishnu3727"
