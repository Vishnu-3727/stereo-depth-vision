"""Kaggle portability wrapper — ARM-P Tier-2 seed-2 replication (seed 2 of P2A).

`finetune_pilot.py` resolves paths as REPO_ROOT/data/kitti2015 (where
REPO_ROOT is parents[3] of scripts/finetune_pilot.py) and imports `src.*`
and `phase1.harness.*`. Those scripts MUST stay byte-identical to the seed-1
versions, so every portability accommodation lives HERE, in this wrapper,
and is listed in portability_notes.md. Research source is never edited.

`finetune_pilot.py:39` uses parents[3], and `run_arm.py` uses
PILOT=parents[1] / REPO=PILOT.parents[1], so the scripts must live exactly
three levels under the repo root::

    <root>/stage_b_armp/seed2/scripts/finetune_pilot.py
    <root>/stage_b_armp/seed2/scripts/run_arm.py
    ...

At runtime on Kaggle this module assembles::

    /kaggle/working/repo/                        (<root>)
        src/  phase1/  checkpoints/  configs/    (copied from the bundle)
        stage_b_armp/seed2/scripts/              (copied from the bundle)
        data/kitti2015 -> <read-only dataset mount>   (symlink)

If the dataset mount was auto-extracted WITHOUT the training/ level (flat
image_2/image_3/disp_occ_0 at the mount root, as observed for dataset A),
data/kitti2015 is instead a real dir whose training/ level symlinks each
split dir back into the read-only mount — Kitti2015Stereo still sees
training/*.

and then invokes the UNMODIFIED scripts as fresh subprocesses with
cwd=<root>, so REPO_ROOT resolves to the assembled tree.

Stdlib only — this module must never import research code, so that the
smoke test's import-only audit stays attributable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BUNDLE_MARKER = "BUNDLE_MARKER.json"
# Dirs that live directly under <root>.
ROOT_DIRS = ("src", "phase1", "checkpoints", "configs")
# Scripts live three levels under <root> so that parents[3] == <root>.
EXP_SUBDIR = Path("stage_b_armp") / "seed2"
DEFAULT_REPO_ROOT = Path("/kaggle/working/repo")


def script_dir(repo_root: Path | str) -> Path:
    """Nested scripts dir: <root>/stage_b_armp/seed2/scripts/."""
    return Path(repo_root) / EXP_SUBDIR / "scripts"


def _walk_markers(name: str) -> list[Path]:
    """Recursive search for dirs directly containing `name` under /kaggle/input."""
    hits = []
    base = Path("/kaggle/input")
    if base.is_dir():
        for dp, _dn, fn in os.walk(base):
            if name in fn:
                hits.append(Path(dp))
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))


def find_bundle_root() -> Path:
    """Locate the code+checkpoint dataset mount (contains BUNDLE_MARKER.json)."""
    env = os.environ.get("KAGGLE_BUNDLE_ROOT")
    if env:
        p = Path(env)
        if (p / BUNDLE_MARKER).is_file():
            return p
        raise FileNotFoundError("KAGGLE_BUNDLE_ROOT has no marker: " + str(p))
    cands = _walk_markers(BUNDLE_MARKER)
    if len(cands) != 1:
        raise FileNotFoundError(
            "expected exactly 1 dir with %s under /kaggle/input, found %d: %s"
            % (BUNDLE_MARKER, len(cands), [str(c) for c in cands]))
    return cands[0]


def _has_training_layout(p: Path) -> bool:
    return (p / "training" / "image_2").is_dir()


def _has_flat_layout(p: Path) -> bool:
    # Kaggle auto-extracted dataset A without the top-level training/ dir:
    # image_2, image_3, disp_occ_0 sit directly at the mount root.
    return (p / "image_2").is_dir() and (p / "image_3").is_dir() \
        and (p / "disp_occ_0").is_dir()


def data_layout(p: Path) -> str | None:
    """Return 'training', 'flat', or None (unrecognised mount)."""
    if _has_training_layout(p):
        return "training"
    if _has_flat_layout(p):
        return "flat"
    return None


def find_data_root() -> Path:
    """Locate the KITTI subset mount (training/image_2, or flat image_2)."""
    env = os.environ.get("KAGGLE_DATA_ROOT")
    if env:
        p = Path(env)
        if data_layout(p) is not None:
            return p
        raise FileNotFoundError("KAGGLE_DATA_ROOT has neither training/image_2 "
                                "nor flat image_2: " + str(p))
    base = Path("/kaggle/input")
    cands: list[Path] = []
    if base.is_dir():
        # mount roots first, then any nested dir (zip extraction may nest)
        roots = sorted(d for d in base.glob("*") if d.is_dir())
        nested: list[Path] = []
        for r in roots:
            for dp, _dn, _fn in os.walk(r):
                nested.append(Path(dp))
        for d in roots + sorted(set(nested) - set(roots),
                                key=lambda p: (len(p.parts), str(p))):
            if data_layout(d) is not None:
                cands.append(d)
                if d in roots:
                    break  # a mount root match wins immediately
    # the bundle mount also matches if it ever embeds data; prefer non-bundle
    cands = [d for d in cands if not (d / BUNDLE_MARKER).is_file()] or cands
    if len(cands) != 1:
        raise FileNotFoundError(
            "expected exactly 1 KITTI dir with training/image_2 (or flat "
            "image_2), found %d: %s"
            % (len(cands), [str(c) for c in cands]))
    return cands[0]


def assemble(repo_root: Path | str = DEFAULT_REPO_ROOT,
             bundle_root: Path | str | None = None,
             data_root: Path | str | None = None) -> dict:
    """Assemble the working tree. Returns {'repo': ..., 'bundle': ..., 'data': ...}."""
    repo = Path(repo_root)
    bundle = Path(bundle_root) if bundle_root else find_bundle_root()
    data = Path(data_root) if data_root else find_data_root()
    if not (bundle / BUNDLE_MARKER).is_file():
        raise FileNotFoundError("bundle marker missing: " + str(bundle))
    layout = data_layout(data)
    if layout is None:
        raise FileNotFoundError("data mount has neither training/image_2 nor "
                                "flat image_2/image_3/disp_occ_0: " + str(data))

    repo.mkdir(parents=True, exist_ok=True)
    for d in ROOT_DIRS:
        src, dst = bundle / d, repo / d
        if not src.is_dir():
            raise FileNotFoundError("bundle dir missing: " + str(src))
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Scripts nest three levels under <root> so that parents[3] == <root>
    # (finetune_pilot.py:39 REPO_ROOT; run_arm.py PILOT/REPO).
    s_src, s_dst = bundle / "scripts", script_dir(repo)
    if not s_src.is_dir():
        raise FileNotFoundError("bundle dir missing: " + str(s_src))
    if s_dst.exists():
        shutil.rmtree(s_dst)
    s_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(s_src, s_dst)
    # Guard the depth contract explicitly: parents[3] of the assembled
    # finetune_pilot.py MUST be <root>.
    resolved = (s_dst / "finetune_pilot.py").resolve().parents[3]
    if resolved != repo.resolve():
        raise RuntimeError("nesting depth wrong: parents[3]=%s != repo=%s"
                           % (resolved, repo.resolve()))
    # run_arm.py's own depth contract: PILOT=parents[1], REPO=PILOT.parents[1].
    _run_arm = s_dst / "run_arm.py"
    if _run_arm.is_file():
        pilot = _run_arm.resolve().parents[1]
        if pilot.parents[1] != repo.resolve():
            raise RuntimeError("run_arm.py depth wrong: PILOT.parents[1]=%s != repo=%s"
                               % (pilot.parents[1], repo.resolve()))

    link = repo / "data" / "kitti2015"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    if layout == "training":
        try:
            os.symlink(str(data), str(link), target_is_directory=True)
            linked = "symlink"
        except OSError as exc:
            raise OSError("cannot symlink %s -> %s: %s "
                          "(no copy fallback: the dataset is ~655MB and the "
                          "read-only mount must stay the single source of truth)"
                          % (link, data, exc))
    else:
        # Flat mount (observed for dataset A: Kaggle extracted the upload
        # without the training/ level). Kitti2015Stereo is NOT edited; the
        # wrapper builds the expected tree: a real kitti2015/ dir whose
        # training/ level symlinks each split dir back into the read-only
        # mount, so the mount stays the single source of truth.
        training = link / "training"
        training.mkdir(parents=True, exist_ok=True)
        try:
            for d in ("image_2", "image_3", "disp_occ_0"):
                os.symlink(str(data / d), str(training / d),
                           target_is_directory=True)
            linked = "training-shim"
        except OSError as exc:
            raise OSError("cannot build training/ shim under %s from %s: %s"
                          % (link, data, exc))
    return {"repo": str(repo), "bundle": str(bundle), "data": str(data),
            "data_link": linked, "data_layout": layout,
            "scripts": str(s_dst)}


def run_script(repo_root: Path | str, name: str, args: list[str],
               env_extra: dict | None = None,
               timeout_s: float = 7200.0) -> subprocess.CompletedProcess:
    """Run an UNMODIFIED bundled script as a fresh process (cwd=repo_root).

    Scripts live nested at <root>/stage_b_armp/seed2/scripts/ (see
    assemble()); cwd is still <root> so relative paths in the scripts
    resolve against the repo root.
    """
    repo = Path(repo_root)
    script = script_dir(repo) / name
    if not script.is_file():
        raise FileNotFoundError("script missing (nested): " + str(script))
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(script)] + list(args),
                          cwd=str(repo), env=env, timeout=timeout_s,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)


def marker(repo_or_bundle: Path | str) -> dict:
    p = Path(repo_or_bundle) / BUNDLE_MARKER
    # assembled repos do not carry the marker; read it from the bundle mount
    if not p.is_file():
        p = find_bundle_root() / BUNDLE_MARKER
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    info = assemble()
    print(json.dumps(info, indent=2))
