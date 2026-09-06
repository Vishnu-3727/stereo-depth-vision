"""Experiment bookkeeping for Phase 1.

Every measurement in this project must be traceable to an experiment directory.
This module is the single place that creates one. It allocates the next free
EXP-XXX id, captures the environment automatically, and refuses to touch a
directory that already exists -- results are never overwritten, and failed runs
are kept exactly as they were.

Typical use:

    from src.common.experiment import Experiment

    with Experiment("kitti2015 baseline eval", config=cfg) as exp:
        ...
        exp.metric("EPE", 8.41)
        exp.note("Protocol: all valid GT pixels, no occlusion mask.")
        exp.conclude("Closest reproduction of the Hailo 8.223 figure so far.")

On exit the directory holds env.json, config.json, metrics.json and log.txt.
If the body raised, status is "failed" and the traceback is stored -- the run
still counts and is still kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Fields the Phase 1 charter requires on every experiment. Anything not filled
# in automatically must come from the config, so a missing value stays visible
# in env.json rather than quietly disappearing.
REQUIRED_CONFIG_FIELDS = (
    "dataset",
    "split",
    "resolution",
    "crop",
    "disparity_range",
    "batch_size",
    "precision",
    "seed",
)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state() -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        # A dirty tree means the commit alone does not identify the code that
        # produced the result. Recording that is the honest option.
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "onnx",
        "onnxruntime",
        "scipy",
        "PIL",
        "skimage",
        "thop",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def hardware() -> dict[str, Any]:
    hw: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        hw["cuda_available"] = torch.cuda.is_available()
        hw["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw["gpu_name"] = props.name
            hw["gpu_total_memory_bytes"] = props.total_memory
            hw["gpu_capability"] = str(props.major) + "." + str(props.minor)
    except Exception:
        hw["cuda_available"] = None
    driver = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if driver:
        hw["nvidia_driver"] = driver.splitlines()[0].strip()
    return hw


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def next_experiment_id(experiments_dir: Path = EXPERIMENTS_DIR) -> str:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    used = [
        int(p.name[4:7])
        for p in experiments_dir.iterdir()
        if p.is_dir() and p.name.startswith("EXP-") and p.name[4:7].isdigit()
    ]
    return "EXP-{:03d}".format(max(used, default=0) + 1)


class Experiment:
    """One experiment directory. Created once, never reused."""

    def __init__(
        self,
        description: str,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        experiments_dir: Path = EXPERIMENTS_DIR,
    ) -> None:
        self.description = description
        self.config = dict(config or {})
        self.id = experiment_id or next_experiment_id(experiments_dir)
        self.dir = Path(experiments_dir) / self.id
        if self.dir.exists():
            raise FileExistsError(
                str(self.dir)
                + " already exists. Experiment results are never overwritten -- "
                "pick a new id or move the old directory aside."
            )
        self.dir.mkdir(parents=True)
        self.metrics: dict[str, Any] = {}
        self.observations: list[str] = []
        self.conclusion: str | None = None
        self.status = "running"
        self._log = (self.dir / "log.txt").open("w", encoding="utf-8")
        self._started = datetime.now(timezone.utc)
        self._write_env()

    # -- recording -------------------------------------------------------
    def metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value
        self.log("metric " + name + " = " + str(value))

    def note(self, text: str) -> None:
        self.observations.append(text)
        self.log("note: " + text)

    def conclude(self, text: str) -> None:
        self.conclusion = text
        self.log("conclusion: " + text)

    def log(self, text: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = "[" + stamp + "] " + text
        print(self.id + " " + line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def path(self, name: str) -> Path:
        """A path inside this experiment's directory, for artifacts."""
        return self.dir / name

    # -- lifecycle -------------------------------------------------------
    def _write_env(self) -> None:
        missing = [f for f in REQUIRED_CONFIG_FIELDS if f not in self.config]
        env = {
            "experiment_id": self.id,
            "description": self.description,
            "date_utc": self._started.isoformat(),
            "git": git_state(),
            "config": self.config,
            "config_hash": config_hash(self.config),
            "config_fields_missing": missing,
            "command": " ".join(sys.argv),
            "cwd": str(Path.cwd()),
            "hardware": hardware(),
            "software": software_versions(),
        }
        (self.dir / "env.json").write_text(
            json.dumps(env, indent=2, default=str), encoding="utf-8"
        )
        if self.config:
            (self.dir / "config.json").write_text(
                json.dumps(self.config, indent=2, default=str), encoding="utf-8"
            )
        if missing:
            self.log("WARNING config fields not supplied: " + ", ".join(missing))

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.status = status
        finished = datetime.now(timezone.utc)
        payload = {
            "experiment_id": self.id,
            "description": self.description,
            "status": status,
            "started_utc": self._started.isoformat(),
            "finished_utc": finished.isoformat(),
            "duration_s": (finished - self._started).total_seconds(),
            "metrics": self.metrics,
            "observations": self.observations,
            "conclusion": self.conclusion,
            "error": error,
        }
        (self.dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        self.log("status: " + status)
        self._log.close()

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.finish(
                "failed", "".join(traceback.format_exception(exc_type, exc, tb))
            )
        else:
            self.finish("completed")
        return False  # never swallow the exception


def demo() -> None:
    """Self-check: an experiment records what it should and refuses reuse."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = {
            "dataset": "kitti2015",
            "split": "val",
            "resolution": [368, 1232],
            "crop": None,
            "disparity_range": 192,
            "batch_size": 1,
            "precision": "fp32",
            "seed": 0,
        }
        with Experiment("self check", config=cfg, experiments_dir=tmp) as exp:
            exp.metric("EPE", 1.23)
            exp.note("nothing unusual")
            exp.conclude("helper works")
            first = exp.dir

        env = json.loads((first / "env.json").read_text())
        met = json.loads((first / "metrics.json").read_text())
        assert env["experiment_id"] == "EXP-001"
        assert env["config_fields_missing"] == [], env["config_fields_missing"]
        assert env["hardware"]["cpu_count"], "hardware not captured"
        assert env["software"]["python"], "software versions not captured"
        assert len(env["config_hash"]) == 16
        assert met["metrics"]["EPE"] == 1.23
        assert met["status"] == "completed"
        assert met["conclusion"] == "helper works"

        # ids advance, directories are never reused
        second = Experiment("second", config=cfg, experiments_dir=tmp)
        assert second.id == "EXP-002"
        second.finish()
        try:
            Experiment("clash", experiment_id="EXP-001", experiments_dir=tmp)
        except FileExistsError:
            pass
        else:
            raise AssertionError("reused an existing experiment directory")

        # a failing body is still recorded, as a failure
        try:
            with Experiment("boom", config=cfg, experiments_dir=tmp):
                raise ValueError("intentional")
        except ValueError:
            pass
        failed = json.loads((tmp / "EXP-003" / "metrics.json").read_text())
        assert failed["status"] == "failed"
        assert "intentional" in failed["error"]

        print("experiment helper self-check passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    demo()
