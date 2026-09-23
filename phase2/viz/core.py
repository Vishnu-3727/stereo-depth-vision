"""Stereo Depth Visualizer V1 -- the numerical half.

Everything here is observation only. It loads the frozen H1-v2 checkpoints,
runs them through Phase 1's own preprocessing, and turns their output into the
quantities a researcher needs to look at: disparity, metric depth, error,
correspondence, region statistics. It writes nothing into ``experiments/``,
``results/training/`` or any Phase 1 file.

Conventions, all verified against this repository rather than assumed:

- Preprocessing is ``src/datasets/kitti2015``: pad/crop to 368x1232 anchored at
  the top-left, then ImageNet normalisation on 0-255 RGB, NCHW. This is exactly
  what ``phase2/scripts/exp_h1_cost_volume.py::validate`` feeds the model, so
  the displayed prediction is the evaluated prediction.
- Ground truth is ``disp_occ_0`` divided by 256.0 (the KITTI convention, which
  is the dataset default the H1 runs used). A stored zero means *no ground
  truth*, never a disparity of zero.
- Model output is a single-channel map at full crop resolution, already in
  disparity pixels -- the refinement stage supplies the scale, and the H1 runs
  compare it directly against pixel ground truth.
- The disparity convention is left-referenced: ``x_right = x_left - d``. This is
  not assumed; ``phase2/tests/test_visualizer.py`` re-derives it photometrically
  from the data.
- The crop is anchored at the top-left and nothing is resized, so the KITTI
  rectified intrinsics (f, cx, cy) apply to cropped coordinates unchanged. If
  the preprocessing ever starts resizing, ``Scene.calibration_note`` must change
  with it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_ROOT = REPO_ROOT / "phase2"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.geometry.stereo import StereoCalibration, parse_kitti_cam_to_cam  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

# The two H1-v2 arms, with the config each was trained under. The shift is not
# stored in the state dict, so loading a checkpoint under the wrong shift would
# silently produce a different model -- hence the pairing lives here, and is
# cross-checked against the config saved beside the weights.
MODELS = {
    "BASE": {"experiment": "EXP-H1-BASE-v2", "cost_volume_shift": "none"},
    "WORKING": {"experiment": "EXP-H1-WORKING-v2", "cost_volume_shift": "left"},
    # EXP-H2 keeps WORKING's cost volume and swaps the regression stage for the
    # standardising one; the swap holds no weights, so the state dict is
    # interchangeable and loading it under the wrong regression would silently
    # be a different model -- hence the pairing lives here too.
    "H2": {"experiment": "EXP-H2-SOFTARGMIN-SCALE", "cost_volume_shift": "left",
           "regression": "standardised"},
    # EXP-H3-VIABILITY-001: H2's architecture with the disparity shift removed,
    # trained for 10 epochs only as a viability gate -- not a converged model.
    "H3": {"experiment": "EXP-H3-VIABILITY-001", "cost_volume_shift": "none",
           "regression": "standardised"},
    # EXP-H2-SEED-REPLICATION-001: the H2 recipe at other seeds. Same
    # architecture and same regression stage; only the seed differs.
    "SEED1": {"experiment": "EXP-H2-SEED-REPLICATION-001-SEED1-RUN2",
              "cost_volume_shift": "left", "regression": "standardised"},
    "SEED2": {"experiment": "EXP-H2-SEED-REPLICATION-001-SEED2-RUN2",
              "cost_volume_shift": "left", "regression": "standardised"},
    # EXP-ARMP-SEED1: the ARM-P recipe at seed 1. Unlike the H1/H2 arms this
    # carries an explicit first-class StereoNetConfig (downsample_levels,
    # num_disparities, regression_normalize live in src/models/stereonet, not
    # in the phase2 standardised-regression monkeypatch), so the loader builds
    # the model from "config" below. No "regression" key: the monkeypatch
    # must NOT be applied to this arm. The checkpoint is a byte copy of the
    # frozen stage_b_armp p2a_best.pth, renamed to the _checkpoint.pth
    # convention so checkpoint_path() keeps working.
    "ARMP_S1": {"experiment": "EXP-ARMP-SEED1", "cost_volume_shift": "right",
                "config": {"downsample_levels": 3, "num_disparities": 24,
                           "cost_volume_shift": "right",
                           "regression_normalize": True}},
    # E3 seed-0 final: the Stage-E E3 recipe (EXP-P2A-SCALE-COVERAGE-001, seed
    # 0, 400 epochs) at the same ARM-P architecture as ARMP_S1 above, hence
    # the same explicit config. The checkpoint is a byte copy of the frozen
    # stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth (sha256
    # 82e58bc441a4...ea79c6d), renamed to the _checkpoint.pth convention so
    # checkpoint_path() keeps working. Scored at 40-scene contract EPE
    # 1.1628882757801255 through the Stage-C runtime path (R4).
    "E3": {"experiment": "EXP-E3-SEED0-FINAL", "cost_volume_shift": "right",
           "config": {"downsample_levels": 3, "num_disparities": 24,
                      "cost_volume_shift": "right",
                      "regression_normalize": True}},
}
# The model the mentor demo and the desktop shortcut show. THIS IS THE ONE
# KNOB: when a better model supersedes the current demo model, add its
# checkpoint to MODELS above and repoint this at the new key. Nothing else --
# not the demo builder, not the desktop shortcut, not the launcher -- names a
# model directly.
DEMO_MODEL = "E3"

DATA_ROOT = REPO_ROOT / "data" / "kitti2015"
CHECKPOINT_DIR = PHASE2_ROOT / "results" / "training"
EXPERIMENTS_DIR = PHASE2_ROOT / "experiments"

# Below this, disparity carries no usable depth: fB/d explodes and the value is
# undefined, not merely large.
MIN_DISPARITY_PX = 1e-3


def fmt(value: float | None, unit: str = "", digits: int = 2) -> str:
    """Format a number for display, or ``N/A`` when missing or undefined.

    Nothing non-finite is ever shown as a number -- an invalid depth must not
    look like a measurement.
    """
    if value is None:
        return "N/A"
    v = float(value)
    if not np.isfinite(v):
        return "N/A"
    return "{:.{d}f}{}".format(v, (" " + unit) if unit else "", d=digits)


@dataclass
class Scene:
    """One validation stereo pair, in model-input coordinates."""

    name: str
    index: int
    split: str
    left: np.ndarray            # H x W x 3 uint8, cropped
    right: np.ndarray           # H x W x 3 uint8, cropped
    gt_disparity: np.ndarray    # H x W float32, 0 = no ground truth
    disparity_scale: float
    original_shape: tuple[int, int]
    calibration: StereoCalibration | None
    calibration_error: str | None = None

    @property
    def gt_valid(self) -> np.ndarray:
        return self.gt_disparity > 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.gt_disparity.shape

    @property
    def calibration_note(self) -> str:
        if self.calibration is None:
            return "Metric depth unavailable: calibration missing ({})".format(
                self.calibration_error
            )
        c = self.calibration
        return (
            "f={:.3f} px  B={:.4f} m  fB={:.2f} px*m  (cx={:.1f}, cy={:.1f}); "
            "top-left crop, no resize, so the rectified intrinsics apply "
            "unchanged".format(c.focal_px, c.baseline_m, c.fB, c.cx, c.cy)
        )

    def gt_depth(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Ground-truth depth, valid only where ground-truth disparity exists."""
        depth, ok = depth_from_disparity(self.gt_disparity, self.calibration)
        if depth is None:
            return None, None
        ok = ok & self.gt_valid
        return np.where(ok, depth, np.nan), ok


def load_scene(
    selector: int | str, split: str = "hailo_val", disparity_scale: float = 256.0
) -> Scene:
    """Load one scene by index within the split, or by file name."""
    ds = Kitti2015Stereo(DATA_ROOT, split=split, disparity_scale=disparity_scale)
    if isinstance(selector, str):
        name = selector if selector.endswith(".png") else selector + "_10.png"
        if name not in ds.names:
            raise KeyError("{} is not in split '{}'".format(name, split))
        index = ds.names.index(name)
    else:
        index = int(selector)
        if not 0 <= index < len(ds):
            raise IndexError(
                "scene index {} outside split '{}' (0..{})".format(
                    index, split, len(ds) - 1
                )
            )
    sample = ds[index]

    calib, calib_error = None, None
    path = ds.calibration_path(sample.name)
    try:
        calib = parse_kitti_cam_to_cam(path)
    except Exception as exc:  # missing file, malformed file, unrectified pair
        calib_error = "{}: {}".format(type(exc).__name__, exc)

    return Scene(
        name=sample.name,
        index=index,
        split=split,
        left=sample.left,
        right=sample.right,
        gt_disparity=sample.disparity,
        disparity_scale=disparity_scale,
        original_shape=sample.original_shape,
        calibration=calib,
        calibration_error=calib_error,
    )


def scene_names(split: str = "hailo_val") -> list[str]:
    return Kitti2015Stereo(DATA_ROOT, split=split).names


def checkpoint_path(model_key: str) -> Path:
    return CHECKPOINT_DIR / (MODELS[model_key]["experiment"] + "_checkpoint.pth")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Prediction:
    model_key: str
    disparity: np.ndarray        # H x W float32, pixels
    checkpoint: Path
    checkpoint_sha256: str
    config: dict
    device: str


class ModelRunner:
    """Loads each checkpoint once and caches predictions per (model, scene).

    Checkpoints are opened read-only and the models stay in eval mode; this
    class never writes weights anywhere.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, tuple[StereoNet, dict, str]] = {}
        self._cache: dict[tuple[str, str], Prediction] = {}

    def _model(self, model_key: str):
        if model_key not in self._models:
            if model_key not in MODELS:
                raise KeyError(
                    "unknown model '{}'; known: {}".format(model_key, ", ".join(MODELS))
                )
            path = checkpoint_path(model_key)
            if not path.exists():
                raise FileNotFoundError("checkpoint missing: " + str(path))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            spec = MODELS[model_key]
            shift = spec["cost_volume_shift"]
            recorded = ckpt.get("config", {}).get("cost_volume_shift")
            if recorded is not None and recorded != shift:
                raise ValueError(
                    "checkpoint {} was trained with cost_volume_shift='{}' but "
                    "{} expects '{}'".format(path.name, recorded, model_key, shift)
                )
            cfg_kwargs = dict(spec.get("config", {}))
            cfg_kwargs.setdefault("cost_volume_shift", shift)
            if cfg_kwargs["cost_volume_shift"] != shift:
                raise ValueError(
                    "{} registry entry is inconsistent: cost_volume_shift='{}' "
                    "but config says '{}'".format(
                        model_key, shift, cfg_kwargs["cost_volume_shift"])
                )
            model = StereoNet(StereoNetConfig(**cfg_kwargs))
            if MODELS[model_key].get("regression") == "standardised":
                from phase2.models import scaled_regression
                scaled_regression.apply_to(model)
            model.load_state_dict(ckpt["model"])
            model.eval().to(self.device)
            self._models[model_key] = (model, ckpt.get("config", {}), sha256(path))
        return self._models[model_key]

    def predict(self, model_key: str, scene: Scene) -> Prediction:
        key = (model_key, scene.name)
        if key in self._cache:
            return self._cache[key]
        model, config, digest = self._model(model_key)
        with torch.no_grad():
            out = model(
                torch.from_numpy(normalize(scene.left)).to(self.device),
                torch.from_numpy(normalize(scene.right)).to(self.device),
            )
        pred = Prediction(
            model_key=model_key,
            disparity=out[0, 0].cpu().numpy().astype(np.float32),
            checkpoint=checkpoint_path(model_key),
            checkpoint_sha256=digest,
            config=config,
            device=self.device,
        )
        self._cache[key] = pred
        return pred


def depth_from_disparity(
    disparity: np.ndarray, calib: StereoCalibration | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Z = fB/d, with invalid disparity reported as invalid, never clamped.

    The finiteness guard is this module's, not the geometry module's: Phase 1's
    ``depth_from_disparity`` accepts ``+inf`` as a valid disparity and returns a
    depth of exactly 0 m, which would display as a real measurement. The frozen
    baseline is not edited to fix that; it is guarded here.
    """
    if calib is None:
        return None, None
    disparity = np.asarray(disparity, dtype=np.float64)
    depth, ok = calib.depth_from_disparity(disparity, min_disparity=MIN_DISPARITY_PX)
    ok = ok & np.isfinite(disparity)
    return np.where(ok, depth, np.nan), ok


def scene_metrics(pred: np.ndarray, scene: Scene) -> dict:
    """Scene-level disparity metrics, using the project's pooled protocol.

    The same rule as ``src/evaluation/metrics.disparity_metrics`` as called by
    the H1 validation loop: valid = gt > 0, pooled over pixels. One scene, so
    these are not comparable to the recorded 10-scene validation figures.
    """
    valid = scene.gt_valid
    if not valid.any():
        return {"valid_pixels": 0}
    return disparity_metrics(pred[valid], scene.gt_disparity[valid]).as_dict()


def correspondence(x_left: float, disparity: float) -> float:
    """Right-image column matching a left-image column, left-referenced stereo.

        x_right = x_left - d

    Confirmed photometrically against KITTI ground truth in the test suite; the
    opposite sign is roughly five times worse on real data.
    """
    return float(x_left) - float(disparity)


def probe_pixel(scene: Scene, predictions: dict[str, Prediction], x: int, y: int) -> dict:
    """Everything known about one pixel. Missing quantities are ``None``."""
    h, w = scene.shape
    if not (0 <= x < w and 0 <= y < h):
        raise IndexError("pixel ({}, {}) outside {}x{}".format(x, y, w, h))

    calib = scene.calibration
    gt_d = float(scene.gt_disparity[y, x])
    gt_valid = gt_d > 0
    gt_depth = float(calib.fB / gt_d) if (gt_valid and calib is not None) else None

    out: dict = {
        "pixel": [int(x), int(y)],
        "gt_disparity": gt_d if gt_valid else None,
        "gt_depth_m": gt_depth,
        "gt_x_right": correspondence(x, gt_d) if gt_valid else None,
        "models": {},
    }
    for key, pred in predictions.items():
        d = float(pred.disparity[y, x])
        finite = bool(np.isfinite(d))
        usable = finite and d > MIN_DISPARITY_PX
        depth = float(calib.fB / d) if (usable and calib is not None) else None
        out["models"][key] = {
            "disparity": d if finite else None,
            "disparity_error": abs(d - gt_d) if (gt_valid and finite) else None,
            "depth_m": depth,
            "depth_error_m": (
                abs(depth - gt_depth)
                if (depth is not None and gt_depth is not None)
                else None
            ),
            "x_right": correspondence(x, d) if finite else None,
        }
    return out


def region_stats(
    scene: Scene, pred: Prediction, x1: int, y1: int, x2: int, y2: int
) -> dict:
    """Robust depth/disparity statistics over a rectangle.

    Medians are reported alongside means because the depth distribution inside a
    box straddling a depth boundary is bimodal, and the mean then names a
    distance nothing in the box is at.
    """
    h, w = scene.shape
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)
    if x2 < x1 or y2 < y1:
        return {"region": [x1, y1, x2, y2], "pixels": 0, "note": "empty region"}

    sl = (slice(y1, y2 + 1), slice(x1, x2 + 1))
    d = pred.disparity[sl].astype(np.float64)
    gt = scene.gt_disparity[sl].astype(np.float64)
    gt_ok = gt > 0
    d_ok = np.isfinite(d) & (d > MIN_DISPARITY_PX)

    stats: dict = {
        "region": [x1, y1, x2, y2],
        "model": pred.model_key,
        "pixels": int(d.size),
        "pred_disparity_valid_pixels": int(d_ok.sum()),
        "gt_valid_pixels": int(gt_ok.sum()),
    }
    if d_ok.any():
        v = d[d_ok]
        stats["pred_disparity_median"] = float(np.median(v))
        stats["pred_disparity_mean"] = float(v.mean())
        if scene.calibration is not None:
            z = scene.calibration.fB / v
            stats["pred_depth_median_m"] = float(np.median(z))
            stats["pred_depth_mean_m"] = float(z.mean())
            stats["pred_depth_min_m"] = float(z.min())
            stats["pred_depth_max_m"] = float(z.max())
            stats["pred_depth_p10_m"] = float(np.percentile(z, 10))
            stats["pred_depth_p90_m"] = float(np.percentile(z, 90))
    if gt_ok.any():
        err = np.abs(d[gt_ok] - gt[gt_ok])
        stats["disparity_mae"] = float(err.mean())
        stats["disparity_median_ae"] = float(np.median(err))
        stats["d1_percent"] = float(
            100.0 * ((err > 3.0) & (err > 0.05 * gt[gt_ok])).mean()
        )
        if scene.calibration is not None:
            both = gt_ok & d_ok
            if both.any():
                fB = scene.calibration.fB
                dz = np.abs(fB / d[both] - fB / gt[both])
                stats["depth_mae_m"] = float(dz.mean())
                stats["depth_median_ae_m"] = float(np.median(dz))
    return stats


def point_cloud(
    disparity: np.ndarray,
    scene: Scene,
    stride: int = 4,
    max_depth_m: float = 80.0,
) -> np.ndarray | None:
    """Back-project disparity to camera-frame XYZ, using the real intrinsics.

    Returns an ``N x 6`` array (X, Y, Z, R, G, B) or ``None`` when calibration
    is unavailable. Points beyond ``max_depth_m`` are dropped rather than drawn
    at an arbitrary distance -- past ~1 px of disparity the depth is noise.
    """
    calib = scene.calibration
    if calib is None:
        return None
    d = disparity[::stride, ::stride].astype(np.float64)
    rgb = scene.left[::stride, ::stride].astype(np.float64) / 255.0
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs * stride
    ys = ys * stride
    ok = np.isfinite(d) & (d > MIN_DISPARITY_PX)
    z = np.full(d.shape, np.nan)
    np.divide(calib.fB, d, out=z, where=ok)
    ok &= np.isfinite(z) & (z > 0) & (z <= max_depth_m)
    if not ok.any():
        return None
    z = z[ok]
    x = (xs[ok] - calib.cx) * z / calib.focal_px
    y = (ys[ok] - calib.cy) * z / calib.focal_px
    return np.column_stack([x, y, z, rgb[ok]])


def recorded_global_metrics(model_key: str) -> dict:
    """The already-recorded validation figures for an H1-v2 arm.

    Read from ``phase2/experiments/<id>/metrics.json``, not recomputed -- the
    recorded protocol (first 10 scenes of ``hailo_val``, pooled over gt > 0
    pixels, validated every 5 epochs) is part of the experiment record and this
    tool must not silently substitute a different one.
    """
    path = EXPERIMENTS_DIR / MODELS[model_key]["experiment"] / "metrics.json"
    if not path.exists():
        return {"source": str(path), "available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", data)
    series = metrics.get("validation_epe_series") or []
    final = series[-1] if series else {}
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "available": True,
        "protocol": (
            "recorded during training: first 10 scenes of hailo_val, full "
            "368x1232 frames, pooled over gt > 0 pixels "
            "(exp_h1_cost_volume.py::validate)"
        ),
        "final_epoch": final.get("epoch"),
        "val_epe": final.get("val_epe"),
        "val_d1": final.get("val_d1"),
        "best_epe": metrics.get("validation_epe_best"),
        "best_epe_epoch": metrics.get("validation_epe_best_epoch"),
    }


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def metadata(scene: Scene, predictions: dict[str, Prediction]) -> dict:
    """Everything needed to reproduce a visualization later."""
    return {
        "tool": "phase2 stereo depth visualizer V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "scene": {
            "name": scene.name,
            "index": scene.index,
            "split": scene.split,
            "shape": list(scene.shape),
            "original_shape": list(scene.original_shape),
            "gt_valid_pixels": int(scene.gt_valid.sum()),
        },
        "preprocessing": {
            "source": "src/datasets/kitti2015",
            "crop": "pad bottom/right to >= 368x1232, then crop top-left anchored",
            "resize": "none",
            "normalisation": "ImageNet mean/std on 0-255 RGB, NCHW",
            "ground_truth": "disp_occ_0 / {} (0 = no ground truth)".format(
                scene.disparity_scale
            ),
        },
        "calibration": (
            {
                "source": scene.calibration.source,
                "focal_px": scene.calibration.focal_px,
                "baseline_m": scene.calibration.baseline_m,
                "cx": scene.calibration.cx,
                "cy": scene.calibration.cy,
                "note": (
                    "applies to cropped coordinates unchanged "
                    "(top-left crop, no resize)"
                ),
            }
            if scene.calibration is not None
            else {"available": False, "error": scene.calibration_error}
        ),
        "disparity_convention": "left-referenced: x_right = x_left - d",
        "models": {
            key: {
                "experiment": MODELS[key]["experiment"],
                "cost_volume_shift": MODELS[key]["cost_volume_shift"],
                "checkpoint": str(p.checkpoint.relative_to(REPO_ROOT)),
                "checkpoint_sha256": p.checkpoint_sha256,
                "device": p.device,
                "precision": "fp32",
                "input_size": list(scene.shape),
                "training_config": p.config,
            }
            for key, p in predictions.items()
        },
        "scene_metrics": {
            key: scene_metrics(p.disparity, scene) for key, p in predictions.items()
        },
        "recorded_global_metrics": {
            key: recorded_global_metrics(key) for key in predictions
        },
    }


def demo() -> None:
    """Self-check on synthetic values: geometry, invalid handling, formatting."""
    calib = StereoCalibration(
        focal_px=721.5, baseline_m=0.533, cx=600.0, cy=180.0,
        width=1242, height=375, source="synthetic",
    )
    d = np.array([[40.0, 0.0, -1.0, 1e-9, np.inf, np.nan]])
    depth, ok = depth_from_disparity(d, calib)
    assert abs(depth[0, 0] - 721.5 * 0.533 / 40.0) < 1e-9, depth[0, 0]
    assert ok.tolist() == [[True, False, False, False, False, False]]
    assert np.isnan(depth[0, 1:]).all()
    assert depth_from_disparity(d, None) == (None, None)

    assert correspondence(742, 41.8) == 742 - 41.8
    assert fmt(None) == "N/A" and fmt(np.nan) == "N/A" and fmt(np.inf) == "N/A"
    assert fmt(9.1837, "m") == "9.18 m"
    print("visualizer core self-check passed")


if __name__ == "__main__":
    demo()
