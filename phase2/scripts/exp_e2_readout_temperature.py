"""EXP-E2-READOUT-TEMPERATURE-001 -- is H2's error sensitive to readout sharpness?

Inference only. No training, no weight change, no change to the cost volume,
aggregation, refinement, the standardisation, the disparity convention, the
scenes or the scoring. A temperature is introduced at exactly one place -- the
softmax the soft-argmin consumes -- and nothing else moves.

    python phase2/scripts/exp_e2_readout_temperature.py run

Grid, bands, gates and the verdict rule are frozen in
`phase2/docs/EXP_E2_READOUT_TEMPERATURE_001_PREREGISTRATION.md`, written before
this ran. Read that first; this file only executes it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.models.scaled_regression import (  # noqa: E402
    StandardisedDisparityRegression, standardise_across_disparity,
)
from phase2.scripts.exp_e1_error_partition import MODELS, Pool, SCENES  # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (  # noqa: E402
    EXPERIMENTS_DIR, FOCUS_SCENES, GATE_MIN_STEREO_D1_POINTS, stereo_probe,
)
from phase2.scripts.h1_mechanism_probe import (  # noqa: E402
    _finish, _forward_parts, load_model,
)
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402
from src.models.stereonet.regression import soft_argmin  # noqa: E402  (frozen)

EXPERIMENT_ID = "EXP-E2-READOUT-TEMPERATURE-001"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / "EXP-E2-READOUT-TEMPERATURE-001"
E1_RESULTS = REPO_ROOT / "phase2" / "results" / "EXP-E1-ERROR-PARTITION-001" / "partition.json"

# Frozen grid (pre-registration section 4).
TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
CONTROL_T = 1.0
# Frozen bands (section 7), reused verbatim from E3.
BAND_D1 = 1.0
BAND_EPE = 0.10
IN_PROCESS_TOLERANCE = 1e-9      # gate 1(a): the intervention itself
RECORD_TOLERANCE = 1e-6          # gate 1(b): continuity with the record
EXPECTED_PARAMETERS = 423586
RIGHT_KEYS = ("right_black", "right_noise", "right_equals_left")
MAP_KEYS = ("initial_constant_mean", "initial_shuffled")


class TemperedStandardisedDisparityRegression(StandardisedDisparityRegression):
    """H2's readout with a temperature on the softmax, and nothing else.

    `standardise_across_disparity` and Phase 1's `soft_argmin` are imported and
    called unchanged; the only new arithmetic is the division by T, which at
    T = 1.0 is exact in IEEE-754 and so must reproduce H2 bit-for-bit.
    """

    def __init__(self, temperature: float, upsample_first: bool = True,
                 eps: float = StandardisedDisparityRegression().eps) -> None:
        super().__init__(upsample_first=upsample_first, eps=eps)
        self.temperature = float(temperature)

    def forward(self, cost: torch.Tensor, size: tuple) -> torch.Tensor:
        import torch.nn.functional as F
        if self.upsample_first:
            cost = F.interpolate(cost, size=size, mode="bilinear", align_corners=True)
            scaled = standardise_across_disparity(cost, dim=1, eps=self.eps)
            self._capture(cost, scaled)
            return soft_argmin(scaled / self.temperature, dim=1)
        scaled = standardise_across_disparity(cost, dim=1, eps=self.eps)
        self._capture(cost, scaled)
        disparity = soft_argmin(scaled / self.temperature, dim=1)
        return F.interpolate(disparity, size=size, mode="bilinear", align_corners=True)

    def extra_repr(self) -> str:
        return "temperature={}, {}".format(self.temperature, super().extra_repr())


def set_temperature(model, temperature: float):
    """Swap only the readout module. The regression stage holds no weights."""
    old = model.regression
    assert isinstance(old, StandardisedDisparityRegression), type(old)
    model.regression = TemperedStandardisedDisparityRegression(
        temperature, upsample_first=old.upsample_first, eps=old.eps)
    return model


class ReadoutStats:
    """Distribution statistics of the tensor the softmax actually consumes."""

    def __init__(self) -> None:
        self.entropy = []
        self.peak_mean = []
        self.peak_max = []
        self.top2_gap = []
        self.disparity_mean = []
        self.disparity_std = []
        self.pixels = []

    def add(self, softmax_input: torch.Tensor, temperature: float,
            disparity: np.ndarray) -> None:
        logits = -softmax_input.double() / temperature
        p = torch.softmax(logits, dim=1)
        entropy = -(p * p.clamp_min(1e-300).log()).sum(dim=1)
        top2 = p.topk(2, dim=1).values
        self.entropy.append(entropy.flatten().cpu().numpy())
        self.peak_mean.append(float(top2[:, 0].mean()))
        self.peak_max.append(float(top2[:, 0].max()))
        self.top2_gap.append(float((top2[:, 0] - top2[:, 1]).mean()))
        self.disparity_mean.append(float(disparity.mean()))
        self.disparity_std.append(float(disparity.std()))
        self.pixels.append(int(entropy.numel()))

    def result(self) -> dict:
        entropy = np.concatenate(self.entropy)
        return {
            "mean_entropy_nats": float(entropy.mean()),
            "median_entropy_nats": float(np.median(entropy)),
            "max_entropy_possible_nats": float(np.log(12)),
            "mean_peak_probability": float(np.mean(self.peak_mean)),
            "max_peak_probability": float(np.max(self.peak_max)),
            "mean_top2_gap": float(np.mean(self.top2_gap)),
            "disparity_initial_mean_candidates": float(np.mean(self.disparity_mean)),
            "disparity_initial_std_candidates": float(np.mean(self.disparity_std)),
            "pixels": int(np.sum(self.pixels)),
        }


def score_at(model, scenes, device: str) -> dict:
    """Pooled accuracy and readout statistics in one pass."""
    pool, stats = Pool(), ReadoutStats()
    finite = True
    model.regression.capture = True
    for scene in scenes:
        left, initial = _forward_parts(model, scene, device)
        captured = model.regression.last["softmax_input"]
        pred = _finish(model, left, initial)
        if not np.all(np.isfinite(pred)):
            finite = False
        pool.add(pred, scene.gt_disparity.astype(np.float64), scene.gt_valid)
        stats.add(captured, model.regression.temperature,
                  initial[0, 0].cpu().numpy().astype(np.float64))
        model.regression.last = {}
    model.regression.capture = False
    return {**pool.result(), "finite": finite, "readout": stats.result()}


def score_at_untempered(model, scenes, device: str) -> dict:
    """Pooled accuracy with H2's readout exactly as it ships -- gate 1(a)."""
    pool = Pool()
    for scene in scenes:
        left, initial = _forward_parts(model, scene, device)
        pool.add(_finish(model, left, initial),
                 scene.gt_disparity.astype(np.float64), scene.gt_valid)
    return pool.result()


def stereo_at(model, probe_scenes, device: str) -> dict:
    rows = stereo_probe(model, probe_scenes, device, np.random.default_rng(0))
    right = min(row[k]["d1_penalty"] for row in rows for k in RIGHT_KEYS)
    mapping = min(row[k]["d1_penalty"] for row in rows for k in MAP_KEYS)
    return {"right_dependence_worst_pt": right,
            "matching_map_dependence_worst_pt": mapping,
            "meets_c2": right >= GATE_MIN_STEREO_D1_POINTS,
            "meets_c3": mapping >= GATE_MIN_STEREO_D1_POINTS,
            "softmax_entropy_probe": float(np.mean(
                [row["softmax_entropy"] for row in rows]))}


def macs_at(temperature: float, device: str) -> float:
    from thop import profile

    from phase2.models import scaled_regression
    from src.models.stereonet import StereoNet, StereoNetConfig
    fresh = StereoNet(StereoNetConfig(cost_volume_shift="left"))
    scaled_regression.apply_to(fresh)
    set_temperature(fresh, temperature)
    fresh = fresh.to(device).eval()
    left = torch.randn(1, 3, 368, 1232, device=device)
    right = torch.randn(1, 3, 368, 1232, device=device)
    macs, _ = profile(fresh, inputs=(left, right), verbose=False)
    return float(macs)


def band_for(delta_epe: float, delta_d1: float) -> str:
    if abs(delta_d1) <= BAND_D1 and abs(delta_epe) <= BAND_EPE:
        return "NEGLIGIBLE"
    if delta_d1 <= -BAND_D1 and delta_epe <= -BAND_EPE:
        return "MATERIAL IMPROVEMENT"
    return "MATERIAL DEGRADATION"


def phase_1_diff() -> str:
    try:
        return subprocess.run(
            ["git", "diff", "--name-only", "phase-1-frozen", "--", "src", "scripts"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:                       # pragma: no cover
        return "unavailable: {}".format(exc)


def table(scores: dict, stereo: dict) -> str:
    lines = ["| seed | T | EPE (px) | dEPE | D1 (%) | dD1 | band | entropy | "
             "peak p | right-dep | map-dep | C2/C3 |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for key in MODELS:
        base = scores[key][str(CONTROL_T)]
        for t in TEMPERATURES:
            row, st = scores[key][str(t)], stereo[key][str(t)]
            lines.append(
                "| {} | {} | {:.3f} | {:+.3f} | {:.2f} | {:+.2f} | {} | {:.3f} | "
                "{:.3f} | {:+.2f} | {:+.2f} | {} |".format(
                    key, t, row["epe"], row["epe"] - base["epe"], row["d1"],
                    row["d1"] - base["d1"],
                    band_for(row["epe"] - base["epe"], row["d1"] - base["d1"]),
                    row["readout"]["mean_entropy_nats"],
                    row["readout"]["mean_peak_probability"],
                    st["right_dependence_worst_pt"],
                    st["matching_map_dependence_worst_pt"],
                    "ok" if st["meets_c2"] and st["meets_c3"] else "FAIL"))
    return "\n".join(lines)


def run(device: str) -> None:
    diff = phase_1_diff()
    if diff:
        raise SystemExit("Phase 1 has diverged from phase-1-frozen: " + diff)

    config = {
        "experiment": EXPERIMENT_ID,
        "hypothesis": ("H2's readout has suboptimal confidence sharpness: "
                       "changing only the inference-time softmax temperature "
                       "materially improves accuracy without destroying stereo "
                       "functionality."),
        "preregistration": "phase2/docs/EXP_E2_READOUT_TEMPERATURE_001_PREREGISTRATION.md",
        "measurement_only": True,
        "training": "none -- trained H2 checkpoints, inference only",
        "intervention": ("p(d) = softmax(-C_standardised(d) / T) at the frozen "
                         "soft_argmin; standardise_across_disparity and "
                         "soft_argmin imported and called unchanged; "
                         "phase2/models/scaled_regression.py not modified"),
        "models": list(MODELS),
        "temperatures": list(TEMPERATURES),
        "control_temperature": CONTROL_T,
        "bands": {"negligible_d1": BAND_D1, "negligible_epe": BAND_EPE},
        "stereo_constraints": "C2 and C3: both dependencies >= {} D1 points".format(
            GATE_MIN_STEREO_D1_POINTS),
        "dataset": "kitti2015",
        "split": "hailo_val, all 40 scenes, pooled over gt > 0",
        "resolution": "368x1232 full frames",
        "crop": None,
        "disparity_range": 12,
        "batch_size": 1,
        "precision": "fp32",
        "seed": [0, 1, 2],
        "phase_1_diff_vs_frozen": diff or "(empty)",
        "known_limitation": ("inference-only: a network adapts to the readout it "
                             "was trained under, so a negative result is weak "
                             "evidence about temperature as a design variable. "
                             "Three seeds, 40 scenes, four probe scenes, one "
                             "budget; no significance test and none claimed."),
    }
    if (EXPERIMENTS_DIR / EXPERIMENT_ID).exists():
        raise SystemExit(EXPERIMENT_ID + " already exists -- records are never overwritten.")

    e1 = json.loads(E1_RESULTS.read_text(encoding="utf-8"))
    scenes = [core.load_scene(i) for i in SCENES]
    probe_scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    hashes_before = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with Experiment("E2: is H2's error sensitive to readout temperature?",
                    config=config, experiment_id=EXPERIMENT_ID,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])
        exp.note("Phase 1 diff vs phase-1-frozen verified empty before starting.")

        t0 = time.time()
        macs = {str(t): macs_at(t, device) for t in (CONTROL_T, TEMPERATURES[0])}
        exp.log("MACs @368x1232: T=1.0 {:.4f} G, T={} {:.4f} G".format(
            macs[str(CONTROL_T)] / 1e9, TEMPERATURES[0],
            macs[str(TEMPERATURES[0])] / 1e9))

        scores, stereo, parameters, untempered = {}, {}, {}, {}
        for key in MODELS:
            base_model = load_model(key, device)
            parameters[key] = base_model.parameter_count()
            # Gate 1(a)'s reference: H2's own readout, unmodified, this process.
            untempered[key] = score_at_untempered(base_model, scenes, device)
            exp.log("{:<6} untempered H2 readout  EPE {:7.3f}  D1 {:6.2f} %".format(
                key, untempered[key]["epe"], untempered[key]["d1"]))
            scores[key], stereo[key] = {}, {}
            for t in TEMPERATURES:
                model = set_temperature(base_model, t)
                scores[key][str(t)] = score_at(model, scenes, device)
                stereo[key][str(t)] = stereo_at(model, probe_scenes, device)
                row, st = scores[key][str(t)], stereo[key][str(t)]
                exp.log("{:<6} T={:<5} EPE {:7.3f}  D1 {:6.2f} %  entropy {:.3f}  "
                        "peak {:.3f}  right {:+8.2f}  map {:+8.2f}  {}".format(
                            key, t, row["epe"], row["d1"],
                            row["readout"]["mean_entropy_nats"],
                            row["readout"]["mean_peak_probability"],
                            st["right_dependence_worst_pt"],
                            st["matching_map_dependence_worst_pt"],
                            "C2/C3 ok" if st["meets_c2"] and st["meets_c3"] else
                            "C2/C3 FAIL"))
            del base_model

        hashes_after = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
        gate = {
            "control_matches_untempered_in_process": all(
                abs(scores[k][str(CONTROL_T)]["epe"] - untempered[k]["epe"])
                <= IN_PROCESS_TOLERANCE
                and abs(scores[k][str(CONTROL_T)]["d1"] - untempered[k]["d1"])
                <= IN_PROCESS_TOLERANCE for k in MODELS),
            "control_matches_record": all(
                abs(scores[k][str(CONTROL_T)]["epe"]
                    - e1["scores"][k]["as trained"]["epe"]) <= RECORD_TOLERANCE
                and abs(scores[k][str(CONTROL_T)]["d1"]
                        - e1["scores"][k]["as trained"]["d1"]) <= RECORD_TOLERANCE
                for k in MODELS),
            "all_finite": all(scores[k][str(t)]["finite"]
                              for k in MODELS for t in TEMPERATURES),
            "checkpoints_unchanged": hashes_before == hashes_after,
            "parameter_count_unchanged": all(v == EXPECTED_PARAMETERS
                                             for v in parameters.values()),
            "macs_unchanged": macs[str(CONTROL_T)] == macs[str(TEMPERATURES[0])],
            "phase_1_diff_empty": phase_1_diff() == "",
        }
        gate["PASS"] = all(gate.values())

        per_temperature = {}
        for t in TEMPERATURES:
            deltas = {k: {"epe": scores[k][str(t)]["epe"] - scores[k][str(CONTROL_T)]["epe"],
                          "d1": scores[k][str(t)]["d1"] - scores[k][str(CONTROL_T)]["d1"]}
                      for k in MODELS}
            bands = {k: band_for(d["epe"], d["d1"]) for k, d in deltas.items()}
            stereo_ok = all(stereo[k][str(t)]["meets_c2"] and stereo[k][str(t)]["meets_c3"]
                            for k in MODELS)
            improves_all = all(b == "MATERIAL IMPROVEMENT" for b in bands.values())
            per_temperature[str(t)] = {
                "per_seed_delta": deltas, "per_seed_band": bands,
                "improves_on_all_seeds": improves_all,
                "stereo_ok_on_all_seeds": stereo_ok,
                "viable": improves_all and stereo_ok,
            }

        viable = [t for t in TEMPERATURES if per_temperature[str(t)]["viable"]]
        improving = [t for t in TEMPERATURES
                     if per_temperature[str(t)]["improves_on_all_seeds"]]
        improves_some_seed = [t for t in TEMPERATURES if t != CONTROL_T and any(
            per_temperature[str(t)]["per_seed_band"][k] == "MATERIAL IMPROVEMENT"
            for k in MODELS)]
        degrades_any = [t for t in TEMPERATURES if t != CONTROL_T and any(
            per_temperature[str(t)]["per_seed_band"][k] == "MATERIAL DEGRADATION"
            for k in MODELS)]

        if not gate["PASS"]:
            verdict, case = "INVALID", "a pre-registered gate failed"
        elif viable:
            verdict, case = "READOUT-SENSITIVE", "A"
        elif improving:
            verdict, case = "INCONCLUSIVE", "C (improvement rejected: stereo lost)"
        elif improves_some_seed:
            verdict, case = "INCONCLUSIVE", "D (seeds disagree)"
        else:
            verdict, case = "READOUT-ROBUST", "B"

        exp.metric("untempered_control", untempered)
        exp.metric("scores", scores)
        exp.metric("stereo_dependence", stereo)
        exp.metric("per_temperature", per_temperature)
        exp.metric("gates", gate)
        exp.metric("macs", macs)
        exp.metric("parameter_counts", parameters)
        exp.metric("checkpoint_integrity", {"before": hashes_before, "after": hashes_after})
        exp.metric("viable_temperatures", viable)
        exp.metric("temperatures_degrading_any_seed", degrades_any)
        exp.metric("case", case)
        exp.metric("verdict", verdict)
        exp.metric("wall_clock_s", time.time() - t0)

        md = table(scores, stereo)
        exp.path("temperature_table.md").write_text(md + "\n", encoding="utf-8")
        (RESULT_DIR / "temperature.json").write_text(json.dumps(
            {"scores": scores, "untempered_control": untempered,
             "stereo_dependence": stereo,
             "per_temperature": per_temperature, "gates": gate, "macs": macs,
             "viable_temperatures": viable, "case": case, "verdict": verdict},
            indent=2), encoding="utf-8")

        print("\n" + md + "\n")
        print("gates: " + ("PASS" if gate["PASS"] else "FAIL " + str(gate)))
        print("viable temperatures: {}".format(viable or "none"))
        print("temperatures degrading at least one seed: {}".format(degrades_any or "none"))
        print("VERDICT: {}  (case {})".format(verdict, case))
        exp.conclude("Readout temperature sweep over {}: viable temperatures {}; "
                     "verdict {} (case {}).".format(
                         list(TEMPERATURES), viable or "none", verdict, case))
        print("\nrecorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"], nargs="?", default="run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
