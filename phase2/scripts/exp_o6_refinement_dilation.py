"""EXP-O6-REFINEMENT-DILATION-001 -- capacity, or receptive field?

E3C's three-block arm changed two things at once: it removed three residual
blocks *and* it removed dilation 8. Its own pre-registration said so before it
ran, and pre-committed this experiment as the disambiguation.

    python phase2/scripts/exp_o6_refinement_dilation.py preflight
    python phase2/scripts/exp_o6_refinement_dilation.py viability
    python phase2/scripts/exp_o6_refinement_dilation.py train --arm D124
    python phase2/scripts/exp_o6_refinement_dilation.py train --arm D128
    python phase2/scripts/exp_o6_refinement_dilation.py train --arm D148
    python phase2/scripts/exp_o6_refinement_dilation.py analyse

Arms, construction, band, decision rule and thresholds are frozen in
`phase2/docs/EXP_O6_REFINEMENT_DILATION_001_PREREGISTRATION.md`. This file only
executes them.

The trick that makes the experiment work: a dilated 3x3 convolution has the same
weight shape and the same MAC count as an undilated one. So all three arms hold
368,098 parameters and identical arithmetic cost, and -- because every arm keeps
refinement blocks 0,1,2 from one seed-0 initialisation and is then *re-dilated*
in place -- bit-identical initial weights. Capacity, MACs and initialisation are
constant by construction; the dilation schedule is the only variable.

Everything except the dilation schedule is imported from the E3b/H2 harness, not
restated. No Phase 1 file, no H2 checkpoint and no existing record is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import phase2.scripts.exp_e3b_refinement_capacity as e3b  # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (  # noqa: E402
    EPOCHS, EXPERIMENTS_DIR, FOCUS_SCENES, OUT_DIR, build_model, git_state,
    measure_macs, recipe_config, stereo_probe,
)
from phase2.scripts.exp_h3_viability import full_forward  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

EXPERIMENT_PREFIX = "EXP-O6-REFINEMENT-DILATION-001"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / EXPERIMENT_PREFIX
PREREGISTRATION = "phase2/docs/EXP_O6_REFINEMENT_DILATION_001_PREREGISTRATION.md"
SEED = 0

# Frozen: pre-registration section 3. Every arm keeps refinement blocks 0,1,2 of
# the seed-0 six-block stack and is then re-dilated in place, so the arms share
# their initial weights exactly (section 4).
KEEP = (0, 1, 2)
ARMS = {
    "D124": {"dilations": (1, 2, 4),
             "role": "E3C's configuration, retrained under this harness"},
    "D128": {"dilations": (1, 2, 8),
             "role": "keeps maximum dilation 8 at three blocks"},
    "D148": {"dilations": (1, 4, 8),
             "role": "different sparse allocation, also keeping dilation 8"},
}
# Dilation changes neither weight shape nor MAC count, so every arm matches
# E3C arm D exactly. The preflight asserts this rather than trusting it.
EXPECTED_PARAMETERS = 368098
EXPECTED_MACS_DEPLOY = 30.964e9

# Reused as a measured input, NOT retrained (section 4).
CONTROL_ID = "EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2"
# E3C's own three-block record, used only as a same-recipe cross-check.
E3C_ARM_D_ID = "EXP-E3C-REFINEMENT-FLOOR-001-ARM-D"
# Frozen in section 7, reused from E3b/E3C, not recomputed.
BAND = {"epe": 0.21380062638697517, "d1": 1.0}
# Frozen in section 8.
RECOVERY_STRONG = 0.50
RECOVERY_WEAK = 0.20

# Recipe fields that must be byte-equal across arms; the dilation schedule and
# the per-arm bookkeeping are the only permitted differences.
RECIPE_KEYS_MUST_MATCH = sorted(recipe_config(SEED, "cpu").keys() - {"device"})


def arm_id(arm: str) -> str:
    return "{}-ARM-{}".format(EXPERIMENT_PREFIX, arm)


def set_dilations(model, dilations: tuple[int, ...]) -> None:
    """Re-dilate the surviving refinement blocks in place.

    Only ``dilation`` and ``padding`` change. Weight tensors are not touched, so
    two arms built this way from the same seed differ in the sampling grid and
    in nothing else. Padding must track dilation or the residual add inside
    ``ResBlock`` would see a shape change.
    """
    blocks = model.refinement.blocks
    if len(blocks) != len(dilations):
        raise ValueError("{} blocks but {} dilations".format(len(blocks), len(dilations)))
    for block, d in zip(blocks, dilations):
        for conv in (block.conv1, block.conv2):
            conv.dilation = (int(d), int(d))
            conv.padding = (int(d), int(d))


def build_arm(arm: str, device: str):
    """Seed-0 six-block H2 model, keep blocks 0-2, re-dilate to this arm."""
    model = build_model(SEED, device)      # seeds torch/np, shift=left, H2 readout
    model.refinement.blocks = nn.Sequential(*[model.refinement.blocks[i] for i in KEEP])
    set_dilations(model, ARMS[arm]["dilations"])
    return model


def effective_dilations(model) -> list[int]:
    """Read the schedule back off the built model, rather than trusting the spec."""
    out = []
    for block in model.refinement.blocks:
        d1, d2 = block.conv1.dilation, block.conv2.dilation
        assert d1 == d2 and d1[0] == d1[1], (d1, d2)
        assert block.conv1.padding == d1 and block.conv2.padding == d2, "padding must track dilation"
        out.append(int(d1[0]))
    return out


def receptive_field(dilations) -> int:
    """Refinement-stack receptive field in full-resolution pixels, one side.

    DERIVED, not measured: each 3x3 conv at dilation d adds d on each side, the
    stack has two convs per block plus the 3x3 input and output convs.
    """
    return 1 + 2 * (1 + 1 + sum(2 * int(d) for d in dilations))


def arm_config(arm: str, device: str) -> dict:
    spec = ARMS[arm]
    config = dict(recipe_config(SEED, device))
    config.update({
        "experiment": arm_id(arm),
        "preregistration": PREREGISTRATION,
        "approved_by": "phase2/docs/EXP_E3C_REFINEMENT_FLOOR_001_PREREGISTRATION.md "
                       "section 8 (pre-committed follow-up)",
        "hypothesis": ("The D1 degradation of the three-block (1,2,4) stack is "
                       "substantially attributable to removing the dilation-8 "
                       "receptive field rather than to having fewer blocks."),
        "arm": arm,
        "arm_role": spec["role"],
        "refinement_blocks": len(spec["dilations"]),
        "refinement_dilations": list(spec["dilations"]),
        "refinement_receptive_field_px": receptive_field(spec["dilations"]),
        "changed_variable": ("refinement dilation schedule only: {} instead of "
                             "(1,2,4). Block count, parameter count, MACs and "
                             "initial weights are identical across all three O6 "
                             "arms by construction.".format(list(spec["dilations"]))),
        "construction": ("seed-0 six-block H2 model, keep refinement blocks "
                         "{}, then re-dilate in place (dilation and padding "
                         "only; weight tensors untouched)".format(list(KEEP))),
        "control": ("{} (6 blocks, seed 0, NOT retrained -- reused as a measured "
                    "input); the within-experiment comparison the hypothesis "
                    "turns on is D128/D148 against D124, both trained "
                    "here".format(CONTROL_ID)),
        "band": BAND,
        "known_limitation": ("one seed per arm -- suggestive/exploratory, no "
                             "significance claimed or claimable. 160 training "
                             "scenes from random initialisation, so nothing "
                             "transfers to a pretrained model. The six-block "
                             "control was trained in an earlier session. MACs "
                             "are arithmetic, not latency."),
        "known_confound": ("none intended within the three-block set: capacity, "
                           "MACs and initialisation are held exactly constant. "
                           "The control comparison carries cross-session harness "
                           "drift; the three-block comparisons do not."),
    })
    return config


def _install() -> None:
    """Point E3b's training loop at this experiment. No E3b record is touched."""
    for arm, spec in ARMS.items():
        e3b.ARMS[arm] = {
            "blocks": len(spec["dilations"]), "keep": KEEP,
            "dilations": spec["dilations"], "parameters": EXPECTED_PARAMETERS,
            "macs_deploy": EXPECTED_MACS_DEPLOY, "role": spec["role"],
        }
    e3b.EXPERIMENT_PREFIX = EXPERIMENT_PREFIX
    e3b.RESULT_DIR = RESULT_DIR
    e3b.arm_id = arm_id
    e3b.build_arm = build_arm
    e3b.arm_config = arm_config
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


# --- preflight ---------------------------------------------------------------

def run_preflight(device: str) -> dict:
    """Blocking. The arms may differ in nothing but the dilation schedule."""
    models = {arm: build_arm(arm, device) for arm in ARMS}
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    out = {"git": git_state(), "arms": {}, "checks": {}}

    # Pass 1 -- weight identity across arms, before thop attaches buffers.
    names = sorted(ARMS)
    reference = models[names[0]]
    ref_state = {k: v for k, v in reference.state_dict().items()}
    weights_identical = True
    for arm in names[1:]:
        state = models[arm].state_dict()
        if set(state) != set(ref_state):
            weights_identical = False
            continue
        weights_identical = weights_identical and all(
            torch.equal(ref_state[k], state[k]) for k in ref_state)

    # Pass 2 -- shapes, finiteness, schedule read back off the model.
    shapes = {}
    for arm, model in models.items():
        model.eval()
        finite, ranges = True, {}
        for scene in scenes:
            stages = full_forward(model, scene.left, scene.right, device)
            for stage, tensor in stages.items():
                if not torch.isfinite(tensor).all():
                    finite = False
                ranges.setdefault(stage, []).append(float(tensor.abs().max()))
            shapes.setdefault(arm, {}).update(
                {k: list(v.shape) for k, v in stages.items()})
        out["arms"][arm] = {
            "blocks": len(model.refinement.blocks),
            "dilations_declared": list(ARMS[arm]["dilations"]),
            "dilations_effective": effective_dilations(model),
            "receptive_field_px": receptive_field(ARMS[arm]["dilations"]),
            "parameters": model.parameter_count(),
            "parameters_expected": EXPECTED_PARAMETERS,
            "parameters_match": model.parameter_count() == EXPECTED_PARAMETERS,
            "finite_at_init": finite,
            "stage_max_abs": {k: max(v) for k, v in ranges.items()},
            "cost_volume_shift": model.cost_volume.shift,
            "regression": type(model.regression).__name__,
            "num_disparities": model.config.num_disparities,
            "macs": measure_macs(model, device),   # last: thop mutates the model
        }

    configs = {arm: arm_config(arm, device) for arm in ARMS}
    recipe_identical = all(
        configs[a][k] == configs[names[0]][k]
        for a in names for k in RECIPE_KEYS_MUST_MATCH)
    macs = [out["arms"][a]["macs"].get("macs") for a in names]
    shapes_identical = all(shapes[a] == shapes[names[0]] for a in names)

    out["checks"] = {
        "arms_registered": sorted(ARMS) == sorted(configs),
        "initial_weights_bit_identical_across_arms": bool(weights_identical),
        "declared_dilations_are_effective": all(
            out["arms"][a]["dilations_effective"] == out["arms"][a]["dilations_declared"]
            for a in ARMS),
        "dilation_schedules_are_distinct": len(
            {tuple(ARMS[a]["dilations"]) for a in ARMS}) == len(ARMS),
        "parameter_counts_match_registration": all(
            out["arms"][a]["parameters_match"] for a in ARMS),
        "macs_identical_across_arms": bool(
            all(m is not None for m in macs) and len(set(macs)) == 1),
        "tensor_shapes_identical_across_arms": bool(shapes_identical),
        "recipe_fields_identical_across_arms": bool(recipe_identical),
        "all_finite_at_init": all(out["arms"][a]["finite_at_init"] for a in ARMS),
        "shift_is_left": all(out["arms"][a]["cost_volume_shift"] == "left" for a in ARMS),
        "readout_is_standardised": all(
            out["arms"][a]["regression"] == "StandardisedDisparityRegression"
            for a in ARMS),
        "num_disparities_is_12": all(
            out["arms"][a]["num_disparities"] == 12 for a in ARMS),
        "control_record_exists": (EXPERIMENTS_DIR / CONTROL_ID / "metrics.json").exists(),
        "phase_1_diff_empty": out["git"]["phase_1_diff_vs_frozen"] == "",
    }
    out["checks"]["h2_checkpoint_sha256"] = core.sha256(core.checkpoint_path("H2"))
    out["checks"]["PASS"] = all(
        v for v in out["checks"].values() if isinstance(v, bool))
    out["control"] = {
        "experiment": CONTROL_ID,
        "checkpoint_sha256": core.sha256(OUT_DIR / (CONTROL_ID + "_checkpoint.pth")),
        "band": BAND,
    }
    return out


# --- viability ---------------------------------------------------------------

def run_viability(device: str) -> dict:
    """Cheap end-to-end gate before spending GPU-hours: construct, forward,
    backward, finite loss and gradients, an optimizer step that moves weights,
    validation, and a checkpoint round-trip -- for every arm."""
    from torch.utils.data import DataLoader

    from phase2.scripts.exp_h1_cost_volume import CroppedKitti, validate
    from src.datasets.kitti2015 import Kitti2015Stereo
    from src.losses.disparity import masked_smooth_l1

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    tmp = RESULT_DIR / "viability"
    tmp.mkdir(parents=True, exist_ok=True)

    out = {"arms": {}}
    for arm in sorted(ARMS):
        model = build_arm(arm, device)
        max_disparity = float(model.config.max_disparity_px)
        loader = DataLoader(CroppedKitti(train_base, seed=SEED), batch_size=2,
                            shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        model.train()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}

        left, right, disparity = next(iter(loader))
        left, right, disparity = left.to(device), right.to(device), disparity.to(device)
        pred = model(left, right)
        loss, _ = masked_smooth_l1(pred, disparity, max_disparity=max_disparity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_finite = all(bool(torch.isfinite(g).all()) for g in grads)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
        refine_grad = float(sum(
            float(p.grad.abs().sum()) for p in model.refinement.parameters()
            if p.grad is not None))
        optimizer.step()
        after = model.state_dict()
        moved = sum(1 for k in before if not torch.equal(before[k], after[k]))

        model.eval()
        metrics = validate(model, val_base, device, limit=2)
        path = tmp / "{}_viability.pth".format(arm)
        torch.save({"model": model.state_dict(), "arm": arm}, path)
        reloaded = torch.load(path, map_location="cpu", weights_only=False)["model"]
        roundtrip = all(torch.equal(after[k].cpu(), reloaded[k]) for k in after)
        path.unlink()

        out["arms"][arm] = {
            "constructs": True,
            "forward_shape": list(pred.shape),
            "forward_finite": bool(torch.isfinite(pred).all()),
            "loss": float(loss),
            "loss_finite": bool(np.isfinite(float(loss))),
            "gradients_present": len(grads) > 0,
            "gradients_finite": bool(grad_finite),
            "gradient_norm": grad_norm,
            "refinement_gradient_abs_sum": refine_grad,
            "refinement_receives_gradient": refine_grad > 0.0,
            "optimizer_moved_tensors": moved,
            "optimizer_updates": moved > 0,
            "validation_epe": metrics.epe,
            "validation_d1": metrics.d1,
            "validation_finite": bool(np.isfinite(metrics.epe) and np.isfinite(metrics.d1)),
            "checkpoint_roundtrip": bool(roundtrip),
        }
    out["PASS"] = all(
        v for arm in out["arms"].values() for k, v in arm.items()
        if isinstance(v, bool))
    out["h2_checkpoint_sha256"] = core.sha256(core.checkpoint_path("H2"))
    return out


# --- analysis ----------------------------------------------------------------

def _metrics(experiment_id: str) -> dict:
    path = EXPERIMENTS_DIR / experiment_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def run_analysis(device: str) -> None:
    exp_id = EXPERIMENT_PREFIX + "-ANALYSIS"
    if (EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(exp_id + " already exists -- records are never overwritten.")

    arms = {a: _metrics(arm_id(a)) for a in ARMS}
    control = _metrics(CONTROL_ID)
    e3c = _metrics(E3C_ARM_D_ID)

    def late(m, key):
        return m["late_window"][key]

    gap = late(arms["D124"], "d1_mean") - late(control, "d1_mean")
    gap_material = abs(gap) > BAND["d1"]

    rows = {}
    for a in sorted(ARMS):
        d_epe_ctrl = late(arms[a], "epe_mean") - late(control, "epe_mean")
        d_d1_ctrl = late(arms[a], "d1_mean") - late(control, "d1_mean")
        d_epe_124 = late(arms[a], "epe_mean") - late(arms["D124"], "epe_mean")
        d_d1_124 = late(arms[a], "d1_mean") - late(arms["D124"], "d1_mean")
        verdict = arms[a].get("stereo_verdict_epoch_200", {})
        rows[a] = {
            "dilations": list(ARMS[a]["dilations"]),
            "receptive_field_px": receptive_field(ARMS[a]["dilations"]),
            "parameters": arms[a]["parameter_count_unique"],
            "late_window": arms[a]["late_window"],
            "delta_epe_vs_control": d_epe_ctrl,
            "delta_d1_vs_control": d_d1_ctrl,
            "material_vs_control": bool(
                abs(d_epe_ctrl) > BAND["epe"] and abs(d_d1_ctrl) > BAND["d1"]),
            "delta_epe_vs_D124": d_epe_124,
            "delta_d1_vs_D124": d_d1_124,
            "material_vs_D124": bool(
                abs(d_epe_124) > BAND["epe"] and abs(d_d1_124) > BAND["d1"]),
            "d1_gap_recovery": ((late(arms["D124"], "d1_mean") - late(arms[a], "d1_mean"))
                                / gap) if gap_material else None,
            "stereo_functional": bool(verdict.get("STEREO_FUNCTIONAL")),
            "stereo_verdict": verdict,
            "aborted": arms[a].get("aborted"),
        }

    # Same-recipe cross-check: this session's D124 against E3C's own arm D.
    cross_check = {
        "reference": E3C_ARM_D_ID,
        "delta_epe": late(arms["D124"], "epe_mean") - late(e3c, "epe_mean"),
        "delta_d1": late(arms["D124"], "d1_mean") - late(e3c, "d1_mean"),
        "band": BAND,
    }
    cross_check["within_band"] = bool(
        abs(cross_check["delta_epe"]) <= BAND["epe"]
        and abs(cross_check["delta_d1"]) <= BAND["d1"])

    gate = {
        "arms_completed": all(
            arms[a].get("aborted") is None and arms[a]["epochs_completed"] == EPOCHS
            for a in ARMS),
        "control_stereo_functional": bool(
            control.get("stereo_verdict_epoch_200", {}).get("STEREO_FUNCTIONAL")),
        "parameters_identical_across_arms": len(
            {arms[a]["parameter_count_unique"] for a in ARMS}) == 1,
        "phase_1_diff_empty": git_state()["phase_1_diff_vs_frozen"] == "",
        "h2_checkpoint_unchanged": core.sha256(core.checkpoint_path("H2"))
        == "e3d48021d7f6a3d4d16fe33031e8ba85fd24389c5ba892f52c7e873e48454927",
    }
    gate["PASS"] = all(gate.values())

    broken = [a for a in sorted(ARMS) if not rows[a]["stereo_functional"]]
    strong = [a for a in ("D128", "D148")
              if rows[a]["d1_gap_recovery"] is not None
              and rows[a]["d1_gap_recovery"] >= RECOVERY_STRONG]
    weak = [a for a in ("D128", "D148")
            if rows[a]["d1_gap_recovery"] is not None
            and rows[a]["d1_gap_recovery"] <= RECOVERY_WEAK]

    if not gate["PASS"]:
        verdict = "INVALID"
    elif broken:
        verdict = "STEREO-BROKEN"
    elif not gap_material:
        verdict = "NO-GAP-TO-EXPLAIN"
    elif len(strong) == 2:
        verdict = "RECEPTIVE-FIELD-IMPLICATED"
    elif len(weak) == 2:
        verdict = "CAPACITY-IMPLICATED"
    elif len(strong) == 1 and len(weak) == 1:
        verdict = "CONFIGURATION-SENSITIVE"
    else:
        verdict = "INCONCLUSIVE"

    payload = {
        "verdict": verdict,
        "gap": {"d1_gap_D124_minus_control": gap, "material": bool(gap_material),
                "band": BAND},
        "control": {"experiment": CONTROL_ID, "late_window": control["late_window"],
                    "parameters": control["parameter_count_unique"],
                    "dilations": [1, 2, 4, 8, 1, 1],
                    "receptive_field_px": receptive_field((1, 2, 4, 8, 1, 1))},
        "arms": rows,
        "cross_check_vs_E3C_arm_D": cross_check,
        "gates": gate,
        "thresholds": {"recovery_strong": RECOVERY_STRONG, "recovery_weak": RECOVERY_WEAK},
    }

    config = {
        "experiment": exp_id,
        "hypothesis": "Analysis of the three O6 arms against the frozen rule.",
        "preregistration": PREREGISTRATION,
        "measurement_only": True,
        "training": "none -- reads the three O6 arm records plus two existing ones",
        "control": CONTROL_ID, "band": BAND,
        "dataset": "kitti2015", "split": "hailo_val (first 10 scenes, recorded protocol)",
        "resolution": "368x1232", "crop": [256, 512], "disparity_range": 12,
        "batch_size": 2, "precision": "fp32", "seed": SEED,
        "known_limitation": ("one seed per arm: suggestive/exploratory, no "
                             "significance claimed. The band is a single-sample "
                             "lower bound on same-seed noise, not a confidence "
                             "interval. The six-block control was trained in an "
                             "earlier session."),
    }
    with Experiment("O6 refinement dilation analysis", config=config,
                    experiment_id=exp_id, experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["known_limitation"])
        exp.log("control (6 blocks, {} params)  late EPE {:.4f}  D1 {:.4f}".format(
            control["parameter_count_unique"], late(control, "epe_mean"),
            late(control, "d1_mean")))
        for a in sorted(ARMS):
            r = rows[a]
            exp.log("{:<6} {:<12} RF {:>4} px  late EPE {:.4f} +- {:.4f}  D1 {:.4f} +- {:.4f}"
                    "  dD1 vs D124 {:+.4f}  recovery {}".format(
                        a, str(tuple(r["dilations"])), r["receptive_field_px"],
                        r["late_window"]["epe_mean"], r["late_window"]["epe_sd"],
                        r["late_window"]["d1_mean"], r["late_window"]["d1_sd"],
                        r["delta_d1_vs_D124"],
                        "n/a" if r["d1_gap_recovery"] is None
                        else "{:.1%}".format(r["d1_gap_recovery"])))
        exp.log("D1 gap D124 - control = {:+.4f} pt (band {:.4f}) -> material {}".format(
            gap, BAND["d1"], gap_material))
        exp.log("cross-check vs {}: dEPE {:+.4f}  dD1 {:+.4f}  within band {}".format(
            E3C_ARM_D_ID, cross_check["delta_epe"], cross_check["delta_d1"],
            cross_check["within_band"]))
        for key, value in payload.items():
            exp.metric(key, value)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "analysis.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
        exp.conclude("O6 verdict {}: D1 gap {:+.4f} pt; recovery D128 {} / D148 {}."
                     .format(verdict, gap,
                             "n/a" if rows["D128"]["d1_gap_recovery"] is None
                             else "{:.1%}".format(rows["D128"]["d1_gap_recovery"]),
                             "n/a" if rows["D148"]["d1_gap_recovery"] is None
                             else "{:.1%}".format(rows["D148"]["d1_gap_recovery"])))
        print("\nVERDICT: " + verdict)
        print("recorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["preflight", "viability", "train", "analyse"])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _install()

    if args.stage == "preflight":
        result = run_preflight(device)
        (RESULT_DIR / "preflight.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
        for arm in sorted(result["arms"]):
            row = result["arms"][arm]
            print("arm {:<6} dilations {:<12} RF {:>4} px  {} params ({})  "
                  "{:.3f} GMAC".format(
                      arm, str(tuple(row["dilations_effective"])),
                      row["receptive_field_px"], row["parameters"],
                      "ok" if row["parameters_match"] else "MISMATCH",
                      (row["macs"].get("macs") or float("nan")) / 1e9))
        for key, value in result["checks"].items():
            print("  {:<46} {}".format(key, value))
        print("\nPREFLIGHT: " + ("PASS" if result["checks"]["PASS"] else "FAIL"))
        if not result["checks"]["PASS"]:
            raise SystemExit(1)
        return

    if args.stage == "viability":
        preflight = RESULT_DIR / "preflight.json"
        if not preflight.exists() or not json.loads(
                preflight.read_text(encoding="utf-8"))["checks"]["PASS"]:
            raise SystemExit("run a passing preflight first")
        result = run_viability(device)
        (RESULT_DIR / "viability.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
        for arm in sorted(result["arms"]):
            row = result["arms"][arm]
            print("arm {:<6} loss {:.4f}  grad {:.3f}  refine-grad {:.3e}  "
                  "moved {} tensors  val EPE {:.3f} D1 {:.2f}%".format(
                      arm, row["loss"], row["gradient_norm"],
                      row["refinement_gradient_abs_sum"], row["optimizer_moved_tensors"],
                      row["validation_epe"], row["validation_d1"]))
        print("\nVIABILITY: " + ("PASS" if result["PASS"] else "FAIL"))
        if not result["PASS"]:
            raise SystemExit(1)
        return

    if args.stage == "train":
        if not args.arm:
            raise SystemExit("--arm is required for train")
        viability = RESULT_DIR / "viability.json"
        if not viability.exists() or not json.loads(
                viability.read_text(encoding="utf-8"))["PASS"]:
            raise SystemExit("run a passing viability gate first")
        e3b.run_training(args.arm, device)
        return

    run_analysis(device)


if __name__ == "__main__":
    main()
