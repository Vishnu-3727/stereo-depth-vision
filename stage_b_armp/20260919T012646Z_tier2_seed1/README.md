# ARM-P Tier-2 Replication — SEED 1 (BUILD + LOW-COMPUTE GATE ONLY)

New dir: `stage_b_armp/20260919T012646Z_tier2_seed1/` (UTC).
Predetermined seed: **1** (single new seed, used by BOTH arms).
Source pilot (seed 0, completed): `stage_b_armp/20260918T142128Z_tier2_pilot/`
(authoritative; read, never modified).

## Status

**BUILD AND GATE ONLY. The 200-epoch training was NOT launched.**
No `optimizer.step()`, no `torch.save()`, no run directory was created
(`control/` and `armp/` do not exist). The manager launches training after
reviewing this gate.

## Design

Same two-arm replication as the seed-0 pilot, identical in everything except
initialization, now at seed 1:

- **ARM-CONTROL**: random init (PyTorch defaults) — P2A's own protocol. This is
  RANDOM INITIALIZATION under the P2A recipe, NOT "P2A checkpoint" initialization.
- **ARM-P**: strict-load of the FT3D Stage-1 best checkpoint
  (`stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth`)
  into the P2A architecture, then the exact P2A KITTI recipe for 200 epochs.

Random-init control isolates exactly one variable (Stage-1 pretraining vs
nothing); a P2A-weights control would confound pretraining with extra KITTI
epochs and is rejected. One seed: no statistical significance is claimed
anywhere.

## Harness

The four scripts and `launch_tier2.ps1` were copied byte-for-byte from the
seed-0 pilot; the ONLY edits (all in the new copies) are:

- (a) `scripts/run_arm.py` line 83: `P2A_SEED` `0` -> `1` (replication variable).
- (b) `launch_tier2.ps1`: `$PILOT` -> the new dir.
- (c) `scripts/eval_tier2.py` line 134: checkpoint provenance label -> new dir.
  This string is a provenance label in the output record ONLY; the real load
  path (line 124, `PILOT / arm / p2a_<tag>.pth`) resolves from the script's own
  dir and needed no edit.

Proof: `harness_diff.json` (difflib, 3 non-equal blocks, all classified;
zero UNEXPECTED).

## Gate outcome

`gate.json`: **GATE PASS (10/10)** — seed resolves to 1 on both arms;
both configs fully resolved and identical except init/arm-identity fields;
Stage-1 checkpoint strict-loads clean (sha256 match, 397954 params, 70 keys);
random control instantiates (397954 params, 70 keys); one-batch smoke finite
on both arms (control loss 14.6319, armp loss 2.7289 — observation, not a
result); optimizer/scheduler constructed fresh after init; no checkpoint
overwritten. Full configs + init record + provenance: `resolved_config.json`.

No results, no verdict — those come from the future 200-epoch run.

## Tier-2 pilot run (seed 0, 200 epochs/arm, sequential)

UTC: 2026-09-19T03:38:21Z. CONTROL first (random init), then ARM-P (Stage-1 init `armp_stage1_best.pth`).

Per-arm: `control/` and `armp/` hold `stdout.log`, `training_log.jsonl` (script output),
`epoch_log.jsonl` (same rows plus `epoch_wall_s`; val null except val epochs),
`p2a_best.pth`/`p2a_final.pth` with `.sha256` sidecars, `integrity_guard.json`,
`p2a_record.json` (script output) and `record.json` (written immediately post-arm).
Frozen-contract scores (mirror of `phase2/scripts/eval_p2a.py`, unmodified):
`tier2_eval.json`. Aggregate: `pilot_results.json`.

CONTROL frozen best EPE/D1: 1.4440242127250567 / 8.61534286473877; final EPE/D1: 1.4517076971230227 / 8.63056849997515; monitor-best epoch: 185; wall: 3826.5s.

ARM-P frozen best EPE/D1: 1.191216765057325 / 6.211033615520366; final EPE/D1: 1.2017375876950878 / 6.239354874846067; monitor-best epoch: 185; wall: 3742.5s.

delta_EPE best: -0.25280744766773156; delta_EPE final: -0.24997010942793496 (negative = ARM-P lower; reference spread 0.0444).

Classification: **PILOT SUPPORTS ARM-P INITIALIZATION** — both best-checkpoint and final-checkpoint deltas are negative beyond the 0.0444 reference spread (single-seed observation only).

Stop conditions: none. Pipeline wall: 7603.1 s.
