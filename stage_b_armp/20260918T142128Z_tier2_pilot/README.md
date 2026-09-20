# ARM-P Tier-2 Pilot — PRE-FLIGHT GATE ONLY (no training launched)

Pilot dir: `stage_b_armp/20260918T142128Z_tier2_pilot/` (UTC).
Seed for the future runs: **0** (predetermined, single).

## Status

**BUILD AND SMOKE-TEST ONLY. The 200-epoch runs were NOT launched.**
Expected wall-clock per 200-epoch arm, from P2A's recorded **3829.4 s**
(`phase2/runs/p2a_scale_coverage/p2a_record.json: wall_clock_s = 3829.35`):
**~3829 s ≈ 63.8 min ≈ 1.06 h per arm** on the same hardware (P2A record says
`cuda`), so **~2.1 h for both arms sequentially**. ARM-P may differ slightly
(lower initial loss → same step count, though; step count is fixed at
200 epochs × 80 batches).

## Design

Two arms, identical in everything except initialization, same seed 0:

- **ARM-CONTROL**: random init (PyTorch defaults) — this IS P2A's own protocol.
- **ARM-P**: strict-load of the FT3D Stage-1 best checkpoint
  (`stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth`)
  into the P2A architecture, then the exact P2A KITTI recipe for 200 epochs.

### Why the control is RANDOM init, not P2A-weights init

Initializing the control from P2A's trained weights would give the control
400 total KITTI epochs (200 from P2A + 200 pilot) versus ARM-P's 200, so any
comparison would confound "pretraining helps" with "more KITTI optimization."
Random init is P2A's own starting point, so control-vs-P2A isolates exactly one
variable: the weight initialization (Stage-1 pretraining vs nothing). A
P2A-initialized control would answer a different, uninteresting question
("does 400 epochs beat 200?") and is therefore rejected.

## Recipe fidelity

`scripts/finetune_pilot.py` mirrors
`phase2/scripts/train_p2a_scale_coverage.py` line-for-line, adding ONLY
`--init {random | <path to .pth>}` and `--arm <output-subdir name>`.
Both arms run through this ONE script so implementation is identical across
arms. Proof: `recipe_diff.json` (difflib, 6 non-equal blocks, all classified
as the two flags or their direct consequences). `recipe_recovered.json` holds
every recovered P2A recipe field; `unrecoverable_fields` is empty — no STOP
condition triggered. No existing repo file was modified; P2A weights, config,
source, and all Stage-A artifacts untouched.

Real-pilot invocation (NOT run here):

```powershell
$env:P2A_SEED = "0"
$env:P2A_OUT_DIR = "<pilot>/runs/arm_control"
python <pilot>/scripts/finetune_pilot.py --init random --arm arm_control
$env:P2A_OUT_DIR = "<pilot>/runs/arm_p"
python <pilot>/scripts/finetune_pilot.py --init <armp_stage1_best.pth> --arm arm_p
```

## Gate results

- **Checkpoint contracts** (`checkpoint_contracts.json`): ARM-P sha256 matches
  `3ae6fb3b…9be7`; strict load into P2A arch clean (missing/unexpected/shape
  mismatches all empty; 397954 params; 70 keys). P2A `p2a_best.pth` sha256
  matches `0868ffd1…6033` — untouched, and NOT used as an init.
- **Smoke test** (`smoke_test.json`): one batch (2×3×256×512) per arm through
  the pilot script's own code paths. Random arm: loss 23.1201 finite;
  ARM-P arm: loss 5.6119 finite (lower, as expected from pretrained weights —
  an observation, not a result). Output shape [2,1,256,512], 70 grads all
  finite, Adam + CosineAnnealingLR instantiate on both arms.
- **Nothing retained**: no `optimizer.step()`, no `torch.save`, no `mkdir`,
  no run directory. Pilot dir listing identical before/after (a `__pycache__`
  `.pyc` from importing the script was deleted afterwards).

## Files

- `README.md` (this file), `recipe_recovered.json`, `recipe_diff.json`,
  `checkpoint_contracts.json`, `smoke_test.json`, `scripts/finetune_pilot.py`

## Hard-prohibition compliance

No training ran; no seeds beyond seed 0 were used (smoke reused seed 0
in-memory only); no existing file modified; cost volume / feature grouping /
correlation / aggregation / refinement / readout untouched; teammate repo not
touched; this is not called Phase 3 anywhere.

## Tier-2 pilot run (seed 0, 200 epochs/arm, sequential)

UTC: 2026-09-18T17:03:51Z. CONTROL first (random init), then ARM-P (Stage-1 init `armp_stage1_best.pth`).

Per-arm: `control/` and `armp/` hold `stdout.log`, `training_log.jsonl` (script output),
`epoch_log.jsonl` (same rows plus `epoch_wall_s`; val null except val epochs),
`p2a_best.pth`/`p2a_final.pth` with `.sha256` sidecars, `integrity_guard.json`,
`p2a_record.json` (script output) and `record.json` (written immediately post-arm).
Frozen-contract scores (mirror of `phase2/scripts/eval_p2a.py`, unmodified):
`tier2_eval.json`. Aggregate: `pilot_results.json`.

CONTROL frozen best EPE/D1: 1.5587715928467816 / 9.94207684501697; final EPE/D1: 1.4809434000217176 / 9.236333151624974; monitor-best epoch: 150; wall: 3782.7s.

ARM-P frozen best EPE/D1: 1.2057590024628537 / 6.337282794742922; final EPE/D1: 1.2087612591692831 / 6.349615822248729; monitor-best epoch: 185; wall: 3839.5s.

delta_EPE best: -0.35301259038392785; delta_EPE final: -0.27218214085243453 (negative = ARM-P lower; reference spread 0.0444).

Classification: **PILOT SUPPORTS ARM-P INITIALIZATION** — both best-checkpoint and final-checkpoint deltas are negative beyond the 0.0444 reference spread (single-seed observation only).

Stop conditions: none. Pipeline wall: 7666.5 s.
