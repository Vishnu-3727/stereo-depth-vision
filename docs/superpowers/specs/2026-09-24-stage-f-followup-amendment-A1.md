# Stage F follow-up — amendment A1: second M1 arm (lambda = 1.0)

- ID: A1
- Date: 2026-09-24
- Status: written BEFORE any pilot launch. No training has been launched under
  the parent design or under this amendment.
- Parent (frozen, DO NOT EDIT): `docs/superpowers/specs/2026-09-24-stage-f-followup-design.md`
  (§7 M1 distribution-supervision pilot: lambda = 0.1, b = 0.5 candidates,
  gates PASS / FLOOR-MOVE / FAIL, kill conditions).

Statement tags: [FACT] (recorded artefact), [MEASUREMENT] (number from a frozen
run), [INFERENCE] (conclusion drawn from measurements), [VERDICT] (frozen
decision).

## 1. Why this amendment exists

[FACT] `stage_f/followup/kaggle/grad_probe_m1.json` (2026-09-24, 8 batches,
real Tier-2 data, seed 0, Adam eps 1e-8) measured per-group gradient L2 norms
for the smooth-L1 term and the M1 (lambda-weighted) term at two checkpoints:
the ARM-P Stage-1 init (the pilot init) and the E3 seed0 final.

[MEASUREMENT] ARM-P init (`A_armp_stage1`): ratio m1/sl1 = 0.005519
(feature_extractor; sl1 L2 6.4537, m1 L2 0.035621) and 0.027651 (aggregation;
sl1 L2 6.7493, m1 L2 0.186627). Refinement has no M1 path (upstream group,
null by construction). So at lambda = 0.1 the M1 gradient is ~0.5–3 % of the
smooth-L1 gradient at the pilot init.

[MEASUREMENT] E3 seed0 final (`B_e3seed0_final`): ratio m1/sl1 = 0.037128
(feature_extractor; sl1 L2 0.405807, m1 L2 0.015067) and 0.114181
(aggregation; sl1 L2 1.152366, m1 L2 0.131578).

[MEASUREMENT] Gradient health: frac_below_eps stays ~0.1–0.3 % in all groups
at both checkpoints (ARM-P init: feature sl1 0.001686 / m1 0.002797,
aggregation sl1 0.001161 / m1 0.001381; E3 final similar) — no dead-gradient
regime. CE means 3.0061 (ARM-P) / 2.9349 (E3); SL1 means 4.6822 (ARM-P) /
0.5776 (E3). Median aggregated-cost std is astronomically large on real data
(~1.3e15 ARM-P, ~3.8e17 E3 final), which is why both standardised softmaxes
suppress gradients yet the M1 path stays connected (nonzero, verified by
`smoke_m1.py` on both bundles).

[INFERENCE] Lambda = 0.1 may be too weak to move the pilot measurably within
200 epochs: the M1 term starts at a ~40:1 gradient disadvantage at the init
that matters. A 10x arm (lambda = 1.0) puts the aggregation-group M1 gradient
at ~28 % of smooth-L1 at init — large enough to matter, small enough not to
drown the primary loss.

## 2. Decision

[VERDICT] The pre-registered lambda = 0.1 arm (f1m) is UNCHANGED — same code,
same bundle (`bundle_f1m/`, marker `stage-f-m1`), same gates. A second M1 arm
(f1m10) with lambda = 1.0 is ADDED before any pilot launch. Everything except
the lambda literal is identical: same init checkpoint (sha
`3ae6fb3b…d9f1a29be7`), same E0 recipe, same data/aug/optimizer/seed/epochs
(200, seed 0), same Laplacian target (b = 0.5 candidates), same concurrent
shared control (f1c). The two bundles differ ONLY in the one patched line
(`loss = loss + 0.1 * _m1_ce` vs `loss = loss + 1.0 * _m1_ce`) and
`BUNDLE_MARKER.json` (`stage-f-m1` vs `stage-f-m1-lam1`, plus the recorded
`m1_lambda` field).

[VERDICT] Identical gates for both M1 arms, against the shared control:
PASS iff contract Δfinal ≤ −0.030 px; FLOOR-MOVE iff train GT<64 Δ ≤ −0.050
with Δfinal ≤ 0; else FAIL. Kill conditions unchanged: NaN/Inf, zero valid
px, strict-load fail, pooled entropy collapse < 0.1 nats.

[VERDICT] Multiplicity: two M1 arms against one control. Any single-seed pass
is admission to 3-seed validation ONLY, never acceptance. If both arms pass,
the arm with the larger Δfinal improvement goes forward to 3-seed validation.
If exactly one passes, it goes forward. No lambda sweep beyond these two
pre-registered values without a further amendment.

## 3. Declared environment difference

[FACT] Kaggle allows 2 concurrent GPU sessions per account; three seed-0 arms
(f1c, f1m, f1m10) cannot run concurrently on one account. f1c and f1m run on
`vishnu3727`; f1m10 runs on `vishnuvardhanksece` (per-arm OWNER map in
`push_f1.py`; `STAGE_E_KAGGLE_USER`, when set, overrides all arms —
documented in the module docstring; the kaggle CLI authenticates via the
ambient `kaggle.json`, exactly as `push_e4.py` does — no `KAGGLE_CONFIG_DIR`
handling exists in either script, so the operator must match the active
account to the arm's owner and verify with `kaggle config view`).

[VERDICT] This is a DECLARED environment difference, not a confound smuggled
in: same GPU type (T4), same code identity (bundles differ only per §2, sha
verified), same data identity (same KITTI subset content under both owners),
same init sha. Kernel/environment records will carry the account slug, so any
account-correlated anomaly is attributable.

## 4. Compute

[INFERENCE] Parent design §10: control 200 ep ≈ 1.3 T4-h, one M1 arm ≈ 1.4
T4-h. Two M1 arms + shared control ≈ 1.3 + 1.4 + 1.4 ≈ 4 T4-h total
(~4.1 T4-h with upload/poll overhead). No change to the 3-seed validation
budget (≈8 T4-h), which runs only for the single arm selected per §2.

## 5. Correction recorded by this amendment

[FACT] An early reading of the grad_probe output described the M1 softmax as
"near-uniform". That reading is WRONG.

[INFERENCE] Cross-entropy upper-bounds entropy (ce ≥ H, Gibbs inequality), so
a CE mean of ~3.0 cannot by itself diagnose uniformity — and the frozen audit
measured the model's own distribution entropy at 1.85 nats
(`stage_f/followup/audit2/AUDIT2_REPORT.md`: mean entropy 1.84847 of
ln24 = 3.17805; median target-bin rank 3.0), i.e. broad but decisively NOT
uniform (uniform would be 3.178 nats). The probe's large cost-std numbers
describe the pre-standardisation scale, not the post-softmax distribution.
This amendment records the correction so no downstream analysis cites the
"near-uniform" claim.

## 6. Files (allowed outputs of this amendment, nothing else)

[FACT] Edited: `stage_f/followup/kaggle/{m1_patch.py` (`apply(text, lam=0.1|1.0)`),
`build_bundle_f1m.py` (`--lam {0.1,1.0}`), `push_f1.py` (arms f1c/f1m/f1m10,
per-arm owners, `dataset <arm>`), `smoke_m1.py` (`--lam`, literal assert per
bundle). Rebuilt: `bundle_f1m/` (finetune sha unchanged — see below) and
`bundle_f1m10/`; `source_integrity_f1m.json`, `source_integrity_f1m10.json`.
New: this file.

[MEASUREMENT] `bundle_f1m/scripts/finetune_pilot.py` sha256 before rebuild
`9e0706b3271f35a3d76930a0d8e5af7597434d9f6675a3f802eaeac6ef8797d7`, after
rebuild identical (`9e0706b3…8797d7`). `bundle_f1m10` finetune sha
`519adacca398bedae780aaa1cb2fd1cd9769c4c389103e1b64bf23e8233d7fb4`.
Unified diff of the two finetune files is exactly one line:
`-            loss = loss + 0.1 * _m1_ce` /
`+            loss = loss + 1.0 * _m1_ce`. Smoke: both bundles SMOKE PASS
(lam 0.1: loss 51.104908 ce 3.856635; lam 1.0: loss 54.575882 ce 3.856635;
same synthetic batch, seed 0 CPU).
