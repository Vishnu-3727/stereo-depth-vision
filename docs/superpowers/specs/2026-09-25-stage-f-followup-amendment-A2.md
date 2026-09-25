# Stage F follow-up — amendment A2: twin preflight gates (fp64 G1, two-step G2) + fresh concurrent control

- ID: A2
- Date: 2026-09-25
- Status: APPROVED by the user 2026-09-25, BEFORE any twin training. No twin
  training has been launched under the parent design, under A1, or under this
  amendment.
- Parent (frozen, DO NOT EDIT): `docs/superpowers/specs/2026-09-24-stage-f-followup-design.md`
  (§6 capacity twin: 32 → 48 single knob, function-preserving embed, 200
  epochs, seed 0, E0 recipe, interpretation thresholds, concurrent control).

Statement tags: [FACT] (recorded artefact), [MEASUREMENT] (number from a frozen
run), [INFERENCE] (conclusion drawn from measurements), [VERDICT] (frozen
decision).

## 1. Why this amendment exists

[FACT] `stage_f/followup/twin/preflight.json` (2026-09-25T00:29:31Z, fp32,
GPU, tolerance 1e-5, first 3 contract hailo_val scenes) recorded G1 FAIL on
all 3 scenes.

[MEASUREMENT] fp32 G1 final-disparity max-abs diff (twin vs ARM-P),
`preflight.json` §g1.scenes: scene `000160_10.png` (index 0)
1.5995330810546875 px; scene `000161_10.png` (index 1) 6.834384918212891 px;
scene `000162_10.png` (index 2) 2.96893310546875 px. Verdict `STOP_G1_FAIL`;
no `twin48_init.pth` written.

[FACT] `stage_f/followup/twin/g1_diagnosis.json` (2026-09-25T00:36:20Z)
ran two controls with the same embed code (imported from
`build_twin_init.py`, never edited).

[MEASUREMENT] fp64 CPU twin-vs-ARM-P (`fp64_twin_vs_armp_cpu`, both
`.double()`, same float64 inputs): max-abs diff EXACTLY 0.0 at EVERY stage on
all 3 scenes — left_features, cost_volume, aggregated_cost,
disparity_initial, disparity_final all 0.0 (`rel` 0.0). `first_divergence_fp64`
is null. Conclusion recorded: `EMBED CORRECT (fp64 matches)`.

[MEASUREMENT] fp32 ARM-P vs itself (`fp32_noise_floor.gpu_vs_cpu_max_abs`,
disparity_final): scene 000160_10.png 2.159027099609375 px; scene
000161_10.png 6.973751068115234 px; scene 000162_10.png 2.9128952026367188
px. GPU-twice on the same card is 0.0 at every stage (deterministic), so the
GPU↔CPU gap is pure fp32 rounding amplified by the huge activation scale
(`magnitude_fp32` aggregated_cost ~1.0e17, disparity_final magnitude
~146–164 px).

[INFERENCE] The fp32 1e-5 gate cannot be passed by ANY exact embed: the
unmodified ARM-P against itself already misses it by ~2.16/6.97/2.91 px
(GPU vs CPU fp32). The embed is bit-exact where it matters (fp64 diff 0.0);
the fp32 mismatch is a property of the operating point (activations ~1e17
through a standardised softmax), not of the widening.

[FACT] `stage_f/followup/twin/g2.json` (2026-09-25T00:41:10Z, fp32 GPU,
batch 2 of 256×512 E0-recipe crops, hailo_calib scenes 0–1, seed 0, NO
optimizer step) recorded: `n_cross_slices` 33, `pass_cross_nonzero` true
(33/33 cross old-out←new-in slices nonzero at step 1); `n_new_slices` 99
with 66 zero at step 1 — every new-output row slice (`new_out_rows`) and
every new-output bias slice (`bias_new_out`) exactly 0.0.

[INFERENCE] The step-1 zeros are structural, not a defect: Net2Net-style
zeroing puts exactly 0 on every weight FROM a new channel INTO an old output,
including at the final 1-channel layers (`aggregation.to_cost`,
`refinement.output_conv`); every path from the loss to a new-channel
activation therefore passes through an exactly-0 weight on step 1, while the
zeroed cross slices themselves see nonzero activations × nonzero old-output
grads and hence nonzero grads. Verdict recorded in `g2.json` was `G2_FAIL`.

[FACT] Correction disclosed by this amendment: the earlier `G2_FAIL` in
`g2.json` came from a STRICTER brief definition (every new slice nonzero at
step 1, `pass_strict_all_new_nonzero`) than the parent spec's §6 text ("a
1-step gradient is nonzero on the new slices", as operationalised by
`build_twin_init.py` as cross-slice-nonzero). The json records both flags
side by side (`pass_strict_all_new_nonzero: false`,
`pass_cross_nonzero: true`); the FAIL verdict followed the brief, not the
spec sentence. A2 below replaces both readings with a single frozen two-step
rule before any training.

## 2. Decision

[VERDICT] G1 is evaluated in float64 on CPU, tolerance UNCHANGED at 1e-5, on
the SAME 3 scenes (first 3 contract hailo_val scenes: `000160_10.png`,
`000161_10.png`, `000162_10.png`). PASS iff per-scene final-disparity
max-abs diff ≤ 1e-5 in fp64 on CPU. Rationale: the gate keeps its numerical
meaning (exact-embed check) while the arithmetic (fp64, single device) no
longer confounds it with fp32 noise that the unmodified model itself
exhibits. The fp32 GPU-vs-CPU numbers above are the frozen justification;
they are not revisited.

[VERDICT] G2 PASS iff EVERY cross slice (old-out ← new-in, 33 slices) has
nonzero grad at step 1 AND EVERY new slice — all 99, including new-output
rows and new-output biases — has nonzero grad at step 2 after ONE optimizer
step of the E0 recipe (Adam, lr 1e-3, betas (0.9, 0.999), same defaults the
F1 arms train with) on a real batch (batch 2 of 256×512 E0-recipe training
crops, hailo_calib, seed 0). Step-1 cross-nonzero proves the new capacity is
reachable from step 0; step-2 all-nonzero proves no new parameter is
permanently dead (the one optimizer step moves the zeroed cross weights off
exactly 0, opening the blocked paths). Both halves are required; either half
failing is G2 FAIL. fp32, GPU, train mode — the precision and device the
training will actually use.

[VERDICT] Everything else in parent §6 is UNCHANGED: 200 epochs, seed 0, E0
recipe (data, aug, loss, Adam 1e-3, batch 2, crop, determinism, contract),
params must be 891074 in [800k, 1.0M], twin seed 0, same ARM-P Stage-1 source
(sha `3ae6fb3b…d9f1a29be7`), same interpretation thresholds (Δ train GT<64
twin − control ≤ −0.050 → CAPACITY SUPPORTED; |Δ| < 0.020 → CAPACITY
WEAKENED; else INCONCLUSIVE; neither is proof).

## 3. Control

[VERDICT] A FRESH concurrent width-32 E0-recipe seed-0 run (t48c) is launched
WITH the twin (t48). The earlier f1c run is NOT reused: it was concurrent
with M1, not with the twin. Standing rule (parent §6): a control is never a
historical record. t48 and t48c share bundle lineage (E0 recipe, same data
identity), seed, epochs, session type, and owner (both
`vishnuvardhanksece`); they differ ONLY in the single width knob and its
init (twin48 embed vs ARM-P Stage-1).

## 4. Files (allowed outputs under this amendment, nothing else)

[FACT] New: this file; `stage_f/followup/twin/finalize_twin_init.py` (imports
`build_twin_init.py`; edits nothing), `twin48_init.pth` (ONLY if A2-G1 and
A2-G2 both PASS), `preflight_a2.json`, `finalize.log`;
`stage_f/followup/kaggle/build_bundle_twin.py`, `bundle_twin/`,
`source_integrity_twin.json`, `push_twin.py`, `kernel_t48_seed0/`,
`kernel_t48c_seed0/`, `smoke_twin.py`, `smoke_twin.json`. No existing file is
edited. No Kaggle call is made by this amendment (bundle/kernel/smoke tooling
only; upload and push are explicit manager commands listed in the worker's
report).
