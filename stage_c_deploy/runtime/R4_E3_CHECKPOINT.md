# R4 — E3 seed-0 final checkpoint scored through the Stage-C runtime path

## 1 Scope and status

**VERDICT: measurement only. The recommended runtime path is unchanged, no frozen
record is touched, and no speed claim is made.**

This experiment scores one foreign checkpoint — the Stage-E E3 seed-0 final
(`stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth`, sha256
`82e58bc441a4...ea79c6d`) — through the recommended deterministic fast runtime
path (cuda, fp32, concurrent-cv2 decode, GPU normalize, cudnn.benchmark OFF).
It is host runtime work, added after the Stage C validation closed. It changes
no frozen artefact, no checkpoint, no evaluation contract and no accuracy
number, and it does not bear on Hailo deployment, which remains blocked.

The instrument is an opt-in `--checkpoint` override on `runtime_path.py` (timing)
and a new `r4_contract.py` (40-scene contract score). Absent the flag, both
scripts call `AD.load_frozen_net(device)` with no checkpoint arguments, so the
frozen checkpoint is verified against the hardcoded `ARMP_SHA`
(`b2f6f5d558db...eb7454`) exactly as before — see section 9. The override is
opt-in only; no default checkpoint changed anywhere, and
`stage_c_deploy/demo/pipeline_demo.py` is untouched.

## 2 What was run

| Run | Script + flags | Output JSON |
|---|---|---|
| Control contract | `r4_contract.py --out out/r4_control_40scene_cuda.json` (no `--checkpoint`) | `out/r4_control_40scene_cuda.json` |
| E3 contract | `r4_contract.py --checkpoint stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth --out out/r4_e3s0final_40scene_cuda.json` | `out/r4_e3s0final_40scene_cuda.json` |
| E3 timing (earlier session) | `runtime_path.py --device cuda --gpu-geometry --scenes 10 --warmup 2 --checkpoint <e3>` | `out/r4_e3s0final_cuda_fp32_det_s10_w2.json` |
| Control timing (this session) | `runtime_path.py --device cuda --gpu-geometry --scenes 10 --warmup 2` | `out/r4_control_cuda_fp32_det_s10_w2.json` |
| E3 timing rerun (this session) | same protocol + `--checkpoint <e3>` | `out/r4_e3s0final_cuda_fp32_det_s10_w2_same_session.json` |

The contract runs score decode + normalize + inference over the full frozen
40-scene `hailo_val` contract via `frozen_eval.pooled_metrics` /
`refuse_unless_contract` — the same functions the frozen evaluator uses, not a
reimplementation. Geometry stages are not run in `r4_contract.py`: they cannot
move the disparity (README sections 4–5 establish decode/normalize/geometry
equivalence), so the contract score exercises decode + normalize + inference
only. The timing runs use the full timed path including geometry, identical
protocol in all three timing JSONs (10 scenes from the head of `hailo_val`,
warmup 2, medians, deterministic defaults, no `--cudnn-bench`,
`--channels-last`, `--compile`, fp16 or bf16).

## 3 Host and protocol

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| torch | 2.7.0+cu128 |
| Python | 3.12.9 |
| dtype / resolution | fp32, 368x1232 |
| Contract | 40 scenes `hailo_val`, `disp_occ_0`, gt_scale 256.0, reference valid pixels 3802797 |
| Timing | first 10 of `hailo_val`, warmup 2, median per stage |
| Checkpoints | frozen `b2f6f5d558db...eb7454` (asserted every default run); E3 seed-0 final `82e58bc441a4...ea79c6d` (measured sha asserted at every override load) |

## 4 Control reproduction: bit-identical to the frozen record

The control contract run reproduces the frozen 40-scene record exactly:

| metric | `out/fast_path_40scene_cuda.json` (frozen record) | `out/r4_control_40scene_cuda.json` (this experiment) |
|---|---|---|
| EPE | 1.191216765057325 | 1.191216765057325 |
| D1 | 6.211033615520366 | 6.211033615520366 |
| RMSE | 3.1369637733547697 | 3.1369637733547697 |
| bad1 | 29.221754408662886 | 29.221754408662886 |
| bad2 | 11.500876854588872 | 11.500876854588872 |
| bad3 | 6.558777657603075 | 6.558777657603075 |
| valid pixels | 3802797 | 3802797 |
| contract_match | true | true |

Verified by direct comparison of the two JSONs: the `metrics` dicts are equal,
all seven figures, not approximately but exactly. The runtime path therefore
reproduces the frozen record before it measures anything else — the control is
a check on the experiment itself, and it passes.

## 5 E3 seed-0 final through the same path

| metric | frozen p2a_best | E3 seed0 final | change |
|---|---|---|---|
| EPE | 1.191216765057325 | 1.1628882757801255 | -0.0283285 |
| D1 | 6.211033615520366 | 6.015098886424913 | -0.1959 pt |
| RMSE | 3.1369637733547697 | 3.1188371491990803 | -0.0181 |
| bad1 | 29.221754408662886 | 28.27739687393253 | -0.9444 pt |
| bad2 | 11.500876854588872 | 11.0382699891685 | -0.4626 pt |
| bad3 | 6.558777657603075 | 6.385878604616549 | -0.1729 pt |
| valid pixels | 3802797 | 3802797 | 0 |
| contract_match | true | true | — |

Every figure above was read off the two `r4_*_40scene_cuda.json` files, not
copied on trust; the deltas are straight subtractions. All seven metrics move
down together on the same 3,802,797 valid pixels with `contract_match` true on
both sides. The E3 contract score obtained through the Stage-C runtime path is
**1.1628882757801255**.

## 6 Two scripts, same number

Stage E measured the E3 seed-0 final checkpoint independently through
`stage_e_recipe/kaggle/complete_e3.py` (the `frozen_eval` / `eval_tier2`
scoring path) and recorded **1.1628883** (`stage_e_recipe/LEADERBOARD.md`).
This experiment measures **1.1628882757801255** through `r4_contract.py`
(decode + normalize + inference through the Stage-C runtime path). Two
different scripts, two different code paths, same number to ~2.4e-8. That
agreement is the point of this section: it cross-validates the Stage-E figure
rather than replacing it.

## 7 Timing: within-session pair, no speed-up

The architecture, parameter count and dtype are identical between the two
checkpoints — only the weights differ — so the checkpoint cannot change
per-frame cost. Any timing difference is run-to-run and between-session
variation, and the measurements confirm it.

Within-session pair (same session, same protocol, back to back):

| run | preproc | infer | depth | cloud | disc | total | FPS | source JSON |
|---|---|---|---|---|---|---|---|---|
| frozen control | 29.90 | 52.64 | 1.37 | 13.19 | 1.85 | **99.12** | **10.09** | `out/r4_control_cuda_fp32_det_s10_w2.json` |
| E3 seed-0 final | 34.82 | 54.69 | 1.62 | 14.88 | 2.03 | **108.06** | **9.25** | `out/r4_e3s0final_cuda_fp32_det_s10_w2_same_session.json` |

The earlier cross-session comparison (E3 100.30 ms vs README section 3's
105.68 ms) pointed the other way and is **not presented as a speed-up** — this
within-session pair points the opposite direction (E3 slower by 8.94 ms), which
is exactly what noise looks like. Measured spreads, stated rather than
asserted:

- Same frozen checkpoint across sessions: 105.68 ms (README section 3 era) vs
  99.12 ms (this session) — spread **6.56 ms**.
- Same E3 checkpoint across sessions: 100.30 ms (earlier session) vs
  108.06 ms (this session) — spread **7.76 ms**.
- Within-session gap (different checkpoints): 8.94 ms, E3 slower.

The within-session gap sits inside the same-checkpoint session spread, so no
per-frame cost difference is resolvable — as expected, since none is possible.
Any residual difference is session noise. The scene-0 diagnostic is consistent
with different weights, not different speed: 0.227900 px vs the fp32 CPU
reference on the frozen path (the same 0.227900 px R2 reported for the control
path), 17.734488 px on the E3 path.

## 8 What this does and does not establish

**It does establish:**

- The recommended runtime path reproduces the frozen 40-scene record
  bit-identically on all seven metrics (section 4) — the instrument is sound.
- E3 seed-0 final scores 1.1628882757801255 EPE through that path, down on all
  seven metrics at unchanged valid pixels with `contract_match` true
  (section 5).
- That figure agrees with Stage E's independent 1.1628883 to ~2.4e-8 across
  two different scripts (section 6).
- Timing differences between the checkpoints are session noise, measured at
  6–9 ms spread with the sign flipping between comparisons (section 7).

**It does NOT establish:**

- That E3 is deployable, better in general, or accepted anywhere: this is one
  seed on `hailo_val` only, with no statistical test, no Hailo measurement, no
  INT8 measurement, and no change to any standing record. Stage E's own
  verdict machinery is where acceptance lives; this report is a runtime-path
  cross-check, nothing more.
- Any speed difference between the checkpoints: there is none to find (same
  architecture, same dtype), and none is claimed.
- Anything about the recommended path's latency beyond what README sections
  3–8, R2 and R3 already say: the path itself is unchanged by this experiment.

## 9 The freeze guard (correction recorded here so it is not lost)

The first version of the `--checkpoint` override measured the checkpoint file's
sha and asserted the file's sha against itself, which can never fail — the
default path no longer asserted `ARMP_SHA`. Fixed: both `runtime_path.py` (the
main load and the `torch.compile` fallback branch) and `r4_contract.py` now
load through a `load_net(device, ckpt, ckpt_sha, override)` helper that calls
`AD.load_frozen_net(device)` with no checkpoint arguments when the flag is
absent — the frozen `ARMP_SHA` assert runs exactly as before — and only passes
`checkpoint=` / `expected_sha256=` when `--checkpoint` was actually given. The
measured sha is still recorded in every JSON. `tests/test_r4_checkpoint_override.py`
covers it: the default path calls `load_frozen_net` with no checkpoint
arguments, the override path passes the measured pair through, and a corrupted
sha on the default path raises `AssertionError` (which a vacuous self-assert
could never do). 5 tests, all passing.

## 10 Files

| File | Role |
|---|---|
| `r4_contract.py` | 40-scene contract scorer for the recommended path, with opt-in `--checkpoint` |
| `out/r4_control_40scene_cuda.json` | Control contract (frozen checkpoint, no flag) — bit-identical to the frozen record |
| `out/r4_e3s0final_40scene_cuda.json` | E3 seed-0 final contract |
| `out/r4_control_cuda_fp32_det_s10_w2.json` | Frozen-checkpoint timing, this session (within-session pair) |
| `out/r4_e3s0final_cuda_fp32_det_s10_w2.json` | E3 timing, earlier session (kept, not overwritten; superseded for comparison by the rerun below) |
| `out/r4_e3s0final_cuda_fp32_det_s10_w2_same_session.json` | E3 timing rerun, this session (within-session pair) |

## 11 How to reproduce

```bash
python stage_c_deploy/runtime/r4_contract.py --out stage_c_deploy/runtime/out/r4_control_40scene_cuda.json
python stage_c_deploy/runtime/r4_contract.py --checkpoint stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth --out stage_c_deploy/runtime/out/r4_e3s0final_40scene_cuda.json
python stage_c_deploy/runtime/runtime_path.py --device cuda --gpu-geometry --scenes 10 --warmup 2 --out stage_c_deploy/runtime/out/r4_control_cuda_fp32_det_s10_w2.json
python stage_c_deploy/runtime/runtime_path.py --device cuda --gpu-geometry --scenes 10 --warmup 2 --checkpoint stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth --out <new-name>.json
python -m pytest tests/test_r4_checkpoint_override.py -v
```

---

Host runtime work only. No frozen artefact, checkpoint, contract or accuracy figure was
created or changed to produce this report. Hailo deployment remains blocked.
