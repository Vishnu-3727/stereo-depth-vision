# Stage A — RESULTS

Diagnostic run `20260909T022238Z`. Four one-epoch runs of the H2 seed-0
configuration (= E3b arm A) plus one operator-level probe. No training
experiment, no architecture change, no historical record altered.

---

## 0. Provenance audit

`MEASURED`:

| question | answer |
|---|---|
| commit referenced by `phase-1-frozen` | **`b4207e518166f5089731b4d037a4a79234f70986`** (`b4207e5`) |
| `git rev-parse phase-1-frozen` returns | `0fc4f296b409f79bc12cd8e294a1bdb34ed1e262` |
| why the two differ | `phase-1-frozen` is an **annotated** tag (`git cat-file -t` → `tag`). `rev-parse` prints the *tag object*; `rev-parse phase-1-frozen^{commit}` dereferences it to `b4207e5`. `git diff phase-1-frozen …` dereferences automatically. **No discrepancy in the frozen state — a reader comparing the two sha values would see one where none exists.** Reported, not repaired. |
| `HEAD` | `58e8a19` — one commit ahead of the tag, `docs: add the Phase 1 overview`, adding `PHASE_1_OVERVIEW.md` only (`git diff --stat phase-1-frozen HEAD` = 1 file, +263 lines) |
| any Phase 1 file differing from the frozen tag | **no** — `git diff phase-1-frozen -- src scripts` is empty, before and after this diagnostic |
| uncommitted work in the tree | yes: `M .gitignore` (9 added ignore lines) and `?? phase2/` |
| is Phase 2 tracked | **NO. `git ls-files phase2` is empty — the entire Phase 2 tree is untracked.** Every script, preregistration, report and experiment record is unversioned |

`DERIVED` — Phase 1's freeze is cryptographically verifiable. **Phase 2's
"frozen before the run" claims are not**: no commit, tag or content hash binds
any preregistration to a time before the run it governs. The experiment records
are self-documenting about it (`env.json` → `dirty_files: ["M .gitignore",
"?? phase2/"]`), so the gap is visible rather than hidden, but it is real.

`INFERRED` — committing the Phase 2 tree is the single cheapest provenance
repair available. **It was not done here**: committing is a repository-wide
action outside this diagnostic's scope, and Stage A's brief is to characterise,
not to change. Recommended to the operator, not executed.

---

## 1. Are the two runs identical at batch 0?

## **YES.**

`MEASURED` — RUN_A and RUN_B, two fresh processes, default flags:

| quantity | value (both runs) |
|---|---|
| initial weights, concatenated sha | `c4d02385e3da28a5` |
| torch CPU / CUDA / numpy state after `build_model` | `96616bad…` / `374708ff…` / `129790f8…` |
| batch-0 sample indices | 111, 54 |
| batch-0 scenes | `000111_10.png`, `000054_10.png` |
| batch-0 crops | (y=96, x=459) and (y=8, x=11) |
| batch-0 gain jitter | −0.013210486329130189 / +0.06404226504432821, −0.0535669373161111 / +0.03615950549094848 |
| batch-0 left / right / disparity tensor sha | `b87563da2698e20c` / `947f0d2c884b6275` / `c2cbb4218344e505` |
| batch-0 forward output sha | `9ac6c0574d69a414` |
| **batch-0 loss** | **17.472463607788086** |
| **data order over the whole epoch (160 items)** | **identical; first difference: none** |

`DERIVED` — the data and RNG pipeline is **not** the source of divergence.
`CroppedKitti` owns a private `np.random.default_rng(seed)`, the DataLoader
sampler draws from the torch CPU stream that `build_model` pins with
`torch.manual_seed(0)`, and `num_workers=0` means no worker seeding is involved.
All of it reproduces.

---

## 2. If not, what differs?

Not applicable — batch 0 is identical. One incidental difference exists and is
recorded for completeness:

- `MEASURED` — python's stdlib `random` module state differs between processes.
  It is **never seeded** by the harness. `DERIVED` — it is also never consumed:
  the identical data order, crops, jitter values and batch-0 tensors prove it
  takes no part in this pipeline. It is a latent hazard for any future code that
  starts using `random`, not a cause of anything observed here.

---

## 3. If batch 0 is identical, when does divergence first appear?

## **At the batch-0 backward pass — the gradient, not the forward.**

`MEASURED`, RUN_A vs RUN_B:

| batch | first differing quantity | |Δ| |
|---:|---|---:|
| 0 | forward output | **0** (sha equal) |
| 0 | loss | **0** |
| **0** | **grad norm** | **3.05e-05 ← earliest observed difference** |
| 1 | loss | 4.77e-06 |
| 2 | loss | 5.86e-04 |
| 5 | loss | 5.52e-03 |
| 20 | loss | 2.97e-02 |
| 40 | loss | 1.63e-01 |
| 60 | loss | 7.94e-01 |
| 79 | loss | **1.07** |

`MEASURED` — the in-process repeat probe isolates it further, with no
cross-process variable at all. Same weights, same batch, same process:

| probe | RUN_A | RUN_B | RUN_C/D (deterministic) |
|---|---|---|---|
| forward ×2 | **bitwise identical** (0.000e+00) | **bitwise identical** | identical |
| backward ×2 | **differs**, max 8.297e-05, **309,529** elements | **differs**, max 5.150e-05, 307,640 elements | **identical, 0 elements** |

`DERIVED` — the forward pass of this architecture is bit-reproducible. **The
backward pass is not.** Divergence is created at the very first optimizer step
and then compounds monotonically through the epoch.

`MEASURED` — after one epoch: epoch mean loss differs by 2.70e-02, the trained
weight hashes differ (`eda467853cd05550` vs `6f437e1198893e43`), and validation
differs by **+1.253619 px EPE** and **−1.235660 D1 points** — both already
larger than the E3b/E3c materiality band after a *single* epoch.

`MEASURED` — validation itself is not a noise source: repeating `validate` on
identical weights in the same process returns bit-identical numbers (ΔEPE 0.0,
ΔD1 0.0) in all four runs.

### Where in the network

`MEASURED` (`locate_op/locate_op.json`) — gradient divergence between two
in-process backward passes, by stage:

| stage | parameters | elements differing | max abs Δ | max Δ relative to that stage's largest gradient |
|---|---:|---:|---:|---:|
| feature extractor | 199,552 | **199,442** | 1.433e-05 | **2.65e-05** |
| aggregation | 111,585 | **109,233** | 4.625e-05 | 3.90e-07 |
| refinement | 112,449 | 150 | 7.153e-07 | 1.62e-07 |

`DERIVED` — divergence is concentrated **upstream of refinement**, in exactly
the two stages whose gradient must travel back through the full-resolution
bilinear upsample of the cost tensor. In relative terms the **matching path is
the worst-affected part of the network** — the path that carries the stereo
learning signal.

---

## 4. Is the dominant cause RNG/DataLoader or GPU/kernel?

## **COMPUTATIONAL (GPU/kernel) NONDETERMINISM — in the backward pass.**

Not asserted from "CUDA is involved". The attribution rests on four independent
`MEASURED` facts:

1. Data, RNG streams, initial weights, forward output and batch-0 loss are
   bit-identical across processes (§1).
2. Repeating the **forward** twice inside one process is bit-identical; repeating
   the **backward** twice inside one process is not (§3). No RNG, no process
   boundary, no data variable is present in that comparison.
3. **Isolating the suspected kernel reproduces the effect on its own.**
   `F.interpolate(..., mode="bilinear", align_corners=True)` from the model's own
   cost-tensor shape `(2, 12, 23, 77) → (256, 512)` — the upsample the
   standardised soft-argmin consumes — backward twice on identical input:
   **not bitwise identical**, 36,612 elements differ, max abs Δ 1.831e-04. With
   determinism controls on: **bitwise identical, max abs Δ 0.0.**
4. Turning on `use_deterministic_algorithms(True)` + `cudnn.deterministic` +
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes two fresh processes agree on
   everything, down to the trained weight hash (§5).

`INFERRED` — the mechanism is non-associative floating-point accumulation in a
scatter-style backward (`atomicAdd`-based gradient accumulation in the bilinear
upsample backward), whose summation order varies with thread scheduling. The
diagnostic measures the *behaviour* and the *fix*; the specific CUDA kernel
implementation was not inspected, so the mechanism is `INFERRED`, the
non-reproducibility and its removal are `MEASURED`.

`DERIVED` — this is a property of the architecture's own critical path, not of
an incidental library call. This model upsamples the entire 12-channel cost
tensor to full resolution before the soft-argmin (Phase 1,
`regression.py` — the deployed graph's own unusual ordering), so the
nondeterministic kernel sits directly between the matching path and the loss.

---

## 5. What deterministic controls were tested, and their effects?

| control | tested | result |
|---|---|---|
| `torch.use_deterministic_algorithms(True)` | yes | **accepted, no error**; `determinism_setup_error: null` |
| `torch.backends.cudnn.deterministic = True` | yes | applied |
| `torch.backends.cudnn.benchmark = False` | yes | already the default in this environment |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | yes | set before CUDA init; required — the `warn_only` pass named cuBLAS as the one kernel class that would otherwise stay nondeterministic |
| seeded DataLoader generator / worker seeding | **not needed** | `num_workers=0`, and §1 shows the data pipeline already reproduces exactly |

`MEASURED` — **no operation in this architecture refused to run under
deterministic algorithms.** `nondeterministic_op_error: null` in both controlled
runs; the epoch and both validations completed.

`MEASURED` — the one warning PyTorch emits under `warn_only=True` is the cuBLAS
workspace warning, quoted verbatim in `locate_op/locate_op.json`. Setting
`CUBLAS_WORKSPACE_CONFIG` clears it.

`MEASURED` — effect, two fresh processes with controls on:

| quantity | RUN_C | RUN_D |
|---|---:|---:|
| batch-0 loss | 17.472482681274414 | 17.472482681274414 |
| epoch mean loss | 10.532406491041183 | 10.532406491041183 |
| median grad norm | 54.79760932922363 | 54.79760932922363 |
| weights after epoch, sha | `356da4325d74ef4d` | `356da4325d74ef4d` |
| validation EPE | 15.32539188316842 | 15.32539188316842 |
| validation D1 | 87.44677261821234 | 87.44677261821234 |
| first differing batch (loss / grad) | none | none |

`DERIVED` — **ΔEPE = 0.000000, ΔD1 = 0.000000, identical trained weights.**
Full bit-for-bit reproducibility is achievable in this environment.

`MEASURED` — cost: 42.0 s / 41.2 s per epoch against 36.1 s / 37.5 s,
i.e. **≈ +13 %** wall clock; `DERIVED` ≈ +8 minutes on a 200-epoch run.

`MEASURED` — controls also change the numbers (RUN_A 17.497 px vs RUN_C
15.325 px after one epoch, and batch-0 loss differs in its last bits).
`DERIVED` — deterministic mode picks different kernels, so a controlled run is a
different draw from the same distribution, **not** a corrected version of an
uncontrolled one. Historical uncontrolled runs cannot be retro-fitted; they can
only be superseded by new controlled ones.

---

## 6. What remains unknown

1. `UNKNOWN` — **the size of the divergence at 200 epochs under default flags.**
   This diagnostic measured one epoch (ΔEPE 1.25 px, ΔD1 1.24 pt). The only
   200-epoch same-config sample in the project is E3c arm D vs O6 D124
   (ΔEPE 0.343, ΔD1 1.970). Whether early divergence grows, saturates or
   partially cancels as the cosine schedule decays is not measured. **Do not
   assume the one-epoch figure is the 200-epoch figure in either direction.**
2. `UNKNOWN` — the *distribution* of same-config outcomes. Two default-flag runs
   give one difference, not a variance.
3. `UNKNOWN` — whether any second-order source (thermal/clock throttling, other
   GPU load, allocator state) contributes. Not varied here; both runs were
   minutes apart on an otherwise idle machine.
4. `UNKNOWN` — whether determinism controls change the *converged* solution
   systematically. One epoch cannot tell; the A-vs-C gap is inside the chaotic
   early regime and no such claim is made.
5. `UNKNOWN` — the exact CUDA kernel and its accumulation order. Behaviour and
   remedy are measured; the implementation was not read.
6. `UNKNOWN` — whether any *other* nondeterministic kernel also contributes.
   The bilinear-upsample backward is proven nondeterministic in isolation and
   the stage attribution is consistent with it, but the diagnostic did not
   exhaustively test every operation in the graph.

---

## 7. What should the next experiment do?

**One recommendation, strictly from the evidence above.**

> **Re-measure the noise band under determinism controls: N ≥ 3 full 200-epoch
> runs of the six-block H2/E3b-arm-A configuration, seed 0, with
> `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
> `cudnn.benchmark=False` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, plus one
> uncontrolled 200-epoch repeat of the same configuration for contrast.**

Why exactly this, and why nothing else first:

- `MEASURED` — controls give bit-identical results at one epoch, so the
  controlled arm has a **falsifiable prediction**: the three runs must return
  identical weight hashes at 200 epochs. If they do, the noise band under
  controls is **exactly zero**, architecture comparisons need no band at all,
  and every future arm becomes a clean single-run measurement.
- If they do **not** agree at 200 epochs, that is itself the finding — a second
  nondeterminism source exists that one epoch could not surface — and the spread
  across the three runs is then the first honest, multi-sample noise band this
  project has had.
- The single uncontrolled repeat is what converts §6.1 from `UNKNOWN` into a
  measurement, and it is the only way to say what the *historical* E3b/E3c/O6
  band actually was, since those runs cannot be re-created.
- Cost: 4 runs × ≈4,100 s ≈ **4.6 GPU-hours**. No architecture is changed, no
  hypothesis is tested, nothing is tuned.

Explicitly **not** recommended next, and not run here: any block-count arm, any
dilation arm, E2b, Scene Flow, or a rerun of E3b/E3c/O6. Their comparisons
cannot be interpreted until the band is real.

`INFERRED`, offered as a note rather than an instruction — before that run, the
operator may wish to commit the Phase 2 tree (§0), so the new protocol is the
first part of Phase 2 with verifiable provenance.

---

### VERDICT

## `COMPUTATIONAL-NONDETERMINISM-IDENTIFIED`

### EVIDENCE

- `MEASURED` — two fresh same-seed processes are **identical at batch 0**:
  same initial weights (`c4d02385e3da28a5`), same scenes (111 / 54), same crops
  ((96,459) / (8,11)), same jitter, same input tensor hashes, same forward output
  hash, same loss `17.472463607788086`, and an **identical data order across all
  160 items** of the epoch.
- `MEASURED` — the **earliest** difference is the batch-0 **gradient norm**
  (308.035553 vs 308.035583, Δ 3.05e-05); losses are still equal at that point.
- `MEASURED` — inside a **single** process, forward ×2 is bitwise identical
  (0.000e+00) while backward ×2 is not (309,529 gradient elements differ,
  max 8.297e-05). No RNG or process boundary is present in that comparison.
- `MEASURED` — the suspected kernel reproduces it in isolation:
  `F.interpolate` bilinear backward on the model's own cost-tensor shape
  `(2,12,23,77) → (256,512)` differs across repeats (36,612 elements,
  max 1.831e-04) and is bitwise identical once determinism controls are on.
- `MEASURED` — divergence is concentrated upstream of refinement: 199,442 of
  199,552 feature-extractor gradient elements and 109,233 of 111,585 aggregation
  elements differ, against 150 of 112,449 in refinement.
- `MEASURED` — it compounds: |Δloss| grows 0 → 4.8e-06 → 5.9e-04 → 2.9e-02 →
  1.07 across batches 0, 1, 2, 20, 79, ending in **ΔEPE +1.254 px and
  ΔD1 −1.236 points after one epoch** — already outside the E3b/E3c band.
- `MEASURED` — with `use_deterministic_algorithms(True)`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, **no operation refused to run** and two
  fresh processes agree exactly: same weight hash `356da4325d74ef4d`,
  ΔEPE 0.000000, ΔD1 0.000000, for ≈ +13 % wall clock.
- `MEASURED` — evaluation is not implicated: repeating `validate` on identical
  weights returns bit-identical EPE and D1 in all four runs.

### IMPACT ON E3b/E3c

**Weakened — and specifically, the materiality band they were judged against is
now known to be unsound; it is not established that their architectural
conclusions are wrong.**

- `MEASURED` here — training under the flags E3b and E3c ran with is not
  reproducible, and one epoch alone produces spreads (1.25 px / 1.24 pt) larger
  than the band those experiments used (0.2138 px / 1.0000 pt).
- `DERIVED` — E3b's band, being a single sample of an unstable quantity, cannot
  support either an equivalence claim (`CAPACITY-REDUCIBLE`) or a difference
  claim. Both directions are unsupported; the correct standing is
  **indeterminate**, pending §7.
- `DERIVED` — O6's reproducibility observation is **corroborated and its cause is
  now identified rather than inferred**. O6 attributed the divergence to GPU
  nondeterminism as an `INFERRED` reading; this diagnostic supplies the
  measurement, and additionally rules out the competing RNG/DataLoader
  explanation that a 2.8 % epoch-0 loss difference in the seed-replication
  self-check had left open.
- **No historical record was changed.** E3b, E3c and O6 verdicts, metrics,
  preregistrations and reports are untouched. This directory is the new,
  additive statement.
- `UNKNOWN` — how large the historical band actually was at 200 epochs. §7's
  uncontrolled repeat is what would measure it.

### NEXT EXPERIMENT

**N ≥ 3 same-config 200-epoch runs of the six-block H2 / E3b-arm-A
configuration with determinism controls on, plus one uncontrolled 200-epoch
repeat of the same configuration.** ≈4.6 GPU-hours. It tests the falsifiable
prediction that the controlled runs are bit-identical at 200 epochs, and it
produces the project's first multi-sample noise band — controlled and
uncontrolled — against which any future architecture comparison can be judged.

**Not executed in this task.**
