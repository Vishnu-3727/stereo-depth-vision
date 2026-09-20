# RELATED RUNS — EXP-CORRESPONDENCE-GEOM-A (Option A)

**Design artifacts only. The experiment has NOT been executed.** No `RESULTS.md`,
no `results.json`, no `ENVIRONMENT.txt`, no `run.log`. No checkpoint was opened
in this record: `select_scenes.py` reads ground truth only. Every record below
is preserved exactly as produced; **none was modified, re-run, re-scored or
deleted.**

Second record under `phase2/diagnostics/correspondence_geom/`. Stage 1
(BUILD AND FREEZE ONLY) of `EXP-CORRESPONDENCE-GEOM-003`, Option A.

---

## Lineage

```
EXP-CORRESPONDENCE-GEOM-001
  phase2/diagnostics/correspondence_geom/20260911T111809Z/
  (executed; verdict stands as recorded there; NOT modified here)
        |
        |  successor_audit_20260911/AUDIT.md derives, from GT only, that the
        |  symmetric-translation statistic class cannot be rebuilt at adequate
        |  strength (±48 unreachable anywhere; at most 2 signed offsets), and
        |  SUCCESSOR_OPTIONS.md assesses Option A as defensible but
        |  underpowered, weaker than the experiment it replaces, and NOT the
        |  recommended successor (Option B was recommended).
        v
EXP-CORRESPONDENCE-GEOM-003, Stage 1, Option A                 (this record)
  phase2/diagnostics/correspondence_geom/20260914T070551Z/
  GEOMETRIC-DESIGN-STAGED — preregistered, NOT executed, freeze INCOMPLETE
  (scene-selection shortfall: 2 of the required 4 scenes; see PREREGISTRATION.md)
        |
        |  direction_decision_20260913T052115Z/DECISION.md section 1 marks the
        |  image-translation slope class RETIRED/REJECTED and the vertical-shift
        |  null INVALID. Both markings post-date the Option A design and are
        |  accepted into this record's KNOWN LIMITATIONS before execution.
        v
  (no execution; no successor authorised by this record)
```

---

## 1. GEOM-001 — the experiment this design descends from

**Record:** `phase2/diagnostics/correspondence_geom/20260911T111809Z/`

GEOM-001 supplies, unchanged: the four checkpoints (sha256 copied verbatim
into `run_geomA.py`), the crop-based translation mechanism (no fill, no
padding, no interpolation), the odd/even statistic structure, the all-units
decision discipline (12/12, no partial counts), and the freeze discipline
(file-byte sha256 re-verified before the first model instantiation).

GEOM-001 also supplies the GT pipeline this record re-uses for scene
selection — validated here by bit-exact reproduction of its four frozen mask
counts (34245 / 46199 / 33250 / 50747) from GT alone before any new
measurement was trusted.

**Not modified, not reinterpreted.**

---

## 2. Successor audit — the design authority for Option A

**Record:** `phase2/diagnostics/correspondence_geom/successor_audit_20260911/`

- `AUDIT.md` §1 (representability band), §4 (maximum symmetric range: at most
  2 signed offsets; C2 monotonicity degenerates to a coin flip), §8 (failure
  modes, esp. candidate-axis clipping, vertical-control contamination,
  designer knowledge).
- `SUCCESSOR_OPTIONS.md`, OPTION A section: the frozen design this record
  implements without redesign and without improvement.

The audit's MEASURED figures this design depends on: `[2,9]` retention
≥ 50 % on 17 of 40 scenes at border 32; `[3,8]` on 0 of 40; 36 of 40 scenes
held out. The audit's figures were measured at border 32; this record's
mandated border is 96, at which the audit's figures do not carry over —
the shortfall measured in Stage 1 (see PREREGISTRATION.md).

**Not modified.**

---

## 3. Direction decision — post-design, binding on interpretation

**Record:** `phase2/diagnostics/correspondence_geom/direction_decision_20260913T052115Z/DECISION.md`

Section 1 (Falsified / retired) retires the image-translation slope class and
invalidates the vertical shift as a null. Both bullets are quoted verbatim in
PREREGISTRATION.md (KNOWN LIMITATIONS) and are binding on any reading of a
future result: the vertical arm is a CONTAMINATED comparison, never a valid
null.

**Not modified.**

---

## 4. Shared provenance

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002, ARCH-001, GEOM-001, **GEOM-A (planned)** |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002, ARCH-001, GEOM-001, **GEOM-A (planned)** |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | same |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | same |

No checkpoint loaded by this design task. None has ever been retrained,
modified or added in this campaign; no extra trained seed has ever been
introduced. GEOM-A adds **no** new weights of any kind.

---

## 5. Source files read for this design (read-only)

| file | what it established |
|---|---|
| `src/models/stereonet/cost_volume.py` | `V = Lf[u+k] − Rf[u]`; `shift="none"` is a no-op |
| `src/models/stereonet/feature_extractor.py` | fully convolutional, total stride 16, no pooling/normalisation/global op |
| `src/models/stereonet/regression.py` | `soft_argmin` over the index grid `0…11` ⇒ candidate units |
| `phase2/models/scaled_regression.py` | readout has **0 parameters** |
| `src/datasets/kitti2015.py` | GT scale 256.0 ⇒ full-resolution px; hailo_val = 40 scenes |
| `phase2/viz/core.py` | `correspondence(x_L, d) = x_L − d`, photometrically confirmed |

`src/` is bit-identical to the frozen Phase-1 tag and was not touched.

---

## 6. Standing state of the claim

Unchanged by this record: geometric correspondence (Level D) is
NOT-DEMONSTRATED — never tested on non-degenerate weights by an admissible
measurement. This record stages, but does not execute, one such attempt, and
records before execution both its accepted limitations and the scene-selection
shortfall that currently blocks its freeze.
