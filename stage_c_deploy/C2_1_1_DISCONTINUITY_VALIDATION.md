# C2.1.1 Discontinuity Validation Gate — evidence-only close-out of the C2.1 gap

Claim labels used throughout: **VERIFIED** (measured / asserted by the
validation run in
`stage_c_deploy/spatial_perception/out/c2_1_1_discontinuity_validation.json`),
**INFERRED** (follows from code inspection or arithmetic, not directly
measured), **UNKNOWN** (not measured, stated as such), **NOT APPLICABLE**.

## 1 Objective

C2.1 passed 251/251 over 40 scenes (**VERIFIED**: pre-existing record in
`stage_c_deploy/spatial_perception/out/c2_1_validation.json`, not re-run here).
Its ONE gap: `discontinuity.py` had no explicit named real/synthetic
correctness gate inside a validator — only its own module self-check. This task
closes that gap with evidence, not features: one new validator, one JSON
record, this report. The discontinuity algorithm itself is FROZEN and was not
touched (**VERIFIED**: SHA256 of `discontinuity.py` identical before and after,
`2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743`).

## 2 Existing Implementation

`stage_c_deploy/spatial_perception/discontinuity.py` (frozen, unchanged):
`_forward_mag` computes FORWARD differences `du[v,u] = |F[v,u+1] - F[v,u]|`,
`dv[v,u] = |F[v+1,u] - F[v,u]|`; output magnitude `= hypot(du, dv)` defined ONLY
where both differences are valid — i.e. the last row and last column are always
invalid (mask False), by design. A difference needs BOTH its pixels valid; one
invalid pixel poisons exactly the differences touching it. `depth_discontinuity`
and `disparity_discontinuity` are the same operator on depth (m/px) and
disparity (px/px); the disparity twin defaults to finite-pixel validity and
accepts a C2-style mask (`d > 1e-3`) to mirror C2's invalid rules. No smoothing
of any kind (**VERIFIED**: code inspection — no smoothing call exists in the
module; the validator hashes the module before and after to prove no change).
**Off-by-one is the convention, not a bug** (**INFERRED** from the definition):
for a vertical step where the value changes between column `k-1` and column
`k`, the magnitude appears at column `k-1` with value equal to the step size.

## 3 Why This Gate Was Needed

C2.1's validator exercised the discontinuity map only indirectly (valid-pixel
counts on real data); the module's hand-derived expectations lived only in its
own `demo()` self-check, which is not a named gate, writes no record, and runs
outside any validator contract. This gate promotes those hand-derived cases —
plus horizontal, constant, multi-region, poisoning, disparity-twin, and real
KITTI sanity checks — into `validate_discontinuity.py`: named checks, printed
evidence, a JSON record, nonzero exit on failure (**VERIFIED**: the run below
exits 0 with 19/19 passing).

## 4 Synthetic Test Design

All expected values are hand-derived literals from the §2 convention; none is
produced by calling `depth_discontinuity` / `_forward_mag` / `disparity_discontinuity`
(**VERIFIED** by inspection of the validator source — expectations are numeric
literals and `math.hypot(4.0, 9.0)` arithmetic only):

1. **Vertical boundary**: 6×10, 10.0 left of column k=5, 30.0 right.
2. **Horizontal boundary**: 8×6, 10.0 above row k=5, 30.0 below.
3. **Constant depth**: 5×7 of 12.5.
4. **Multiple regions**: 6×8 with TL=10.0, TR=14.0, BL=19.0, BR=23.0 —
   vertical step 4 (cols 3|4), horizontal step 9 (rows 2|3); corner pixel (2,3)
   has du=|14−10|=4 and dv=|19−10|=9, so hypot(4,9)=sqrt(97)≈9.848857801796104
   derived arithmetically in the test.
5. **Invalid-pixel poisoning**: 4×5 of 7.0 with pixel (1,2) invalid (NaN).
   Hand-derived: du_ok false at (1,1),(1,2); dv_ok false at (0,2),(1,2); the
   magnitude mask on [:-1,:-1] is therefore False exactly at (0,2),(1,1),(1,2)
   plus the last row/column.
6. **Disparity twin**: the step field re-run through `disparity_discontinuity`
   with default validity, plus a C2-style mask case (`d[0,0]=0`,
   `valid = d > 1e-3`): hand-derived mask[0,0]=False (both touching differences
   poisoned), mask[1,1]=True, borders False.

Tolerance rule (**INFERRED** from float64 arithmetic, stated here): the
constant/step cases use exact equality (`==`) because they are exact float64
operations on exactly-representable values (10.0, 30.0, 20.0, 4.0, 9.0, 0.0);
only the corner `hypot(4,9)=sqrt(97)` comparison uses `<= 1e-12`, because
sqrt(97) is not exactly representable. No loose blanket tolerance anywhere.

## 5 Expected Results

| Test | Expected |
|---|---|
| vertical boundary | mag == 20.0 at col 4 (all valid rows), 0.0 at every other valid column, mask True exactly on [:-1,:-1] |
| horizontal boundary | mag == 20.0 at row 4 (all valid cols), 0.0 elsewhere valid, mask True exactly on [:-1,:-1] |
| constant depth | mag == 0.0 everywhere mask True (24 pixels), mask True exactly on [:-1,:-1] |
| multiple regions | mag[2,3] == hypot(4,9) ≈ 9.848857801796104 (≤1e-12); mag[0,3]==4.0; mag[2,0]==9.0; mag[0,0]==0.0; mag[2,4]==9.0; mask == [:-1,:-1] |
| invalid poisoning | mask False exactly at (0,2),(1,1),(1,2) + last row/col; True at (0,0),(2,3) |
| disparity twin | step case identical to depth twin; C2-mask case mask[0,0]=False, mask[1,1]=True |
| real KITTI sanity | shape == input; finite where mask True; bitwise-identical repeats; inputs unmodified; non-empty map (descriptive fraction only) |
| determinism | two calls bitwise-identical (mag with equal_nan, mask exact) |
| frozen ARM-P unchanged | checkpoint SHA256 == `b2f6f5…feb7454` before and after |
| C2 output unchanged | max\|depth − fB/d\| == 0 with identical NaN pattern on the sanity scene |

## 6 Observed Results

Validator output (**VERIFIED**, pasted verbatim from the run):

```text
C2.1.1 discontinuity validation: 19/19 passed; overall=PASS; wrote C:\Users\vishn\stereo_depth_vision\stage_c_deploy\spatial_perception\out\c2_1_1_discontinuity_validation.json
```

```text
[PASS] synthetic vertical boundary (step at col k, magnitude at k-1) :: A=10.0 B=30.0 k=5: expected mag==20.0 at col 4 (got 20.0), 0.0 at all other valid cols (left_zero=True right_zero=True); mask==[:-1,:-1] exactly: True
[PASS] synthetic horizontal boundary (step at row k, magnitude at k-1) :: A=10.0 B=30.0 k=5: expected mag==20.0 at row 4 (got 20.0), 0.0 elsewhere valid; mask==[:-1,:-1] exactly: True
[PASS] synthetic constant depth (magnitude 0 where valid) :: expected 0.0 everywhere mask True, mask True exactly on [:-1,:-1]; got mag[0,0]=0.0, valid_count=24 (expected 24)
[PASS] synthetic multiple regions (corner hypot(step_u, step_v)) :: expected mag[2,3]==hypot(4,9)=9.848857801796104 (got 9.848857801796104, |diff|=0.000e+00 <= 1e-12); mag[0,3]==4.0 (got 4.0); mag[2,0]==9.0 (got 9.0); mag[0,0]==0.0 (got 0.0); mag[2,4]==9.0 (got 9.0); mask==[:-1,:-1]: True
[PASS] synthetic invalid-pixel poisoning (exact false indices) :: invalid pixel (1,2) -> expected mask False exactly at (0,2),(1,1),(1,2) plus last row/col; full-mask equality: True; spot (000 vs 111/102 False, 000/233 True): True
[PASS] disparity twin (same operator, incl. C2-style d>1e-3 mask) :: step case identical to depth twin: True (mag[0,4]=20.0); C2-mask d[0,0]=0 excluded -> mask[0,0]=False (True), mask[1,1]=True (True), borders False: True
```

Result table (every row backed by the pasted evidence above or in §7–§10):

| Test | Expected | Observed | Status |
|---|---|---|---|
| vertical boundary | 20.0 at col 4, 0.0 elsewhere, mask [:-1,:-1] | got 20.0 / zeros / exact mask | PASS (**VERIFIED**) |
| horizontal boundary | 20.0 at row 4, 0.0 elsewhere, mask [:-1,:-1] | got 20.0 / zeros / exact mask | PASS (**VERIFIED**) |
| constant depth | 0.0 × 24 valid | mag[0,0]=0.0, valid_count=24 | PASS (**VERIFIED**) |
| multiple regions | hypot(4,9) at (2,3) + edge spots | \|diff\|=0.000e+00; 4.0/9.0/0.0/9.0 as expected | PASS (**VERIFIED**) |
| invalid-pixel poisoning | mask False exactly at (0,2),(1,1),(1,2)+borders | full-mask equality True | PASS (**VERIFIED**) |
| disparity twin | same operator + C2-style mask | step identical; mask[0,0]=False, mask[1,1]=True | PASS (**VERIFIED**) |
| real KITTI sanity | shape/finite/untouched/non-empty | 368×1232; finite; untouched; frac>1.0 m/px = 0.097931 (descriptive) | PASS (**VERIFIED**) |
| determinism | bitwise-identical repeats | mag equal_nan + mask exact | PASS (**VERIFIED**) |
| frozen ARM-P unchanged | SHA256 match before+after | `b2f6f5…feb7454` both | PASS (**VERIFIED**) |
| C2 output unchanged | max\|depth−fB/d\|==0 | 0.000e+00, NaN-pattern identical | PASS (**VERIFIED**) |

The algorithm needed NO change: every hand-derived expectation matched on the
first run, so per the task rule nothing was tuned and no tolerance was widened
(**VERIFIED**: `discontinuity.py` hash unchanged, §10).

## 7 Real KITTI Sanity Check

Scene `000160_10.png` (one scene only; the 40-scene run already exists, so this
is deliberately cheap). Labelled SANITY, never ACCURACY — KITTI provides no
boundary ground truth here (**VERIFIED** as a scope fact: no boundary labels
were loaded or compared):

```text
[PASS] real KITTI sanity: shape == input, finite where mask True :: scene 000160_10.png: in=(368, 1232) out=(368, 1232) shape_ok=True; finite-where-valid=True (valid=451777/453376)
[PASS] real KITTI sanity: inputs not modified (bitwise-identical) :: depth and disparity arrays bitwise-identical after call: True
[PASS] real KITTI sanity: map non-empty where real transitions exist (descriptive, not a threshold) :: DESCRIPTIVE (not correctness): fraction of valid pixels with mag>1.0 m/px = 44243/451777 = 0.097931; mag[valid] min=0.000428 max=34.230391 mean=0.392399
```

The 1.0 m/px threshold is a DESCRIPTIVE statistic, not a correctness gate: the
check passes on non-emptiness (44243 > 0), and the fraction 0.097931 carries no
pass/fail meaning (**VERIFIED**: the validator asserts `nonempty`, never
`frac > x`).

## 8 Determinism

```text
[PASS] real KITTI sanity: deterministic across repeats (bitwise-identical) :: two calls bitwise-identical (equal_nan): mag=True mask=True
```

Two consecutive `depth_discontinuity` calls on the real scene are
bitwise-identical (**VERIFIED**). Inference determinism and spatial
determinism over all 40 scenes remain covered by the pre-existing C2.1 record
(**VERIFIED**, not re-run).

## 9 Resource Measurement

DEVELOPMENT-MACHINE MEASUREMENT (never Hailo performance), discontinuity
operation ALONE — excluding inference and the rest of the spatial layer
(**VERIFIED**):

```text
[PASS] resource: discontinuity-only runtime and memory measured :: DEVELOPMENT-MACHINE MEASUREMENT: 21.037 ms/frame over 20 reps (scene 000160_10.png, (368, 1232)); tracemalloc peak for one call=23990224 bytes; output buffers=4080384 bytes (mag 3627008 + mask 453376)
```

Per-frame runtime ≈ 21 ms on this machine (LOW by the C2.1 scale: 5–100 ms);
measurable additional memory ≈ 4.1 MiB output buffers plus ≈ 23.99 MiB
tracemalloc transient peak for one call. No Hailo claim, no neural compute, no
accelerator move (**NOT APPLICABLE** — nothing was compiled, quantized, or
benchmarked on device).

## 10 Artifact Integrity

- Checkpoint SHA256 before = after =
  `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`
  (**VERIFIED**).
- ONNX SHA256 on disk before = after =
  `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989`
  (**VERIFIED**); the task prompt AGAIN carries the transposed typo
  `...deed1dce26...` — disk is authoritative, recorded as a documentation
  discrepancy, nothing fixed, nothing renamed.
- `discontinuity.py` SHA256 before = after = `2e81e9…1b5743` — the algorithm
  was not modified (**VERIFIED**).
- C2 depth unchanged on the sanity scene: `max|depth − fB/d| = 0.000e+00`,
  NaN-pattern identical (**VERIFIED**).
- Prior artifacts preserved: `c2_1_validation.json` exists, both
  `*_c2_1_spatial.png` figures present; this task wrote only
  `validate_discontinuity.py`, this report, and
  `out/c2_1_1_discontinuity_validation.json` (**VERIFIED**). The C2/C2.1
  validation artifacts were not re-run or overwritten.
- Scope check (**VERIFIED** by construction): no file outside
  `stage_c_deploy/spatial_perception/validate_discontinuity.py`,
  `stage_c_deploy/C2_1_1_DISCONTINUITY_VALIDATION.md`, and
  `stage_c_deploy/spatial_perception/out/c2_1_1_discontinuity_validation.json`
  was created or modified by this task; `discontinuity.py`, `validate_spatial.py`,
  both reports, `src/**`, the checkpoint, the ONNX, and `.gitignore` are
  untouched. No git commits were made.
- Note: the working tree contains pre-existing uncommitted changes from earlier
  work (e.g. `src/models/stereonet/*`, `.gitignore`); they are unrelated to
  this task and were neither made nor needed by it (**VERIFIED** via
  `git status`: none of this task's three paths touches a tracked file's
  prior state).

## 11 Limitations

A PASS here does NOT mean (**VERIFIED** by design — none of these exists in
code, record, or report):

- object-boundary detection: a large gradient is a depth step in the data,
  nothing more; no detector, confidence map, segmentation, or labelling exists;
- occlusion ground truth: no occlusion labels were compared;
- KITTI boundary accuracy: KITTI provides no boundary ground truth here, so no
  boundary-accuracy number is stated or implied; the real-data check is SANITY
  (shape / finiteness / determinism / non-modification / non-emptiness) only;
- real-camera or Hailo performance: all timings are DEVELOPMENT-MACHINE
  MEASUREMENT; no compilation, quantization, HEF, or device benchmark was
  performed (**NOT APPLICABLE**).
- No confidence maps, detection, segmentation, optical flow, temporal
  filtering, refinement, neural network, or real-camera work was added
  (**NOT APPLICABLE** — explicitly forbidden, none done).

## 12 Final Verdict

**PASS** — 19/19 named checks green, exit code 0, with the frozen algorithm
proven unchanged, C2 depth proven bitwise-unchanged on the sanity scene, frozen
artifacts intact, and discontinuity-only cost measured (≈ 21 ms/frame, ≈ 4 MiB
outputs on the development machine). The C2.1 gap is closed: the discontinuity
operator now has an explicit named real/synthetic correctness gate inside a
validator, with a JSON record and pasted evidence for every substantive claim.
