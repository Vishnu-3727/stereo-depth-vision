# AUDIT — EXP-CORRESPONDENCE-INDEX-001 PROVENANCE AND ADMISSIBILITY

Audit date: 2026-09-11 (record written 2026-09-11T01:11:13Z).
Scope: the two execution records under `phase2/diagnostics/correspondence_index/`.

This audit is read-only with respect to every historical record. Nothing in
`20260910T164326Z/` or `20260910T164346Z/` was modified, deleted, re-run or
re-scored. No model was trained, no checkpoint was loaded, no inference was
performed. The only computation performed here is arithmetic over the recorded
JSON outputs, permutation algebra, and a read of the ground-truth disparity maps
to characterise the retained mask.

Throughout, evidence is tagged **MEASURED** (read or recomputed directly from the
records or the filesystem), **DERIVED** (arithmetic on measured values),
**INFERRED** (interpretation), **UNKNOWN** (not determinable from the record).

---

## 0. HEADLINE FINDING — THE DIRECTORY NAMES INVERT THE ACTUAL ORDER

The task framing assumes `20260910T164326Z` ("Run 1") executed first and
`20260910T164346Z` ("Run 2") second, and asks whether Run 2 was constructed to
repair Run 1's random-control failure.

**That premise is false.** Every filesystem timestamp shows `…164346Z` executed
*first* and `…164326Z` executed *second*. The directory names — which are
self-assigned strings, not filesystem facts — are 20 seconds apart in the
opposite order from the files they contain.

To avoid compounding the confusion, this report uses the directory IDs, not
"Run 1"/"Run 2":

| Label used here | Directory | User's label | Verdict recorded in it |
|---|---|---|---|
| **RUN-A** | `20260910T164346Z` | "Run 2" | `…SENSITIVITY-ESTABLISHED` |
| **RUN-B** | `20260910T164326Z` | "Run 1" | `…SENSITIVITY-NOT-DEMONSTRATED` |

**RUN-A is the earlier execution. RUN-B is the later execution.**

---

## A. INSPECTION OF BOTH RECORDS

### A.1 Filesystem chronology (MEASURED)

NTFS creation and modification times, UTC, for every file under
`correspondence_index/`, sorted by creation:

```
16:43:58.829  RUN-A  PREREGISTRATION.md          <-- first prereg written
16:44:09.619  RUN-B  PREREGISTRATION.md          <-- second prereg written (+10.8 s)
16:44:15.209  RUN-A  run_index.py                    RUN-A harness
16:44:36.202  RUN-A  results.json                <-- RUN-A DATA EXISTS
16:44:36.204  RUN-A  run.log
16:45:04.847  RUN-A  summarize.py
16:45:26.285  RUN-A  ENVIRONMENT.txt
16:45:26.287  RUN-A  RESULTS.md                  <-- RUN-A VERDICT (ESTABLISHED) EXISTS
16:45:31.507  ----   exp_correspondence_index.py <-- RUN-B harness first written (+5.2 s)
16:45:38.017  RUN-B  gate_stdout.log             <-- RUN-B EXECUTION BEGINS
16:45:46.981  RUN-B  gate_negative_control.json
16:45:57.745  RUN-B  positive_stdout.log
16:46:07.981  RUN-B  positive_checkpoints.json
16:46:17.558  RUN-B  report_stdout.log
16:46:22.088  RUN-B  results.json                <-- RUN-B VERDICT (NOT-DEMONSTRATED)
16:48:03.828  RUN-B  exp_correspondence_index.snapshot.py
16:48:03.832  RUN-B  ENVIRONMENT.txt
16:48:46.271  RUN-B  RESULTS.md
```

Four facts follow directly (MEASURED):

1. RUN-A's preregistration precedes RUN-B's preregistration by 10.8 s.
2. **RUN-B's preregistration (16:44:09.619) precedes RUN-A's results
   (16:44:36.202) by 26.6 s.** RUN-B's protocol document was therefore frozen
   before RUN-A's data existed.
3. **RUN-B's harness (16:45:31.507) postdates RUN-A's final verdict document
   (16:45:26.287) by 5.2 s.** RUN-B's executable code was authored after RUN-A's
   `ESTABLISHED` verdict was on disk.
4. RUN-A's entire lifecycle (prereg → harness → data → verdict) completes before
   RUN-B's harness exists. RUN-A cannot have been influenced by RUN-B's results,
   which did not exist for another 96 seconds.

**Caveat (MEASURED):** `git ls-files phase2` returns 0 files. Phase 2 is
untracked. There is no commit, no signature and no external timestamp authority
for either record. NTFS mtimes are the only chronology available and they are
mutable. RUN-B's own `ENVIRONMENT.txt` states this honestly (`phase 2 committed
False`; RESULTS.md: "no cryptographic versioning is claimed"). RUN-A's
`ENVIRONMENT.txt` does not mention it.

### A.2 Preregistration content

| Item | RUN-A (`…164346Z`) | RUN-B (`…164326Z`) |
|---|---|---|
| Size | 3 092 bytes | 8 283 bytes |
| Self-declared freeze | "Frozen before execution (2026-09-10T16:43:46Z)" | "Frozen 2026-09-10, before any arm was executed" |
| identity | `[0,1,2,3,4,5,6,7,8,9,10,11]` | `[0,1,2,3,4,5,6,7,8,9,10,11]` |
| +1 | `[11,0,1,2,3,4,5,6,7,8,9,10]` | `[11,0,1,2,3,4,5,6,7,8,9,10]` |
| +2 | `[10,11,0,1,2,3,4,5,6,7,8,9]` | `[10,11,0,1,2,3,4,5,6,7,8,9]` |
| −1 | `[1,2,3,4,5,6,7,8,9,10,11,0]` | `[1,2,3,4,5,6,7,8,9,10,11,0]` |
| **random** | **`[7,2,10,0,5,11,1,8,4,9,6,3]`** | **`[1,10,9,5,3,8,11,0,6,7,2,4]`** |
| random generator recorded | **none — "fixed, single draw"** | `numpy.random.default_rng(20260910).permutation(12)` |
| random reproducible from record | **NO** | **YES** (verified, §C.1) |
| "no fixed points" required | not stated | stated and true |
| Mask | GT valid ∧ GT/16 ∈ [2,8] | GT valid ∧ GT/16 ∈ [2,8] |
| Scenes | `[27,0,31,6]`, `hailo_val` | `[27,0,31,6]`, `hailo_val` |
| Endpoint | `disparity_initial` only | `disparity_initial` only |
| Gate rule | `max|ΔV|==0` and `max|Δd|==0`, else STOP/INVALID | identical |
| Positive criterion | qualitative `d(+2)>d(+1)>d(0)>d(−1)`; +2 generally > +1; **no unit slope, no threshold** | identical, plus explicit "no numerical slope threshold is preregistered and none may be invented afterwards" |
| Random-control criterion | "random = substantial but unstructured (no directional prediction, no threshold)"; "Random ≈ ordered in magnitude+structure → do NOT claim C" | "If ordered and random are indistinguishable in **both** magnitude **and** spatial structure, candidate-coordinate tracking is **not** established"; §10 crit 6 "random does not reproduce the ordered directional pattern" |
| Claim ceiling | level C only; never D/E | level C only; never D/E |

Both preregistrations state the same question, the same tensor location, the same
four ordered arms, the same mask, the same scenes, the same endpoint, the same
gate and the same claim ceiling. **The only substantive difference is the random
permutation** — and, decisively, whether its provenance is recorded.

### A.3 Checkpoints and hashes (MEASURED — identical across both runs)

| key | sha256 | RUN-A | RUN-B |
|---|---|---|---|
| NEG `shift=none+standardized` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | yes | yes |
| POS 6b seed0 | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | yes | yes |
| POS 6b seed1 | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | yes | yes |
| POS 6b seed2 | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | yes | yes |

Both runs loaded the same four checkpoints, byte-identical. No extra seeds were
introduced by either run. No checkpoint was retrained.

### A.4 Environment and code integrity (MEASURED)

RUN-B records a harness hash. It verifies:

```
claimed 031a8829f8cbc46972183b78896db18c417cea38e7d01eb4693fb57cd29b5058
actual  031a8829f8cbc46972183b78896db18c417cea38e7d01eb4693fb57cd29b5058   (snapshot)
actual  031a8829f8cbc46972183b78896db18c417cea38e7d01eb4693fb57cd29b5058   (parent copy)
```

`diff exp_correspondence_index.py 20260910T164326Z/exp_correspondence_index.snapshot.py`
is empty. The snapshot is a true copy of the executed harness.

RUN-B's other provenance claims all verify:

```
HEAD                    58e8a19908ddbd35451652c61aef478f56b51ebd   VERIFIED
phase-1-frozen^{commit} b4207e518166f5089731b4d037a4a79234f70986   VERIFIED
git diff phase-1-frozen -- src scripts                  EMPTY      VERIFIED
working tree            M .gitignore, ?? phase2/                   VERIFIED
```

Phase 1 is untouched. `src/` and `scripts/` are bit-identical to the frozen
Phase-1 tag.

**RUN-A records no harness hash.** Its `ENVIRONMENT.txt` is 359 bytes and lists
only device, torch version, determinism flags, git HEAD and mode. `run_index.py`
is present but unhashed and not separately snapshotted, so there is no record
that the file now on disk is the file that produced `results.json`.

### A.5 Was the protocol frozen before execution?

- **RUN-A:** prereg 16:43:58.829 → harness 16:44:15.209 → data 16:44:36.202.
  Ordering is correct. **YES, by filesystem evidence only** (16 s prereg-to-code
  margin; no external attestation).
- **RUN-B:** prereg 16:44:09.619 → harness 16:45:31.507 → data 16:45:46.981.
  Ordering is correct. **YES for the prose protocol.** But see §D.3: the decisive
  numeric rule that produced RUN-B's verdict exists only in the harness, which was
  written 5.2 s after RUN-A's verdict document.

### A.6 Was the random permutation frozen before results?

- **RUN-A:** the permutation is in a prereg file that predates the results by
  37 s. But the record contains **no generator, no seed, no draw log**. A reverse
  search for the permutation over `numpy.default_rng` seeds 0–400 000,
  `numpy.RandomState` seeds 0–400 000, `random.Random(s).shuffle` seeds 0–400 000
  and `torch.randperm` seeds 0–200 000 found **no match**. The same search
  trivially recovers RUN-B's permutation. So RUN-A's permutation is **not
  reproducible from its own record**: "single draw" is an unverifiable assertion.
  This is **UNKNOWN**, not evidence of manipulation.
- **RUN-B:** `numpy.random.default_rng(20260910).permutation(12)` reproduces
  `[1,10,9,5,3,8,11,0,6,7,2,4]` exactly (verified, §C.1). The permutation is
  independently regenerable and demonstrably not hand-picked. The seed `20260910`
  is the date, a conventional non-adaptive choice.

### A.7 Cross-references between runs

`grep -rn "164326Z"` and `grep -rn "164346Z"` across the whole repository return
hits **only inside each run's own directory**. Neither run's prose, code, JSON or
log mentions the other directory, the other permutation, or the other verdict.
Confirmed by searching each directory for the other's literal permutation list:
no hits.

**Interpretation (INFERRED):** this cuts both ways. It removes any *textual*
evidence of contamination, and it is simultaneously a provenance defect — two
executions of the same experiment ID, with different frozen controls and opposite
verdicts, and neither record discloses that the other exists. A reader of either
directory alone would believe it is the sole execution.

---

## B. CROSS-CONTAMINATION ANALYSIS

### B.1 The CASE 1 / CASE 2 question, as posed

The question was whether `…164346Z` (RUN-A) was created after observing
`…164326Z` (RUN-B) and altered to repair a random-control failure.

**Answer: neither CASE 1 nor CASE 2 as framed — the premise is inverted.**

RUN-A's preregistration, harness, data and verdict all predate the first byte of
RUN-B's harness. RUN-B's random-control failure did not exist, in any form, at
any point in RUN-A's lifecycle. **RUN-A cannot be a post-hoc repair of RUN-B.**
(MEASURED, §A.1.)

### B.2 The question that actually applies: was RUN-B contaminated by RUN-A?

This is the live risk, and the answer is partial.

**Clean (MEASURED):**

- RUN-B's *preregistration* was written 26.6 s before RUN-A's `results.json`
  existed. Its research question, arms, mask, scenes, gate, positive criteria,
  claim ceiling and random permutation were all fixed before any RUN-A number
  existed.
- RUN-B's random permutation is seed-derived and verifiable, so it could not have
  been selected to produce any particular outcome.
- RUN-B's verdict is **negative**. A post-hoc rescue produces a positive. RUN-B
  produced `NOT-DEMONSTRATED` — against its own interest.

**Contaminated (MEASURED):**

- RUN-B's *harness* was written 5.2 s after RUN-A's `RESULTS.md`. The author had
  RUN-A's `ESTABLISHED` verdict and its full delta table available while writing
  the code.
- The harness contains a numeric criterion-6 rule that appears **nowhere** in
  RUN-B's preregistration (§D.3). That rule is the sole cause of RUN-B's verdict.

**Assessment (INFERRED):** the numeric threshold is not tuned to force a result.
Applied to RUN-A's recorded deltas it yields "random does not reproduce ordered"
for all three checkpoints (§D.5), i.e. it would have *passed* RUN-A. It fails
only on RUN-B's own data. A threshold chosen to manufacture an outcome would not
behave that way. The defect is that a decisive quantitative rule was authored
after another run's numbers were visible — a real independence break, even though
it operated against interest.

### B.3 Timestamp and integrity checks

- No file in either directory has an mtime earlier than its own creation time
  except `exp_correspondence_index.snapshot.py` (created 16:48:03.828, mtime
  16:45:31.512). That is the expected signature of a *copy* preserving the source
  mtime, and its hash matches the parent file exactly. Not a defect.
- No generated summary in either directory reproduces the other's numbers.
- No protocol amendment file, errata, addendum or revision exists in either
  directory.
- No evidence of post-hoc threshold editing *within* a run: RUN-B's recorded
  `results.json` criteria booleans reproduce exactly from the snapshot code
  applied to the raw arrays (§D.4).

---

## C. AUDIT OF THE RANDOM PERMUTATIONS

Convention in both runs: `V'(k) = V(π(k))`. A readout that tracks candidate
position moves a signature at candidate `d` to `π⁻¹(d)`.

### C.1 Provenance check (MEASURED)

```
numpy.random.default_rng(20260910).permutation(12)
  = [1, 10, 9, 5, 3, 8, 11, 0, 6, 7, 2, 4]
  == RUN-B frozen random permutation          -> TRUE
```

RUN-B's permutation reproduces exactly. RUN-A's does not reproduce from any seed
in the scanned ranges of four common generators.

### C.2 Permutation algebra (MEASURED / DERIVED)

| | RUN-A (`…164346Z`) | RUN-B (`…164326Z`) |
|---|---|---|
| π | `[7,2,10,0,5,11,1,8,4,9,6,3]` | `[1,10,9,5,3,8,11,0,6,7,2,4]` |
| π⁻¹ | `[3,6,1,11,8,4,10,0,7,9,2,5]` | `[7,0,10,4,11,3,8,9,5,2,1,6]` |
| fixed points | **{9}** | **{}** (none) |
| π⁻¹(d), d = 2…8 | `[1, 11, 8, 4, 10, 0, 7]` | `[10, 4, 11, 3, 8, 9, 5]` |
| displacement π⁻¹(d) − d | `[−1, 8, 4, −1, 4, −7, −1]` | `[8, 1, 7, −2, 2, 2, −3]` |
| mean π⁻¹ over band (uniform) | 5.8571 | 7.1429 |
| mean d over band (uniform) | 5.0000 | 5.0000 |
| **expected displacement, uniform band** | **+0.8571** | **+2.1429** |
| displacement sd (uniform) | 4.5175 | 3.8333 |
| displacement range | [−7, +8] | [−3, +8] |

**RUN-B's reported post-hoc observation is confirmed exactly.** Its RESULTS.md
states `mean π⁻¹ over band = 7.1429`, `mean d over band = 5.0000`, `expected
delta = +2.1429`. All three recompute identically here. The frozen random
permutation is **not** directionally neutral over the retained band under a
uniform-band assumption. That finding stands, is not reinterpreted, and is not
used to rescue anything.

### C.3 The retained band is not uniform — and this changes the picture

Both preregistrations retain `GT/16 ∈ [2,8]`. The actual occupancy of that band
was read from the ground-truth disparity maps of the four focus scenes (MEASURED
— GT read only, no model, no inference):

```
scene 000187_10.png  n= 33 026   mean d = 2.8816   median d = 2.8606
scene 000160_10.png  n= 42 676   mean d = 2.9670   median d = 2.9175
scene 000191_10.png  n= 37 892   mean d = 2.8629   median d = 2.8202
scene 000166_10.png  n= 62 908   mean d = 2.7061   median d = 2.5955
POOLED               n=176 502   mean d = 2.8357   median d = 2.7507
```

These counts reproduce the mask sizes recorded in both runs
(33 026 / 42 676 / 37 892 / 62 908) exactly, in both `run.log` (RUN-A) and
`gate_negative_control.json` / `positive_checkpoints.json` (RUN-B). The mask is
reproducible and model-independent, as preregistered.

Occupancy by integer candidate bin over d = 2…8:

```
d =      2       3       4       5       6     7     8
w = 0.3532  0.4878  0.1527  0.0064  0.000 0.000 0.000
```

**99.4 % of retained pixels lie in d ∈ [2,4]. Bins 6, 7 and 8 are empty.** The
nominal band `[2,8]` is, in the data, effectively `[2,4]`.

Re-weighting the displacement by actual occupancy (DERIVED):

| | RUN-A | RUN-B |
|---|---|---|
| expected displacement, **uniform** band | +0.8571 | +2.1429 |
| expected displacement, **GT-weighted** | **+4.1536** | **+4.3695** |

**Both permutations are severely and almost equally band-biased once the real
occupancy is used.** The apparent difference between them (+0.86 vs +2.14) is an
artefact of assuming a uniform band that the data does not have.

Two consequences follow (DERIVED):

1. **Neither run had a directionally neutral random control.** RUN-A's
   permutation is no better than RUN-B's on the statistic that matters.
2. RUN-B's own post-hoc diagnosis — that its permutation was "badly chosen" and
   that its +2.14 band bias explains the +1.98 observed response — **is not
   supported once occupancy is accounted for.** The GT-weighted prediction is
   +4.37, more than twice the observed +1.98. The agreement RUN-B noted is
   coincidental under the correct weighting.

### C.4 Band bias does not predict either observed response (DERIVED)

| | uniform prediction | GT-weighted prediction | observed random Δ |
|---|---|---|---|
| RUN-B (`…164326Z`) | +2.1429 | +4.3695 | **+1.9793** |
| RUN-A (`…164346Z`) | +0.8571 | +4.1536 | **−0.2033** |

(Observed = mean over the three positive checkpoints; RUN-B uses its own
difference-of-medians statistic, RUN-A its median-of-differences statistic.)

Two permutations with nearly identical GT-weighted band bias (+4.15 vs +4.37)
produced responses of **opposite sign** (−0.20 vs +1.98). **The
position-tracking band-bias model has no demonstrated predictive power over the
random arm.**

**INFERRED:** this is the most consequential finding of the audit. The random
permutation control, as designed in both runs, is a single uncalibrated draw
whose expected response is unknown and, on this evidence, unpredictable.
Criterion 6 in both runs asks whether one such draw "reproduces the ordered
pattern" with no null distribution to compare against. That criterion therefore
has **no established discriminative power in either direction** — it can neither
confirm nor refute candidate-coordinate sensitivity as implemented.

### C.5 A reference-frame caveat on §C.2–C.4

Both preregistrations state that candidate indices are never reinterpreted as
physical disparity, and both record the unresolved right-referenced /
left-referenced discrepancy in the cost volume. The band analysis in §C.2–C.4 —
including RUN-B's own post-hoc note — **does** map `GT/16` onto candidate index
`d`. That mapping is exactly the reinterpretation both preregistrations disavow.
The §C.3–C.4 numbers are therefore DERIVED **under an assumption both protocols
reject**, and are admissible only as a design warning for the next experiment,
never as a result.

### C.6 Mask design intent was not achieved (DERIVED)

RUN-B §7 states the band "keeps candidate content away from the extreme indices
for |m| ≤ 2". With 99.4 % of retained content at d ∈ [2,4], a −1 arm places
content at index 1 and a −2 arm at index 0 — the extreme. The stated protective
purpose of the mask does not hold for the −1 arm, which is one of the scored
ordered arms.

---

## D. VERIFICATION OF RESULTS FROM RAW DATA

All values below were recomputed from `gate_negative_control.json`,
`positive_checkpoints.json` (RUN-B) and `results.json` (RUN-A). RESULTS.md was
not used as a source.

### D.1 Negative-control gate — both runs PASS, exactly (MEASURED)

**RUN-B**, recomputed over all 20 arm×scene cells in `per_scene`:

```
max |volume  − identity|                 = 0.0
max |disparity_initial − identity|       = 0.0
max |identity_index_select − original|   = 0.0
gate PASS per preregistered rule          = True
recorded gate["PASS"] flag                = True
pooled identity median = 6.536478   mean = 6.482193
```

**RUN-A**, recomputed from `volume_diffs` and `rows`:

```
NEG volume diffs, n = 16, max               = 0.0
NEG disparity max |full-frame diff|, n = 16 = 0.0
gate PASS (recomputed)                      = True
recorded gate                               = {"negative_invariant": true}
```

Both gates pass literally and bit-exactly. The mandatory "same `index_select` for
every arm including identity" requirement was honoured in both harnesses
(verified by reading the code, and confirmed by the 0.0 identity-vs-original
diff). For contrast, the positive checkpoints' volume diffs are enormous (RUN-A
min over 48 POS cells = 1.799 × 10⁷; RUN-B up to 1.17 × 10¹¹), confirming the
permutation genuinely alters non-degenerate volumes.

### D.2 Ordered arms — both runs agree to four decimal places (MEASURED)

The two runs share identity, +1, +2 and −1 exactly, so these arms are a direct
cross-validation. RUN-A records `delta_median_vs_identity` (median of per-pixel
differences); RUN-B records `per_pixel_median_delta` computed the same way:

| checkpoint | arm | RUN-A `delta_median_vs_identity` | RUN-B `per_pixel_median_delta` |
|---|---|---:|---:|
| 6b_seed0 | −1 | −1.7804 | −1.780434 |
| 6b_seed0 | +1 | +1.7861 | +1.786055 |
| 6b_seed0 | +2 | +2.9650 | +2.965045 |
| 6b_seed1 | −1 | −1.2030 | −1.202954 |
| 6b_seed1 | +1 | +1.0935 | +1.093480 |
| 6b_seed1 | +2 | +1.9338 | +1.933827 |
| 6b_seed2 | −1 | −1.4651 | −1.465081 |
| 6b_seed2 | +1 | +1.4094 | +1.409429 |
| 6b_seed2 | +2 | +2.5545 | +2.554482 |

Identity absolute medians also agree exactly: NEG 6.5365, seed0 5.5358,
seed1 4.1448, seed2 4.6954 in both runs.

**The two independent executions are numerically identical on every shared arm.**
Determinism held. Neither run's shared-arm data was fabricated, edited or
mis-transcribed.

Ordered criteria, recomputed:

- RUN-A: `d(+2) > d(+1) > d(0) > d(−1)` holds **12/12** (checkpoint, scene).
  `+1 > 0` 3/3, `−1 < 0` 3/3, `+2 > +1` 3/3.
- RUN-B: `d(+2) > d(+1) > d(0) > d(−1)` holds **12/12**. Same directional
  results.

Criteria 1–5 pass in both runs, on identical numbers.

### D.3 PROTOCOL DEVIATION — RUN-B's criterion 6 is not the preregistered rule

RUN-B's preregistration, §9 and §10.6:

> §9 — "**No numerical expected output is preregistered** … Interpreted
> qualitatively … If ordered and random are indistinguishable in **both magnitude
> and spatial structure**, candidate-coordinate tracking is **not** established."
>
> §10.6 — "random does not reproduce the ordered directional pattern."

RUN-B's harness, `exp_correspondence_index.snapshot.py:358–360`:

```python
# random must NOT reproduce the ordered directional pattern; judged by
# whether it lands inside the ordered progression's step scale.
rand_matches_plus1 = abs(dm["random"] - dm["m_plus_1"]) < 0.5 * abs(dm["m_plus_1"]) \
    if dm["m_plus_1"] != 0 else False
```

Three deviations (MEASURED):

1. **A numeric threshold was introduced.** The factor `0.5` appears nowhere in
   the preregistration. §9 states no numerical expected output was preregistered;
   §8 states "no numerical slope threshold is preregistered and none may be
   invented afterwards." The operative rule is a number invented after the prose
   was frozen.
2. **The spatial-structure half was never measured.** The preregistered rule is a
   conjunction (magnitude **and** structure). The recorded arrays contain only
   scalar summaries — `mean, median, std, min, max, delta_mean, delta_median,
   per_pixel_median_delta, max_abs_diff` — and **no** spatial-structure
   comparison of any kind. Half of criterion 6 was not executed.
3. **The code is stricter than the prose.** The preregistration requires *both*
   halves to be indistinguishable before declaring failure. The code declares
   failure on the magnitude half alone. Applied literally, the preregistered
   criterion 6 is **UNDETERMINED**, not failed.

**Direction of the deviation:** conservative. It made RUN-B's verdict more
negative than its literal protocol requires. It is not a rescue, not a goalpost
move toward a positive claim, and RUN-B's RESULTS.md explicitly refuses to
re-score against a different control. But the threshold was authored 5.2 s after
RUN-A's verdict was on disk (§B.2), so it is not independent of RUN-A.

### D.4 RUN-B verdict recomputation — the code's rule was applied correctly

Pooled `delta_median_vs_identity` (difference of medians), recomputed from
`positive_checkpoints.json`:

| checkpoint | −1 | identity | +1 | +2 | random | \|rand−(+1)\| | 0.5·\|+1\| | rand_reproduces |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 6b_seed0 | −1.7788 | 0.0000 | +1.9338 | +2.8208 | +2.3269 | 0.3931 | 0.9669 | **True** |
| 6b_seed1 | −1.2219 | 0.0000 | +1.1251 | +1.9217 | +1.5548 | 0.4297 | 0.5625 | **True** |
| 6b_seed2 | −1.4928 | 0.0000 | +1.4455 | +2.4833 | +2.0561 | 0.6106 | 0.7228 | **True** |

```
all_ordered_monotone              = True
all_directions_correct            = True
random_does_not_reproduce_ordered = False
-> VERDICT = CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
```

This matches the recorded `results.json` verdict and every recorded
per-checkpoint boolean, exactly. **Given its own code, RUN-B's verdict is
correct.** Given its own *preregistration* read literally, the verdict should be
"criterion 6 undetermined".

RUN-B's RESULTS.md reports "observed mean delta +1.9793" for the post-hoc note;
that is the mean of the three difference-of-medians random deltas
`(2.3269 + 1.5548 + 2.0561)/3 = 1.9793`. Confirmed.

### D.5 RUN-A verdict recomputation — no decidable rule exists in code

RUN-A's `run_index.py` computes **no criterion beyond the negative gate**. Its
return value is `0 if neg_ok else 2`; there is no `verdict` variable, no ordered
check, no random check. `summarize.py` prints a table and nothing else. The
`ESTABLISHED` verdict exists **only as prose written by hand into RESULTS.md**.

Recomputed from RUN-A's raw `rows` (mean of per-scene median-of-differences):

| checkpoint | −1 | identity | +1 | +2 | random |
|---|---:|---:|---:|---:|---:|
| NEG_shift_none | +0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| POS_6b_seed0 | −1.7804 | 0.0000 | +1.7861 | +2.9650 | **−0.8096** |
| POS_6b_seed1 | −1.2030 | 0.0000 | +1.0935 | +1.9338 | **−0.3602** |
| POS_6b_seed2 | −1.4651 | 0.0000 | +1.4094 | +2.5545 | **+0.5600** |

Per-scene ordering `d(+2) > d(+1) > d(0) > d(−1)`: **12/12** confirmed. Random
arm full-frame `max|Δ|`: seed0 6.234–6.957, seed1 5.444–6.229, seed2
6.342–6.469 — RUN-A's stated "5.4–7.0" is accurate.

RUN-A's RESULTS.md table mixes two statistics in one row: the absolute column is
the mean of per-scene medians (e.g. seed0 random 4.7799) while the parenthetical
delta is the mean of per-scene median-of-differences (−0.8096). The difference of
the two reported columns is −0.7559, not −0.8096. Both numbers are individually
correct and recomputable; the table is internally inconsistent in which estimator
it displays. Minor reporting defect; does not change any sign or ordering.

Applying RUN-B's threshold to RUN-A's data (for comparison only — **not** a
re-score of either run):

```
seed0  |−0.8096 − 1.7861| = 2.5957  vs  0.8930   -> does NOT reproduce ordered
seed1  |−0.3602 − 1.0935| = 1.4537  vs  0.5467   -> does NOT reproduce ordered
seed2  |+0.5600 − 1.4094| = 0.8494  vs  0.7047   -> does NOT reproduce ordered
```

RUN-A's random arm is clearly separated from its ordered arms — opposite sign on
two of three checkpoints. This is the substantive basis of its `ESTABLISHED`
verdict, and under any reasonable magnitude reading it is met. The
spatial-structure half of the criterion was, as in RUN-B, **never measured**;
RUN-A's phrase "large unstructured full-frame changes" is supported only by a
`max|Δ|` magnitude statistic, not by any structural comparison.

### D.6 Summary of protocol deviations

| # | Run | Deviation | Direction |
|---|---|---|---|
| D-1 | RUN-B | criterion-6 numeric threshold `0.5·\|Δ+1\|` not in preregistration | conservative (toward negative) |
| D-2 | RUN-B | spatial-structure half of criterion 6 not executed | incomplete |
| D-3 | RUN-B | harness authored 5.2 s after RUN-A's verdict existed | independence break |
| D-4 | RUN-A | no criterion evaluated in code; verdict is hand-written prose | unfalsifiable |
| D-5 | RUN-A | random permutation has no recorded generator; not reproducible | provenance gap |
| D-6 | RUN-A | spatial-structure half of criterion 6 not executed | incomplete |
| D-7 | RUN-A | no harness hash, no code snapshot integrity record | provenance gap |
| D-8 | RUN-A | RESULTS.md table mixes two estimators in one row | reporting |
| D-9 | both | neither record discloses the existence of the other | provenance gap |
| D-10 | both | criterion 6 uses a single uncalibrated permutation with no null distribution | design |
| D-11 | RUN-A | random permutation has a fixed point (index 9); RUN-B's design explicitly required none | design |

RUN-A's `RESULTS.md` states "Protocol deviations: none." That statement is
**contradicted by D-4, D-5, D-6, D-7 and D-11.**

---

## E. ADMISSIBILITY DETERMINATION

### RUN-A — `20260910T164346Z` (user's "Run 2", verdict ESTABLISHED)

**STATUS: EXPLORATORY-ONLY**

**Evidence.**

- Preregistration created 16:43:58.829Z; harness 16:44:15.209Z; data
  16:44:36.202Z; verdict 16:45:26.287Z. Ordering correct; entire lifecycle
  completes before RUN-B's harness exists (16:45:31.507Z). Not a post-hoc repair
  of RUN-B. (MEASURED)
- Negative-control gate passes bit-exactly: 16/16 volume diffs 0.0, 16/16
  disparity diffs 0.0. Recomputed from raw. (MEASURED)
- Ordered arms: 12/12 scene-checkpoint cases satisfy `d(+2) > d(+1) > d(0) >
  d(−1)`; numerically identical to RUN-B to 4 dp. (MEASURED)
- Random arm clearly separated from ordered arms: −0.81 / −0.36 / +0.56 against
  +1.79 / +1.09 / +1.41. (MEASURED)
- Random permutation `[7,2,10,0,5,11,1,8,4,9,6,3]` has **no recorded generator or
  seed** and does not reproduce from any seed in the scanned ranges of four
  common RNGs. (MEASURED / UNKNOWN)
- The harness computes no success criterion. The `ESTABLISHED` verdict is
  hand-written prose in RESULTS.md. (MEASURED)
- No harness hash, no integrity record for the executed code. (MEASURED)
- The permutation has a fixed point at index 9; under GT-weighted occupancy its
  expected band displacement is +4.15, i.e. it is **not** a neutral control.
  (DERIVED)

**Reason.** The chronology is clean and the measurement is sound and
reproducible. What fails is falsifiability of the decisive criterion. The rule
that separates `ESTABLISHED` from `NOT-DEMONSTRATED` was never reduced to a
decidable form before execution, was never implemented in code, and was applied
by hand after the numbers were visible. Compounding this, the control that the
verdict rests on cannot be regenerated from the record, so a reader cannot verify
it was drawn rather than chosen. A confirmatory result requires that a reader,
given only the preregistration, could have predicted the verdict from the raw
arrays. That is not possible here.

**May support:** that the degenerate `shift="none"` readout is algebraically
invariant under candidate permutation; that the ordered ±1/±2 response is large,
consistent and 12/12 ordered; that *this particular* scramble produced a response
of opposite sign to the ordered arms on 2 of 3 checkpoints.

**May NOT support:** a confirmatory claim of candidate-coordinate sensitivity
(level C). Nothing at levels D or E. No claim that its random permutation was a
neutral or unbiased control.

---

### RUN-B — `20260910T164326Z` (user's "Run 1", verdict NOT-DEMONSTRATED)

**STATUS: EXPLORATORY-ONLY**

**Evidence.**

- Preregistration created 16:44:09.619Z, **26.6 s before** RUN-A's
  `results.json` existed. The protocol document is independent of RUN-A's
  outcome. (MEASURED)
- Random permutation reproduces exactly from the recorded generator
  `numpy.random.default_rng(20260910).permutation(12)`. Verified. No fixed
  points, as claimed. (MEASURED)
- All provenance claims verify: harness sha256 `031a8829…` matches; `HEAD
  58e8a19…` matches; `phase-1-frozen b4207e5…` matches; `git diff phase-1-frozen
  -- src scripts` empty; snapshot is a byte-identical copy of the executed
  harness. (MEASURED)
- Negative gate passes bit-exactly across all 20 arm×scene cells, recomputed from
  raw. (MEASURED)
- Ordered arms 12/12; criteria 1–5 all pass; numerically identical to RUN-A.
  (MEASURED)
- Its post-hoc band-bias observation (`mean π⁻¹ over band 7.1429` vs `mean d
  5.0000`, expected +2.1429) recomputes exactly. (MEASURED)
- **Harness created 16:45:31.507Z — 5.2 s after RUN-A's `RESULTS.md`.** The
  decisive criterion-6 threshold `0.5·|Δ+1|` exists only in that harness and
  nowhere in the preregistration. (MEASURED — deviations D-1, D-3)
- Spatial-structure half of criterion 6 never measured; no such field exists in
  any output. (MEASURED — deviation D-2)
- Applying its own threshold to RUN-A's data yields "clean" on all three
  checkpoints, so the threshold is not tuned to force an outcome. (DERIVED)

**Reason.** The preregistration is independent of RUN-A's results and the
execution is the better-documented of the two by a wide margin. But the rule that
produced the verdict was authored after RUN-A's verdict was on disk, is a numeric
threshold the preregistration explicitly forecloses, and implements only half of
the preregistered conjunction. Read literally against its own protocol, criterion
6 is **undetermined**, not failed — so the recorded `NOT-DEMONSTRATED` is a
conservative interpretation rather than the preregistered verdict. Its RESULTS.md
claim "no threshold was tuned post hoc" is true in spirit (the threshold does not
favour any outcome) but false in letter (the threshold did not exist at
preregistration time).

**May support:** that the degenerate readout is algebraically invariant under
every candidate permutation tested (bit-exact 0.0); that the ordered response is
large and 12/12 ordered across three seeds; that **its own frozen random control
was badly chosen** and that, as designed, this experiment did not settle level C.
Its explicit refusal to re-score against a different control is correct practice
and should be preserved as written.

**May NOT support:** a confirmatory refutation of candidate-coordinate
sensitivity. `NOT-DEMONSTRATED` means "this design did not settle it", never "the
model is insensitive to candidate coordinates". Nothing at levels D or E. Its
post-hoc band-bias explanation is additionally **not supported** once GT
occupancy is used (§C.3): the GT-weighted prediction is +4.37 against an observed
+1.98.

---

## F. THE TWO RUNS ARE NOT MERGED

No permutation was combined, no verdict averaged, no statistic pooled across
runs. The shared-arm cross-check in §D.2 is a *consistency verification* of two
independently computed arrays, not a merge: no derived quantity in this audit
mixes RUN-A and RUN-B data.

Both records are preserved exactly as found. RUN-A remains the earlier execution
with an `ESTABLISHED` verdict; RUN-B remains the later execution with a
`NOT-DEMONSTRATED` verdict and its self-reported design flaw intact.

Neither is confirmatory. Both are exploratory, for different reasons:

- RUN-A: sound chronology, undecidable criterion, unverifiable control.
- RUN-B: verifiable control, decidable criterion — authored too late and not the
  preregistered one.

---

## G. CLAIM CEILING AS IT NOW STANDS

**Candidate-coordinate sensitivity (level C) is NOT confirmatorily established.**
The maximum supported claim —

> "Under direct candidate-axis re-indexing, the trained aggregation/readout
> pipeline is sensitive to candidate coordinate identity."

— may **not** be asserted as confirmatory on the basis of either record.

What *is* confirmatorily established by both records jointly, on identical
numbers, against fully preregistered and threshold-free criteria:

1. **MEASURED.** The degenerate `shift="none"` checkpoint's `disparity_initial`
   is **bit-exactly invariant** (0.0) under all four non-trivial candidate
   permutations, on all four scenes, in both runs. The negative control is
   algebraically sound and empirically confirmed.
2. **MEASURED.** For all three 6-block checkpoints, on 12/12 scene-checkpoint
   cases in both runs, `d(+2) > d(+1) > d(0) > d(−1)` holds, with responses of
   1.09–2.97 candidates against a control that is exactly zero. Criteria 1–5 of
   RUN-B's preregistration pass, with no invented threshold.
3. **MEASURED.** The two executions agree to four decimal places on every shared
   arm. Determinism held.

What blocks the level-C claim:

4. **DERIVED.** Criterion 6 — "the scramble must not reproduce the ordered
   pattern" — rests on a single uncalibrated permutation in both runs, with no
   null distribution. Under GT-weighted occupancy the two permutations have
   nearly identical band bias (+4.15, +4.37) yet produced responses of opposite
   sign (−0.20, +1.98). The control has **no demonstrated discriminative power**.
5. **MEASURED.** The spatial-structure half of criterion 6 was never executed in
   either run.

Explicitly **not** established and **not** claimed by anything in this audit:
geometric correspondence (D), genuine disparity search (E), correct matching,
final disparity correctness, monotonic physical translation response, stereo
depth correctness. The words *correspondence*, *matching* and *disparity search*
are not used as synonyms for the ordered response reported above. Candidate
indices are not physical disparity; the recorded right-referenced /
left-referenced discrepancy remains open and uncorrected.

---

## H. FINAL VERDICT

```
RUN-A  20260910T164346Z  ("Run 2", ESTABLISHED)      -> EXPLORATORY-ONLY
RUN-B  20260910T164326Z  ("Run 1", NOT-DEMONSTRATED) -> EXPLORATORY-ONLY

OVERALL -> NO-CONFIRMATORY-EVIDENCE
```

Neither execution is admissible as confirmatory evidence for or against
candidate-coordinate sensitivity. Neither is INVALID: both gates pass
bit-exactly, both sets of raw numbers reproduce, and the two agree exactly on
every shared arm.

---

## I. RECOMMENDATION FOR THE NEXT EXPERIMENT — NOT LAUNCHED

Not designed here and not started, per instruction. Four constraints that any new
preregistration must satisfy, all of which follow from the findings above:

1. **Replace the single-draw scramble with an empirical permutation null.** Both
   failures trace to one uncalibrated control. The cost is negligible — 80 arms
   ran in 7.5–7.7 s with volumes cached — so a few hundred random permutations
   per checkpoint are affordable. Preregister the ordered arms' position within
   that null as the criterion, with the threshold written into the protocol *and
   into the code* before any arm runs.
2. **Fix the mask, or stop describing it as `[2,8]`.** 99.4 % of retained pixels
   lie in d ∈ [2,4] and bins 6–8 are empty. The band does not do what §7 of
   RUN-B's protocol says it does, and the −1 arm pushes content to index 1.
3. **Measure spatial structure, or delete it from the criterion.** Both runs
   preregistered a magnitude-and-structure conjunction and executed only
   magnitude. Either record a structural statistic or preregister a
   magnitude-only rule honestly.
4. **Resolve, or explicitly bracket, the reference-frame discrepancy before any
   geometric diagnostic.** A correspondence experiment necessarily converts
   candidate indices to physical disparity — the exact step both current
   protocols forbid. That conversion cannot be left unresolved in a design whose
   purpose is to separate candidate-coordinate sensitivity from geometric
   correspondence.

Additionally, the next record should carry a `RELATED_RUNS.md` naming both
existing executions, so no future reader encounters either directory believing it
is the only one.

---

## CLOSING QUESTION

> **Which experiment, if any, is the admissible basis for the next geometric
> correspondence diagnostic, and why?**

**Neither run is admissible as a confirmatory basis, and no level-C result may be
carried forward as established.**

What *is* admissible, and is the only thing the next diagnostic may build on, is
the part of both records that was preregistered without a threshold, executed
identically twice, and reproduces bit-exactly:

- the **negative-control gate** — the degenerate `shift="none"` readout is
  algebraically and empirically invariant (0.0) under every candidate
  permutation. This is the first control in this project that cannot respond by
  construction rather than by expectation, and it is sound in both runs;
- the **ordered-arm response** — `d(+2) > d(+1) > d(0) > d(−1)` on 12/12
  scene-checkpoint cases, three seeds, magnitudes 1.09–2.97 candidates, agreeing
  to four decimal places across two independent executions.

Those establish that the aggregation/readout path **responds to candidate-axis
re-indexing in an ordered, direction-consistent way**. They do not establish that
it *tracks coordinate identity*, because the control that was supposed to
separate ordered tracking from generic candidate-axis disturbance failed in both
runs — in RUN-B by producing an indistinguishable response, in RUN-A by being
unverifiable and judged by hand.

The reason neither is confirmatory is the same in both cases and is not about
chronology: **the decisive criterion was never both preregistered and decidable.**
RUN-A had chronological independence but no decidable rule and an unreproducible
control. RUN-B had a reproducible control and a decidable rule, but the rule was
authored after RUN-A's verdict existed and is not the rule its own protocol
specifies.

Therefore the next geometric correspondence diagnostic must be preregistered
fresh, must treat candidate-coordinate sensitivity as **open**, and must not
assume it as a premise. It inherits the gate design and the ordered-arm protocol
— both of which work — and must replace the random-permutation control with a
calibrated null before any verdict language is written.
