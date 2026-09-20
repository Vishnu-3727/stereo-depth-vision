# SUCCESSOR ASSESSMENT — after GEOM-002

Companion to `POSTMORTEM.md` and `STATISTIC_AUDIT.md`.

**Assessment only. Nothing executed, nothing designed.** No successor
preregistration is written. **No new `t` values, no new statistic, no new
control, no new scenes are selected here.** No training, no inference. GEOM-001
and the GEOM-002 record are untouched.

**Anti-post-hoc compliance.** The GEOM-002 response curves are used nowhere below
except to state that a design choice was inadequate. No conclusion here rests on
their shape, on the 3/12 count, or on the C1–C4 tallies.

---

## THE EIGHT QUESTIONS

### 1. Is there still a scientifically useful question that can be answered?

**Yes.** Level D — geometric correspondence — remains **unanswered**, not
answered negatively.

| experiment | outcome | why it settled nothing about D |
|---|---|---|
| Stage A | halted | falsified analytic anchor; positives never run |
| INDEX-001 ×2 | exploratory-only | uncalibrated single-draw control |
| INDEX-002 | not demonstrated | structurally mismatched null |
| ARCH-001 | CASE A | statistic with ~no power (reference interval 97.4 % of range) |
| GEOM-001 | not demonstrated | representability defect: 99.8 % of mask could not exhibit the hypothesis at `Δ=48` |
| **GEOM-002** | **no admissible result** | hard stop breached; C3 misspecified; C4 vacuous |

**Six experiments, zero admissible determinations at Level D.** The question is
not exhausted — it has never been cleanly asked.

**Crucially:** none of the failures is an *impossibility proof*. Contrast Level C,
which was abandoned because the right image enters the cost volume as a
`k`-constant term, so the candidate axis is **structurally the wrong axis** — an
in-principle barrier. **No such barrier has been demonstrated for Level D.** The
GEOM-001 and GEOM-002 failures are an arithmetic error and a statistical
misspecification: competence failures, both fixable.

### 2. Can it be answered without reusing observed GEOM-002 response data to choose favourable conditions or statistics?

**Partially — and the residual contamination must be disclosed, not papered over.**

**Clean:** the two decisive diagnoses are **pure algebra on the frozen
definitions**, independent of any observation. `Slope` is a linear functional and
therefore non-injective on shape (`STATISTIC_AUDIT.md` C.2–C.4); `t = 0` is a
degenerate regime for `shift="none"` (D.1). Both were provable before execution.
A successor motivated by these needs no observed value.

**Contaminated:** I have now seen every response curve from six experiments. Any
statistic I select is selected by someone who knows the shapes. This is
irreducible for this auditor.

**Honest position:** a successor is defensible only if its statistic is fixed by
**the hypothesis's mathematical form** — a shape-sensitive condition follows from
"ramp versus step" being a shape question, not from having watched a step — and if
the contamination is disclosed in the preregistration as it was in GEOM-002 §0.
The strongest available mitigation is to have the successor specified by a party
that has not read the GEOM-002 curves.

### 3. Can the existing 3 checkpoints and held-out scenes support a defensible successor?

**Yes, with a stated ceiling.**

- **Weights: irreducible.** Training is forbidden; the three `shift="left"`
  checkpoints are the entire population. Reuse is unavoidable and caps evidential
  strength.
- **Scenes: still available.** **MEASURED:** `hailo_val` holds 40 scenes; 4 used
  by GEOM-001, 4 by GEOM-002. **32 remain never used** in any correspondence
  experiment. That is a genuine, free source of unmeasured data.
- **Compute: trivial.** Minutes.

### 4. Would a successor be confirmatory, consistency evidence, or exploratory?

```
CONSISTENCY EVIDENCE at best — and closer to EXPLORATORY than GEOM-002 was.
```

Not confirmatory: weights reused, and the designer has read six response curves.
The status has degraded monotonically across this campaign — confirmatory →
consistency → exploratory — and a seventh experiment sits at the weak end. **Any
successor must state this before execution, not after.**

### 5. Is the synthetic self-pair construction still worth pursuing?

**The construction is sound; the statistic was the failure.** Keep the distinction
sharp.

**Sound (and demonstrated):** `d = a_R − a_L = t` exactly and uniformly; no fill,
padding or interpolation; crop stride-aligned and invariant; **it genuinely
dissolved the representability defect that killed GEOM-001** — every level
remained on the candidate axis and no clipping occurred. `V = 0` exactly at
`k = t/16` gives an unambiguous match, and the only left/right pathway is the cost
volume.

**Two problems now attached to it:**

- **`t = 0` is structurally special.** Left and right become byte-identical, so
  `V ≡ 0` and the aggregation receives no image information at all. For
  `shift="none"` this is a different computational regime (`POSTMORTEM.md` Task B),
  and it corrupted C4. Any reuse must confront it.
- **OOD remains.** A constant-disparity plane is not natural stereo, so a negative
  result stays ambiguous — declared in advance and unchanged.

**Assessment: the construction remains the cleanest available way to pose the
question, but it cannot be reused unexamined.**

### 6. Is a different control or statistic fundamentally required?

**Yes — a different statistic is fundamentally required. The controls are
largely sound.**

- **Statistic: required.** `STATISTIC_AUDIT.md` C.5 — no sample size repairs a
  statistic that discards the dimension carrying the distinction. A
  **shape-sensitive** condition is necessary. *Which one is not selected here.*
- **Vertical arm: keep.** The arm is a properly matched null; it was the statistic
  consuming it that failed.
- **Search-free arm: keep the arm, not the comparison.** Sound as a `k`-constant
  baseline; its `Slope` was corrupted by the degenerate `t = 0` point.
- **Already latent in the design:** C2, strict monotonicity, is shape-sensitive
  and was applied to the **horizontal arm only**. That asymmetry is visible in the
  frozen preregistration without any data.

### 7. Is the random-weight aggregation arm now necessary?

**Yes. It has moved from optional to effectively mandatory for any positive claim.**

`DESIGN_AUDIT.md` §6.4 flagged in advance that a pass could not distinguish
learned from architecturally-forced correspondence, and excluded the arm for
simplicity. **GEOM-002 then produced direct evidence that this architecture emits
substantial structured output from weights alone:** with `V ≡ 0` — no image
information whatsoever — the pipeline returned `disparity_initial` with median
7.603198 and per-pixel range 1.888 … 9.184, **bit-identical across four different
images**.

**DERIVED:** an architecture that generates a structured, ~7-candidate output from
a zero input cannot be assumed to be reporting learned correspondence when it
responds to a real one. Without an untrained-weight arm, a positive result is
uninterpretable.

### 8. Is the correspondence question better answered by a different intervention class entirely?

**Genuinely open — and this is the most important unresolved strategic question.**

Three classes have now failed:

| class | failed because |
|---|---|
| real-pair image translation (Stage A, GEOM-001) | artifact even in Δ; then a representability defect |
| candidate-axis re-indexing (INDEX-001/002/003, ARCH-001) | **in-principle**: the right image is `k`-constant in the cost volume |
| synthetic self-pair (GEOM-002) | statistic misspecification + procedural breach |

Only the candidate-axis class was refuted **in principle**. The other two failed
on execution and statistics.

**UNKNOWN:** whether a fourth class exists that is cleaner. Candidates were
surveyed in `successor_audit_20260911/SUCCESSOR_OPTIONS.md` and none dominated.
**No new class is proposed here** — proposing one would be designing.

**A pattern worth weighing before authorising anything:** every experiment in this
campaign has ended by discovering a defect in its own design rather than by
answering its question. Six for six. That is itself evidence about the difficulty
of instrumenting this architecture, and it argues for a successor that is *simpler
and more conservative* than its predecessors, not more elaborate.

---

## WHY A SUCCESSOR IS WARRANTED — and what must change

1. **The question is unanswered and no impossibility has been shown.** Level D's
   two failures are an arithmetic error and a statistical misspecification, both
   diagnosed and both fixable. Level C, by contrast, was closed by a structural
   argument. That asymmetry is the whole case.
2. **The diagnoses are clean.** C3's non-injectivity and C4's vacuity are algebra
   on the frozen definitions, not inferences from observed curves.
3. **Unmeasured data remain.** 32 held-out scenes.
4. **The construction works.** GEOM-002 proved the self-pair eliminates the
   representability defect. That is real progress, retained even though the run is
   inadmissible.
5. **The cost is minutes**, and the outcome is falsifiable.

**Preconditions — a successor is warranted only if all are met:**

- a **shape-sensitive** primary condition, its form fixed by the hypothesis rather
  than by inspection of any curve;
- the **`t = 0` degeneracy** resolved or excluded from any baseline;
- the **random-weight aggregation arm** included (question 7);
- **hard stops written against verified premises** — every analytic anchor
  independently checked, since a false premise in a hard stop fires on correct
  behaviour;
- **hard stops actually enforced in the harness** — GEOM-002's was not;
- the contamination of question 2 **disclosed in the preregistration**, with the
  status stated in advance as **consistency evidence at best**;
- ideally, the design specified by a party that has not read the GEOM-002 curves.

**Nothing above constitutes a design.** No statistic, no `t` values, no scenes, no
controls and no decision rule are selected.

---

## FINAL

```
SUCCESSOR-JUSTIFIED
```

**Because** the geometric-correspondence question remains open with no
impossibility demonstrated; because GEOM-002's two decisive defects are provable
from its own frozen definitions rather than from its data, and are therefore
correctable without post-hoc selection; because the synthetic construction itself
was validated even as the run was invalidated; and because 32 unmeasured scenes
and a few minutes of inference remain available.

**Not warranted unconditionally.** The preconditions above — above all a
shape-sensitive statistic, the random-weight arm, and enforced hard stops written
against verified premises — are load-bearing. Absent them, a seventh experiment
would repeat the pattern of the previous six.

**No successor is designed, preregistered, or launched. Stopping at the audit.**
