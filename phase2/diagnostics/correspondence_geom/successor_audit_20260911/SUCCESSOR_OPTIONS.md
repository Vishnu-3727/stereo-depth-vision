# SUCCESSOR OPTIONS — post-GEOM-001

Companion to `AUDIT.md`. **Nothing executed.** No GEOM-001 response value is used
as a design input or as evidence anywhere below.

Four options were considered. Two are scientifically defensible and are compared
in full (objective 10). Two are rejected with reasons.

---

## OPTION A — Symmetric translation, band mask, fresh scenes

**Question.** Does `disparity_initial` shift by `−Δ/16` candidates under
horizontal right-image translation, on scenes never previously used?

**Independent variable.** `Δ ∈ {−32, −16, 0, +16, +32}` px, horizontal; matched
vertical arm at the same offsets. Crop-based, no fill, crop 1168 × 304
(**DERIVED**, stride-aligned).

**Mask.** `GT/16 ∈ [2, 9]` ∧ border — the band **derived in §1 of `AUDIT.md`**
from the representability constraint, not from any response.

**Scenes.** Selected from the 36 held-out `hailo_val` scenes by a GT-only
criterion fixed in advance (e.g. `[2,9]` retention ≥ 50 %; **MEASURED**: 17 of 40
scenes qualify).

**Controls.** Vertical arm (matched magnitude, same code path, no predicted
slope). `shift="none"` search-free model, measured not assumed.

**Statistic.** `Odd(Δ) = [m(+Δ) − m(−Δ)]/2` at `Δ ∈ {16, 32}`;
`S_axis = Σ Odd(Δ)Δ / ΣΔ²`; anchor `S_h = −1/16`.

**Expected under true correspondence.** `Odd_h(16) ≈ −1`, `Odd_h(32) ≈ −2`,
`S_h ≈ −0.0625`, `S_v ≈ 0`.
**Under artifact / generic binocular dependence.** `Odd ≈ 0` both axes, even parts
grow with `|Δ|`, `S_h ≈ S_v`.

### What Option A could prove
That the signed translation response is present, axis-specific, and exceeds the
search-free baseline, **on scenes whose response has never been measured**.

### What Option A could NOT prove
- Anything at adequate power on the monotonicity condition: **only 2 signed
  offsets exist (§4 of `AUDIT.md`), so `C2` reduces to one inequality —
  probability 1/2 under any null.** The GEOM-001 design audit required ≥ 3 and
  would have rejected this.
- Independence from the published GEOM-001 curve: the same intervention class on
  the same weights, with the mechanism's behaviour already visible.
- Anything about the 40 % of pixels the band discards, which are the majority of
  the natural disparity distribution.

**Assessment: defensible but underpowered. It is weaker than the experiment it
replaces.**

---

## OPTION B — Synthetic fronto-parallel self-pair sweep ★

**Question.** When the right image is a pure horizontal translation of the left
by `t` px — a stereo pair whose true disparity is exactly `t` everywhere — does
`disparity_initial` track `t/16` candidates?

**Why this dissolves the GEOM-001 defect.** The disparity is **set**, not
perturbed. There is no dependence on the scene's GT distribution, so the
representability constraint of §1 cannot bind: every `t` is chosen inside the
axis by construction.

**Independent variable.** `t ∈ {16, 32, 48, 64, 80, 96}` px = 1…6 candidates.
**DERIVED**: crop margin `M = 96`, crop **1040 × 176 = (65 × 16) × (11 × 16)**,
stride-aligned, identical for every condition, no fill, no interpolation, all `t`
multiples of 16. Predicted response `1…6` on a `0…11` axis — clear of both rails
(failure mode 6 and 9 of `AUDIT.md` §8).

**Matched control.** Vertical self-pair `(L, shift_v(L, t))` at the same `t`:
identical image, identical displacement magnitude, identical crop and code path;
no horizontal correspondence exists.

**Artifact control.** `shift="none"` search-free model on the identical sweep —
measured, never assumed.

**Mask.** Purely geometric: crop interior minus a fixed border. **No GT is
required at all** — the ground truth is `t` by construction and uniform. This
removes the GT-band machinery, the scene-selection question, and the
mask-occupancy confound that has recurred since INDEX-001.

**Statistic (one-sided; odd/even is unavailable and unnecessary).**

```
m_axis(t) = median( disparity_initial[mask] )
Slope_axis = Σ (m(t) − m(0)) · (t/16)  /  Σ (t/16)²        OLS through the origin
ANCHOR: Slope_h = 1 candidate per candidate     (exact, by construction)
```

**Why one-sided is acceptable here.** Negative `t` would mean negative disparity,
which is unrepresentable — the same constraint, faced honestly. The
signed-versus-magnitude separation that odd/even provided is replaced by two
stronger discriminators: an **exact slope of 1** and the **matched vertical arm**.

| mechanism | `Slope_h` | `Slope_v` | monotone in `t` |
|---|---|---|---|
| true correspondence | **≈ 1** | ≈ 0 | yes, horizontal only |
| generic binocular dependence | arbitrary, ≈ `Slope_v` | arbitrary | possibly both — magnitude grows with `t` |
| architectural / padding artifact | small | small | irregular |
| search-free control | measured | measured | measured |

**Power.** 6 levels. Strict monotonicity by chance is `1/6! = 1/720` per unit,
against `1/2` for Option A. This is the decisive difference.

### What Option B could prove
That the pipeline locates a horizontal correspondence and reports it at
approximately the correct candidate, over a 6× range, specifically on the
epipolar axis, beyond what a provably search-free model produces.

### What Option B could NOT prove
- That it works on **real** stereo. A constant-disparity plane is the easiest
  possible input: identical texture, no occlusion, no slant, no ambiguity.
- Correct disparity on natural scenes, sub-candidate precision, or generalisation.
- **A negative result is ambiguous**: failure could mean no correspondence
  machinery *or* out-of-distribution input (constant disparity is never seen in
  training).

**Asymmetry, stated in advance: Option B is a strong falsifier of a *positive*
claim and a weak confirmer of real-stereo competence.** Given that this
investigation has repeatedly been misled by weak confirmers, that asymmetry is a
feature.

---

## OPTION C — rebuild the symmetric sweep on the same four scenes — **REJECTED**

`m_h(Δ)` and `m_v(Δ)` at `Δ ∈ {±16, ±32}` are **already published** in GEOM-001
`RESULTS.md` §4–§5 for all 16 cells. A successor here recomputes a statistic from
public inputs. This is the failure mode that made INDEX-001, INDEX-002 and
ARCH-001 non-confirmatory. **Rejected.**

---

## OPTION D — widen the candidate axis or retrain — **REJECTED**

The representability ceiling comes from a 12-candidate axis meeting data that
occupy ~0…5 candidates. Widening the axis or retraining on larger disparities
would remove the constraint — but training is forbidden, it would change the
object under study, and every existing record would become incomparable.
**Rejected. Noted only to record that the constraint is architectural, not
merely methodological.**

---

## SIDE-BY-SIDE

| | **A** symmetric, fresh scenes | **B** synthetic self-pair ★ |
|---|---|---|
| representability constraint | binds hard; band mask discards ~55 % | **does not bind** — disparity is set |
| usable levels | 2 signed offsets | **6** |
| monotonicity under the null | 1/2 | **1/720** |
| GT dependence | band mask + scene selection | **none** |
| scene-distribution confound | present | **absent** |
| anchor | `−1/16` cand/px | **slope 1 cand/cand, exact by construction** |
| inputs already public? | mechanism visible from GEOM-001 | **never measured on any checkpoint** |
| input realism | real stereo pairs | synthetic, OOD |
| interpretation of a negative | reasonably clean | **ambiguous (OOD)** |
| forward passes | 4 models × N scenes × 9 | 4 models × N scenes × 13 |
| cost | ~1 min | ~1–2 min |

**Simplest design that can actually discriminate (objective 11): Option B.** It
removes the failure mode that broke GEOM-001 rather than working around it,
needs no GT, no band, and no scene selection, and buys a 360× stronger
monotonicity constraint.

**Both A and B should use held-out scenes** (**MEASURED**: 36 of 40 unused), and
neither can be fully confirmatory because the 3 checkpoints are irreducibly
reused and the designer has seen the GEOM-001 curve (`AUDIT.md` §0, §7).
