# Stage E — INT8 control: ARM-P does not survive INT8

**Measured, not inferred.** `int8_control/int8_control.json`, produced by
`int8_control.py` on CPU. `contract_match` true on all six scorings.

## 1 Result

| seed | torch fp32 | ONNX fp32 | ONNX int8 | **P** | fp32 D1 | int8 D1 |
|---|---|---|---|---|---|---|
| 0 | 1.2137379 | 1.2137381 | 6.9218759 | 5.7081378 | 6.3613 % | **83.8534 %** |
| 1 | 1.1962886 | 1.1962887 | 7.0895897 | 5.8933010 | 6.3261 % | **86.2004 %** |
| 2 | 1.2013257 | 1.2013257 | 6.3594673 | 5.1581416 | 6.4304 % | **80.3372 %** |

```
P_control = 5.5865268 px    (spread 0.7351594)
gate for candidates: P_candidate <= 5.6365268 px
```

This is not degradation. A D1 of 80–86 % means four in five pixels are wrong by
more than the 3 px / 5 % threshold: the model is destroyed by quantization, not
merely blunted.

## 2 What it is not

- **Not an export problem.** fp32 ONNX matches fp32 PyTorch to between
  2.0e-8 and 1.95e-7 px. The export is clean; the damage is entirely in
  quantization.
- **Not overflow.** Zero non-finite pixels in all three int8 models. This is
  precision collapse and saturation, not NaN propagation.
- **Not a quirk of this script.** The identical procedure applied to the
  *reference* model in Phase-1 EXP-015 gave fp32 1.313 → int8 1.655, a penalty
  of 0.342 px. **ARM-P's penalty is 16.3x that.** Same quantizer, same
  settings, same calibration split, same evaluator.
- **NOT Hailo evidence.** This is ONNX Runtime CPU static QDQ, QInt8
  activations and weights, per-channel. Hailo's compiler quantizes by a
  different procedure with its own scaling and optimization levels. Stage D
  remains **BLOCKED** and nothing here changes its status in either direction.

## 3 What it probably is

Consistent with the Stage-C dynamic-range audit (`DR-0`), which measured ARM-P
`aggregated_cost` intermediates at ~1.57e18–2.53e18 against the reference
model's ~1e1–1e2, and recorded int8 survival as **UNKNOWN**. It is no longer
unknown for this quantizer: a per-tensor activation scale cannot represent a
tensor spanning eighteen orders of magnitude, so the cost volume quantizes to
noise and the disparity regression that reads it collapses.

DR-0 measured the range. This measures a consequence. The two agree.

**This does not retroactively make DR-1 a pass.** DR-1's Variant A rescaling
failed its own 1e-3 px equivalence gate at 1.4816284e-02 px and stays **FAIL**.

## 4 The problem this creates for the Stage-E gate

The gate is `P_candidate <= P_control + 0.05`. With `P_control = 5.5865268`,
that threshold is **5.6365268 px** — a bar a candidate clears by being *equally
destroyed*. As a numerical-survivability test it is now close to vacuous: it
cannot certify that a candidate survives int8, because the control does not.

The gate still does its literal job, which is to stop a candidate making int8
*worse* than the control. It simply cannot mean what its name implies.

**The rule is not changed here.** It was pre-registered, the measurement came
after, and altering a criterion because a measurement embarrassed it is exactly
what pre-registration exists to prevent. The problem is recorded for decision
instead.

## 5 Consequence for the Stage-E objective

The objective reads: *improve the mean 3-seed KITTI EPE while preserving the
~400k-parameter footprint and passing the pre-registered INT8
numerical-survivability gate*. That wording assumes int8 survivability is a
property ARM-P has and Stage E must avoid losing.

**It is not a property ARM-P has.** Under this quantizer ARM-P is not
int8-deployable at all, and no recipe-only intervention in E1/E2/E3 is likely
to change that: EMA, batch size and schedule length do not reshape an
eighteen-orders-of-magnitude activation range, which is a function of the
cost-volume formulation.

## 6 Open deployment question — not authorized, not investigated

Whether ARM-P can be made int8-viable at all is a real question and a
**separate** one: it would touch quantization procedure (per-tensor versus
per-channel, percentile or entropy calibration, the ONNX Runtime
pre-processing step this run skipped, or excluding the aggregation subgraph
from quantization) or, failing all of those, the model formulation itself.

None of that is authorized, none of it was attempted, and none of it belongs
inside a recipe-optimization campaign. Recorded as an **OPEN DEPLOYMENT
QUESTION**.

One honest caveat on procedure: ONNX Runtime emitted
*"Please consider pre-processing before quantization"* and this run, like
EXP-015 before it, did not run that step. Control and candidates are affected
identically so the comparison stays like-for-like, but part of the absolute
penalty may be attributable to the omission. The 16.3x gap against the
reference model under the *same* omission indicates the bulk of it is not.

## 7 Reproduce

```bash
python stage_e_recipe/int8_control.py
```

CPU only. Every candidate must be measured by this same script on the same
machine, or the comparison against `P_control` is not like-for-like.
