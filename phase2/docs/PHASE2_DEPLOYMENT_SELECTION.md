# PHASE-2 DEPLOYMENT SELECTION RECORD

**This is a deployment decision, not an experimental verdict.** It changes no
result, no threshold and no preregistered label. It is filed separately from
the experiment record precisely so the two cannot be confused later.

Decided by the project owner, 2026-09-17, on the evidence in
`phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md`.

---

## 1. The two things being kept apart

| | value |
|---|---|
| **Scientific verdict of `EXP-P2A-SCALE-COVERAGE-001`** | **INCONCLUSIVE** — unchanged, not revised, not upgraded |
| **Mechanism verdict** | **NOT CONFIRMED** — unchanged (mean slope_hi 0.5701 vs the 0.60 gate) |
| **Formal experimental incumbent** | **ARM-V** — unchanged |
| **Model selected for deployment** | **P2A** |

P2A is **not** marked CONFIRMED or ACCEPTED anywhere. The 0.0093 px miss of the
ACCEPTED gate remains a miss. `phase1/results/LEADERBOARD.md` is not edited and
ARM-V's status in it is not altered.

## 2. Selection rationale

The preregistered gate is an *experimental acceptance criterion*. Using it as a
*deployment rule* would ship a model whose measured mean error is 23 % higher,
on the strength of a 0.0093 px threshold miss. The gate was also defined as
*ARM-V mean − ARM-V spread*, and ARM-V's spread was itself inflated by the
high-disparity instability this experiment removed — so reducing that
instability made the gate harder to clear. That is a property of the rule's
construction; it is recorded, and it does not change the verdict.

Measured basis for the selection, all under the unchanged frozen contract
(40 scenes, 3,802,797 valid pixels, `contract_match true`):

1. Mean EPE **1.4409826** vs ARM-V **1.7727436** — **−0.3317610 px**.
2. Every P2A seed beat every ARM-V seed (1.4150/1.4486/1.4594 vs
   1.8903/1.5493/1.8786).
3. Seed spread **0.3410 → 0.0444 px**.
4. D1 **10.27 % → 8.59 %**.
5. Diagnosed failure region: GT ≥ 64 EPE **11.76 → 5.15**; GT ≥ 96 EPE
   **33.31 → 12.42**; maximum prediction **91–114 → 131–143 px**.
6. Low disparity did not regress: GT < 64 EPE **1.2897 → 1.2617**.
7. Identical 397,954 parameters, identical graph, identical operators,
   identical inference cost, identical evaluation contract.
8. No tuning, seed selection, stacking or protocol change contaminated the
   comparison.

## 3. What is explicitly NOT claimed

Scale augmentation is **not** claimed to have confirmed the diagnosed
mechanism. The mechanism gate was missed at 0.5701 against 0.60. The supportable
statement is narrower:

> Scale augmentation substantially reduced the high-disparity failure and
> produced a large, consistent accuracy improvement, but the preregistered
> mechanism criterion was not reached.

Also not claimed: that scale coverage is the whole mechanism; that the residual
[128,144) px failure (31–35 px EPE) shares that cause; that a different scale
range would do better. None of those were tested.

## 4. Which checkpoint is deployed, and how it was chosen

Selection rule, stated before it was applied: **the seed with the lowest
frozen-contract EPE.**

| seed | frozen EPE | checkpoint |
|---|---|---|
| **0** | **1.4149796** | `phase2/runs/p2a_scale_coverage/p2a_best.pth` |
| 1 | 1.4485736 | `phase2/runs/p2a_scale_coverage_s1/p2a_best.pth` |
| 2 | 1.4593946 | `phase2/runs/p2a_scale_coverage_s2/p2a_best.pth` |

**Deployed: seed 0**, SHA-256 `0868ffd137a9985306bf5563…`.

Reporting rule attached to this choice: **1.4149796 px is a single-seed
figure** and must be reported as the accuracy of *this artifact*, never as the
method's expected accuracy. The honest expectation for a retrain is the 3-seed
mean, **1.4409826 px**. Both numbers appear together wherever the deployed
model's accuracy is quoted.

## 5. Frozen state after this decision

| item | state |
|---|---|
| Phase 0 | CLOSED |
| Phase 1 | CLOSED — incumbent ARM-V |
| Phase 2 optimization | CLOSED — no further accuracy experiments |
| `EXP-P2A-SCALE-COVERAGE-001` | COMPLETE, verdict INCONCLUSIVE, mechanism NOT CONFIRMED |
| Deployment model | P2A seed 0 |
| Phase 3 | none, and none will be created |

No rescue experiment, no threshold modification, no extra seed, no retuning, no
stacking, no multi-scale inference, no SceneFlow.

## 6. Next and final step

Export and deployment validation of P2A seed 0: ONNX export, numerical parity
against the trained PyTorch graph, deployed accuracy measured through the
exported artifact under the frozen contract, runtime and resource measurement,
and an operator-level compatibility check against the reference deployment
graph. Recorded in `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md`.
