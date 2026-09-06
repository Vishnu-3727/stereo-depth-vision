# Phase 1 — What It Was, What Happened, What We Got

An orientation document. If you are picking this project up cold, or you need to
explain it to someone who was not involved, start here.

The other three top-level documents answer different questions:

| Document | Answers |
|---|---|
| **This one** | What is Phase 1 about, why did we do it, what happened, what did we get |
| [`PHASE_1_FINAL_REPORT.md`](PHASE_1_FINAL_REPORT.md) | The charter questions, in order, with evidence |
| [`docs/phase_1_project_report.md`](docs/phase_1_project_report.md) | The technical findings in depth |
| [`docs/phase_1_closure_audit.md`](docs/phase_1_closure_audit.md) | Is it correct, reproducible and safe to freeze |

---

## 1. Why this project exists

Microchip wants a stereo depth vision system for edge silicon that beats the
current practical benchmark on the tradeoff that actually matters in deployment:
accuracy against latency against memory against power.

"Stereo depth vision" means the whole chain, not just a neural network:

```
two cameras -> calibration -> rectification -> feature extraction ->
stereo matching -> disparity -> refinement -> metric depth (Z = fB/d) -> output
```

The network estimates correspondence. Fixed geometry turns that into metres. A
system has to do both, and has to do them inside an edge power budget.

The obvious reference point is Hailo's StereoNet deployment: a real, shipping,
publicly documented stereo model compiled for a real edge accelerator, with
published accuracy and throughput figures. If we want to claim we beat it, we
first have to know exactly what it is.

## 2. Why a forensic phase, rather than just building something

The temptation in a project like this is to read the paper, build something that
looks better on paper, and benchmark it. That fails in a specific and expensive
way: you end up comparing your model against your *assumption* of the baseline
rather than the baseline itself, and you discover the mismatch late, after the
architecture is committed.

So Phase 1 was defined as deliberately non-creative. The rule was: **understand
the reference system well enough to design against it, and change nothing.**

Specifically it was forbidden to improve the baseline, tune it, fix its
weaknesses, or start designing a replacement. Weaknesses found were to be
recorded — problem, evidence, measurement, hypothesis, possible direction — and
left alone.

That constraint turned out to be the single most valuable decision in the phase.
Three of the four most important findings would have been destroyed by an early
"improvement", because in each case the interesting thing *was* the defect.

## 3. What we actually did

Roughly in order.

**Acquired the real artifacts, not the documentation.** Hailo's ONNX, the
compiled HEF, the compiler script, the Model Zoo configuration, the profiler
report, the application source, and the upstream PyTorch implementation. All
hashed into a manifest. The working rule became: *the executable artifact
outranks anything written about it.* That ordering is why the biggest finding was
found at all.

**Reconstructed the architecture from the ONNX**, layer by layer, 168 nodes with
every shape resolved — rather than from the paper, which describes a different
model, or the upstream README, which describes it incorrectly.

**Recovered the evaluation protocol.** Hailo publishes an accuracy number. It
does not publish what that number means. We reverse-engineered the evaluator, the
data split, the crop, and the ground-truth scaling from the Model Zoo source, and
reproduced the published figure to four significant figures.

**Wrote an independent implementation** from our own specification — not a port
of the upstream code — so that agreement with the reference would be evidence
that the specification was right. It matches to a relative 1e-7 at every stage.

**Measured everything.** Accuracy on disparity and on metric depth, binned by
range. Per-stage latency and memory on GPU and CPU. Precision at fp32, fp16 and
int8. Static operation and activation costs. Failure behaviour.

**Built the missing half.** The reference system has no depth conversion at all,
so we implemented the geometry ourselves in order to measure depth accuracy.

**Surveyed the field** — twelve competing stereo architectures, read from primary
sources, to know what has already been tried.

**Recorded all of it as immutable experiments.** Eighteen of them, each stamped
with the git commit, configuration, hardware and software versions. None deleted,
including the ones that turned out to be wrong.

## 4. What we found

Four things, in order of how much they change the plan.

### The reference model does not do stereo matching

Its cost volume — the component whose entire job is to compare the two camera
views at twelve candidate shifts — produces **twelve bit-identical slices**.
Maximum difference between any slice and the first: exactly 0.0.

The cause is three lines of graph. A zero-padding block is appended to the *right*
of the feature map, then sliced straight back off, so the intended shift never
happens. The bug is in the upstream implementation and was carried faithfully
through Hailo's export and compilation.

The model still scores 8.15 % D1 error on KITTI. **It reaches that without
searching for correspondences at all** — it reads depth out of a single
photometric difference at zero shift, plus the left image.

This is not a case of the model ignoring the second camera: corrupting the right
image costs 90 D1 points. It genuinely uses stereo. It just never searches.

### The published accuracy number is not what its label says

Hailo publishes "EPE 8.223". That figure is the KITTI D1 outlier *percentage* —
their own evaluator names the variable `three_pixel_correct_rate`. The model's
actual end-point error is **1.313 px**.

Anyone benchmarking against 8.223 as an EPE would conclude the model is roughly
six times worse than it is, and would design against a phantom.

### Operation counts do not predict speed

Across pipeline stages, the error between share-of-MACs and share-of-runtime runs
from 0.67× to 142×. One stage that performs *zero* multiply-accumulates consumes
3.6 % of GPU time. And the ranking of which stage matters shifts by 6× between a
GPU and a CPU running the identical graph.

Any efficiency claim about edge silicon derived from FLOP counts alone is
unsupported. That includes claims we might have been tempted to make ourselves.

### Depth error is dominated by geometry, and aggregate metrics hide it

Disparity error is essentially flat with distance — about 1.2 to 1.5 px
everywhere. Depth error grows **43×** across the same range, from 0.21 m inside
10 m to 9.16 m at 50–80 m. That is the `Z²/(fB)` relation, derived analytically
and then observed in a real model's output.

The headline depth accuracy looks strong (96.6 % within 25 %) only because 81 %
of ground-truth pixels sit inside 20 m. At 50–80 m, one pixel in four fails.

## 5. What we got wrong, and why that is in the report

Five errors were made and corrected during the phase. They are preserved
alongside their corrections rather than tidied away, because the record of how a
conclusion was reached is part of the evidence.

Two are worth naming because of what they teach.

**A script asserted its own conclusion.** The training experiment recorded
"validation improves" while its own validation error rose from 18.497 px to
19.216 px. The sentence was a string literal — it did not depend on the run at
all. Fixed by deriving every such statement from the measured values, and pinned
by a test that fails if any outcome claim is hard-coded again.

**A stage mapping was wrong, and an external audit caught it, not us.** The
compiler-profiler analysis mapped Hailo's convolutions to the wrong architectural
stages and reported refinement as 95.2 % of compiled operations. The corrected
figure is 90.6 %.

The instructive part is not the error. It is that **the wrong figure disagreed
with our own independent analysis by nearly five points, and nobody questioned
it** — because both numbers supported the same qualitative conclusion. Agreement
in direction masked disagreement in magnitude. Two routes to the same quantity
giving different answers is exactly the signal this project's discipline exists
to surface, and it was missed.

The analysis now fails loudly if those two routes diverge by more than a point.

## 6. What state everything is in

| | |
|---|---|
| Experiments | 18, contiguous, none deleted, every commit resolving |
| Tests | 75 |
| Claim checks | 73 — every headline figure asserted against its experiment record |
| Reference artifacts | 10, all hash-verified, restorable into a fresh clone |
| Reproducibility | Proven from an empty clone: fetch artifacts, run everything, 73/73 |
| Git history | 14 commits, linear, never rewritten |
| Frozen at | tag `phase-1-frozen` |

The repository verifies itself. `verify_claims.py` checks that the numbers in the
documents match the numbers in the experiment records, that no experiment claims
an improvement its data denies, and that the analysis scripts derive rather than
assert their conclusions. It reports how many checks it could not run, and why,
so a partial pass can never be mistaken for a clean one.

## 7. What we did *not* do, and why that is fine

This is a reconstruction and analysis environment. **It is not a production
stereo-depth system and does not claim to be.** Camera synchronisation,
rectification, live calibration ingestion, confidence output and a production
depth interface are all absent, and were never in scope.

Four genuine research gaps remain, each explicitly labelled rather than papered
over:

- **Middlebury scene-class failure analysis — DEFERRED.** Needs a dataset we did
  not download. Textureless, reflective and thin-structure behaviour is untested.
- **Physical Hailo measurement — UNKNOWN.** No device was available. Hailo's
  profiler report is a *compiler estimate*, and is labelled as such everywhere; we
  never infer device latency from it.
- **Competitor measurement — DEFERRED.** Twelve architectures surveyed from
  primary sources; none implemented or benchmarked.
- **Full-scale training — DEFERRED.** Only a short convergence proof was run, and
  it did not demonstrate validation improvement. The recipe is recorded.

None of these blocks Phase 2. All of them would be dishonest to claim.

## 8. What this buys Phase 2

Concretely, four things we did not have at the start:

1. **A baseline we can trust and reproduce**, with a frozen evaluation protocol,
   so any future comparison is apples to apples.
2. **A known-correct implementation** matching the reference to 1e-7, which any
   new architecture can be diffed against stage by stage.
3. **A measured map of where the cost actually is** — refinement is 90.6 % of the
   arithmetic and 73 % of the runtime; aggregation, which most of the published
   efficiency literature optimises, is already 4.2 %. That alone redirects the
   design effort.
4. **A first question worth asking.**

That question is **H1: what is a working cost volume worth?**

The baseline reaches 8.15 % D1 with its matching stage inert. Whether repairing
it yields a large gain or almost none is genuinely unknown, and the answer
determines everything downstream — including how much of the refinement stage's
90 % compute share is spent repairing the defect rather than doing useful work.

The corrected shift already exists in the code behind a configuration flag that
Phase 1 never enabled. It stays disabled until Phase 2 measures it deliberately,
under the frozen protocol, against the frozen baseline.

## 9. Picking this up

```
git checkout phase-1-frozen
python scripts/fetch_reference.py    # restore vendor artifacts (~270 MB)
python -m pytest tests/ -q           # 75 tests
python scripts/verify_claims.py      # 73 checks against the records
```

Then read [`docs/phase_1_project_report.md`](docs/phase_1_project_report.md) for
the findings and [`docs/research_questions.md`](docs/research_questions.md) for
what is open.

---

## The point

The goal of Phase 1 was never to make the repository look complete. It was to
make the scientific record trustworthy: every completed claim correct, every
mistake corrected and traceable, every unknown labelled as unknown, and the whole
thing reproducible by someone who was not here.

That is what "frozen" means. Phase 2 can now build on it without wondering which
parts to believe.
