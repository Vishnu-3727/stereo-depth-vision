# R3 — sequence throughput: overlapping decode with inference

## 1 Scope and status

**VERDICT: ACCEPTED. Decode prefetch gains 17.96–21.89 ms per frame at bit-identical output.
10.3 → 13.0 FPS on a sequence.**

This is host runtime work on the development machine, added after the Stage C validation
closed. It changes no frozen artefact, no checkpoint, no evaluation contract and no accuracy
number, and it does not bear on Hailo deployment, which remains blocked. Nothing in the
default runtime path or the demo was rewired by this experiment; see section 9.

## 2 The observation this attacks

R2 established that the inference stage (~53 ms) resists the obvious attacks: channels_last
fails bit-identity, torch.compile is not runnable on this host. The second-largest stage is
preprocessing, ~28–33 ms, which is dominated by the two PNG decodes.

That decode is CPU work. The inference is GPU work. In the per-frame benchmark they are
strictly serialised — the GPU idles while the CPU decodes, then the CPU idles while the GPU
runs. Over a *sequence* they need not be: frame N+1 can be decoded while frame N is on the
GPU. The intervention is therefore **scheduling, not computation**: the per-frame work is
byte-for-byte the same, only its order in time changes.

`cv2.imread` releases the GIL, so a Python thread gives real overlap here rather than
cooperative-only concurrency.

## 3 Host and protocol

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| torch | 2.7.0+cu128 |
| dtype / resolution | fp32, 368x1232 |
| Scenes | first 10 of `hailo_val`, run as a sequence |
| Warmup | 5 frames before timing |
| Per-frame work | concurrent cv2 decode, GPU normalize, frozen net, torch-CUDA geometry with frozen numpy occupancy, cudnn.benchmark OFF |
| Statistic | wall clock of the whole 10-frame sequence; median over repeats |
| Sampling | **four independent processes**: one at 3 repeats, three at 5 repeats |
| Checkpoint | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` (asserted every run) |

Two schedules are compared:

- **serial** — decode frame N, then run frame N on the GPU, then decode frame N+1. This is
  the schedule the per-frame benchmark measures.
- **prefetch** — one background thread decodes ahead into a bounded queue (depth 2) while the
  main thread runs the GPU.

## 4 Hypothesis

H1: the prefetch schedule reduces the wall clock per frame over a sequence, while leaving
the computed disparity bit-identical to the serial schedule.

## 5 Pre-registered gate (decided before any measurement)

ACCEPT only if ALL THREE hold:

1. **Equivalence** — disparity bit-identical to the serial schedule: `0.0` px on 10/10 scenes.
2. **Determinism** — the prefetch schedule, run in two separate processes, gives `0.0` px on
   10/10 scenes.
3. **Gain above noise** — the throughput gain must exceed the measured run-to-run spread.
   A gain inside the noise is NOT a result.

Criteria 1 and 2 are the same standard that rejected cudnn.benchmark (`README.md` section 8)
and channels_last (`R2_INFERENCE_SPEED.md`). Criterion 3 was added because R2 showed the
per-frame timings on this host carry a spread comparable to the effects being chased.

## 6 Gate measurements

Source: `out/r3_gate.json`. Comparison instrument: `r2_equiv.py cmp` (element-wise, float64).

| Check | Bit-identical scenes | Max abs disparity diff | Verdict |
|---|---|---|---|
| prefetch vs serial | 10/10 | **0.0 px** | equivalence PASS |
| prefetch, process A vs process B | 10/10 | **0.0 px** | determinism PASS |
| serial vs the R2 control dump | 10/10 | **0.0 px** | the streaming loop reproduces the timed path exactly |

The third row is a control on the experiment itself: `r3_stream.py` reimplements the
per-frame pipeline as a loop, so it had to be shown to compute what `runtime_path.py`
computes before any timing from it means anything. It does, exactly.

## 7 Measured throughput

Median wall clock per frame over the 10-frame sequence, one row per independent process:

| process | repeats | serial ms/frame | prefetch ms/frame | gain ms/frame |
|---|---|---|---|---|
| 1 | 3 | 97.15 | 75.26 | 21.89 |
| 2 | 5 | 96.99 | 76.55 | 20.44 |
| 3 | 5 | 96.23 | 77.45 | 18.77 |
| 4 | 5 | 94.82 | 76.86 | 17.96 |

| quantity | value |
|---|---|
| serial median range across processes | 94.82 – 97.15 ms/frame (10.29 – 10.55 FPS) |
| prefetch median range across processes | 75.26 – 77.45 ms/frame (12.91 – 13.29 FPS) |
| cross-process median spread | serial **2.33 ms/frame**, prefetch **2.19 ms/frame** |
| gain | **17.96 – 21.89 ms/frame** |

**Criterion 3 passes with room.** Every prefetch median lies below every serial median, and
the smallest gain (17.96 ms) is about 8x the largest cross-process median spread (2.33 ms).

The gain of ~20 ms is smaller than the ~28–33 ms preprocess stage, as expected: the overlap
hides the decode behind the GPU work but does not make either disappear, and the queue
handover and the H2D upload are not overlapped.

### A note on within-run spread

The *within-run* spread across repeats is much larger than the cross-process spread of the
medians — up to 131.3 ms over 10 frames in one serial run. That is tail behaviour, not
centre-of-distribution behaviour; individual repeats occasionally hit a slow frame. The
median over repeats is stable to ~2.3 ms/frame across four processes, which is why the
comparison is made on medians and why the per-process medians are all listed above rather
than pooled into one number.

One early single-repeat prefetch run measured 87.15 ms/frame. With `--repeats 1` the median
is that single sample, including its cold first frame; it is not evidence against the
four-process figures above and is recorded here only so the number is not lost.

## 8 Result

- H1 **supported**. Decode prefetch is **ACCEPTED** under all three pre-registered criteria.
- Sequence throughput on this host: **~10.3 → ~13.0 FPS**, a gain of 17.96–21.89 ms/frame.
- The output is **bit-identical**, not approximately equal: 0.0 px on 10/10 scenes, against
  both the serial schedule and a second process.
- This is a *scheduling* result. No kernel, dtype, tensor layout, model or computed value was
  changed. That is why it can pass a gate that channels_last cannot.

## 9 What this does NOT do

- **It does not change the default runtime path.** `runtime_path.py`, `fast_run_scene` and
  `pipeline_demo.py` are untouched by this experiment. Prefetch is only meaningful for a
  sequence; the demo is single-shot, where it would gain nothing.
- **It does not improve single-frame latency.** The first frame of a sequence still costs the
  full serial time. This is a throughput result, not a latency result.
- **It says nothing about a camera source.** The measured decode is a KITTI PNG read. A live
  camera has a different acquisition cost, and whether the same overlap helps there is
  **NOT MEASURED**.
- **It says nothing about Hailo** — out of scope, unchanged, still blocked.
- 40-scene contract accuracy under the prefetch schedule — **NOT MEASURED, and not needed**:
  bit-identical disparity on the gate scenes means the contract score cannot move. The
  standing contract figure remains 1.191216765057325 px (`out/fast_path_40scene_cuda.json`).

## 10 Files

| File | Role |
|---|---|
| `r3_stream.py` | The two schedules, the dump hook, and the timing harness |
| `out/r3_stream_s10_r3.json` | Process 1 (3 repeats) |
| `out/r3_stream_pool1..3_s10_r5.json` | Processes 2–4 (5 repeats each) |
| `out/r3_gate.json` | The gate measurements and the pooled per-process medians |

## 11 How to reproduce

```bash
python stage_c_deploy/runtime/r3_stream.py --mode both --scenes 10 --repeats 5 --out <json>
python stage_c_deploy/runtime/r3_stream.py --mode both --scenes 10 --repeats 3 --dump-dir <tmp>/r3
python stage_c_deploy/runtime/r2_equiv.py cmp <tmp>/r3/serial <tmp>/r3/prefetch
```

---

Host runtime work only. No frozen artefact, checkpoint, contract or accuracy figure was
created or changed to produce this report. Hailo deployment remains blocked.
