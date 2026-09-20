# PROVENANCE ANOMALY — `next_direction_audit_20260911/`

Forensic record, 2026-09-11. **Read-only.** Nothing in
`phase2/diagnostics/correspondence/next_direction_audit_20260911/` was
overwritten, deleted, or had its timestamps normalised.

---

## 1. THE FACTS (MEASURED)

NTFS creation / modification times, UTC:

```
directory        created 03:14:42.814      <- created by the session-32 auditor's mkdir
AUDIT.md         created 03:16:13.708   modified 03:16:13.709   27 124 bytes
design.json      created 03:16:34.535   modified 03:16:34.535    4 401 bytes
RELATED_RUNS.md  created 03:16:34.600   modified 03:16:34.602    3 369 bytes
ADDENDUM.md      created 03:23:58.077   modified 03:23:58.078   10 728 bytes
```

`design.json` and `RELATED_RUNS.md` were written **66 ms apart**, 21 s after
`AUDIT.md`. That cadence is a program emitting files in sequence, not a human.

sha256:

```
e9aeff48ce5d5167e32f02ed6a8a8174b7e068a5e0931f3ee406c8fdcc0038e2  AUDIT.md
9e1ce01333c91652b9e1d1e378bf03d841e87bad60360dfe86eb238388c841cd  design.json
a7b9b6fef485989b741cb21f67bf4577d33461f31c528974770d77bbd6c08cad  RELATED_RUNS.md
6bbb3ccf888dc85ea2c7dbf05340ba9c323cec7bd26b40b2a3d8f8e319e6e472  ADDENDUM.md
```

---

## 2. WHAT WAS INSPECTED

| source | result |
|---|---|
| `git ls-files` on the directory | **0 tracked files** — `phase2/` is untracked (`?? phase2/`), so git carries no authorship for any of it |
| `git log` / `git status` | `HEAD 58e8a19`; no commit touches `phase2/` |
| generator scripts in the repo | `grep -rln "next_direction_audit\|design.json"` over `*.py *.sh *.ps1` → **none** |
| other references anywhere | only `AUDIT.md` and `RELATED_RUNS.md` reference their own directory |
| hidden / temp / backup artifacts | **none** (`*.bak *.tmp *.swp *.orig`, hidden attribute) |
| NTFS alternate data streams | only `:$DATA` on all four files — **no writer tag** |
| concurrent sessions on this account | a peer session **`Stereo depth vision phase 2 [f13692]`** exists (Remote Control, currently **offline**). Its activity window cannot be read from here. |
| new experiment records between 03:03Z and 03:16Z | **none** — the newest record is `correspondence_arch/20260911T025312Z` (ARCH-001, RESULTS.md at 03:03Z) |

---

## 3. CONTENT AND STYLE ANALYSIS

`AUDIT.md` cites `CASE-A-ARCHITECTURE-INDUCED`, "near-zero power", and the
`97.4 %` figure — all of which first appear in ARCH-001's `RESULTS.md`, written
at **03:03Z**. The content therefore **postdates 03:03Z** and is derived from the
legitimate record chain.

Systematic style divergence from the session-32 auditor:

| fingerprint | `AUDIT.md` (unknown author) | `ADDENDUM.md` (session-32) |
|---|---:|---:|
| lines | 426 | 224 |
| occurrences of `β` | 16 | 2 |
| occurrences of "Pattern" | 5 | 0 |
| occurrences of "odd" | **0** | 12 |
| `design.json` formatting | minified, single line, **alphabetically sorted keys** | every JSON emitted by session-32 this session used `indent=2` and insertion order |

The absence of any odd/even treatment in `AUDIT.md`, against its centrality in
`ADDENDUM.md`, is by itself sufficient to show the two were not written by the
same process.

---

## 4. ANSWERS TO A1

1. **Who created `AUDIT.md`?** — **AUTHORSHIP-UNKNOWN.**
2. **Who created `design.json`?** — **AUTHORSHIP-UNKNOWN.**
3. **Who created `RELATED_RUNS.md`?** — **AUTHORSHIP-UNKNOWN.**
4. **Derived from a prior legitimate audit?** — **Yes.** Every record it cites
   exists, with the verdicts it quotes, and its reasoning tracks the documented
   chain Stage A → INDEX-001 ×2 → provenance audit → INDEX-002 → INDEX-003 design
   audit → ARCH-001 design + execution.
5. **Does its content predate the latest audit's reasoning?** — Its *files*
   predate `ADDENDUM.md` by 7 min 44 s. Its *content* postdates ARCH-001 (03:03Z).
   Whether its reasoning predates the session-32 auditor's reasoning cannot be
   established: that auditor's reasoning was not on disk until 03:23:58Z.
6. **Does `ADDENDUM.md` merely supplement them?** — **Yes.** It adds three
   things and disputes two; it modifies nothing. See §A.3 and §A.6 there.
7. **Was any experimental result available to the author beforehand?** — **No
   unrecorded result.** No experiment record was created between ARCH-001
   (03:03Z) and the audit files (03:16Z), and none has been created since. The
   unknown author had access to **exactly the public record**, nothing more.
8. **Provenance defect, or unexplained concurrent-agent write?** —
   **UNEXPLAINED CONCURRENT-AGENT WRITE**, not a provenance defect. Nothing is
   fabricated: every cited number traces to an existing record. What is missing is
   an authorship attribution, and that gap is a property of the workspace
   (`phase2/` untracked, no session tagging), not of the document.

**Not asserted:** that the peer session `[f13692]` wrote these files. It is a
plausible and legitimate candidate; it is **not proven**, and no evidence
available from here can prove it.

---

## 5. A2 — MATERIALITY OF THE DIFFERENCES

| axis | pre-existing `design.json` | `ADDENDUM.md` | this preregistration | material? |
|---|---|---|---|---|
| hypothesis | signed horizontal translation response | same | same | **no** |
| intervention | right-image translation, zero fill | zero fill, wider sweep | **crop-based, no fill** | **yes** — fill asymmetry can leak into the odd part |
| Δ range | ±16, ±32 | ±16…±64 | **±16, ±32, ±48** | **yes** — ±32 gives only 2 signed pairs; ±64 rails the soft-argmin |
| border | 32 px | 64 px | **112 px** (crop 48 + inner 64) | **yes** — follows from the crop margin |
| arms | 144 | 272 | **208** | no (cost only) |
| statistic | plain OLS slope `β_H` on 5 raw points | odd-part slope | **odd-part slope `S_h`** | **yes** — Stage A itself warns a plain OLS over a V is an artefact |
| null | vertical comparator + published degenerate baseline | same | same, baseline **re-measured in-run** | **yes** — removes a published number from the decision path |
| decision rule | `≥9/12` + cut at `β_H < −0.0321` | 12/12, threshold-free | **12/12, threshold-free** | **yes** — the cut is half-derived from a measured value |
| claim ceiling | correspondence-consistent search at candidate resolution | direction + specificity only | **direction + specificity only** | **yes** — the pre-existing ceiling over-claims relative to what its rule tests |

**The two designs are scientifically compatible in hypothesis, intervention
class, controls and unit structure, and differ materially in statistic, sweep,
fill mechanism, decision rule and claim ceiling.** Both are preserved. Neither
was silently reconciled.

---

## 6. A3 — CONVERGENCE

The session-32 auditor reports reaching `BYPASS-LEVEL-C` and
`EXP-CORRESPONDENCE-GEOM-001` before reading the pre-existing files. That
sequence is consistent with its own tool trace but **cannot be verified from the
filesystem**, and the independence of the unknown author is unestablished.

Per A3, therefore:

```
DESIGN-CONVERGENCE-OBSERVED  —  NOT CLAIMED
```

The agreement is **recorded as an observation** and is **not** treated as formal
replication, because authorship and independence are not demonstrated. No weight
is placed on it anywhere in this preregistration.
