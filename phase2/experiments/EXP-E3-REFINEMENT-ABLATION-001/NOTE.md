# NOTE — additive; this record's measurements are complete, its status is not

The run completed every measurement and wrote them to `metrics.json` and
`phase2/results/EXP-E3-REFINEMENT-ABLATION-001/ablation.json`, then crashed on the
final terminal print:

    UnicodeEncodeError: 'charmap' codec can't encode character '\u0394'

The table header used the character "Δ", and this Windows console encodes stdout
as cp1252. The exception propagated out of the `Experiment` context, so the record
is marked **status: failed** and carries no conclusion — but nothing was
mis-measured, and all four pre-registered gates passed inside it (harness
fidelity, keep-5 equals drop-5, finiteness, checkpoint integrity).

The script now uses ASCII in that header, and `EXP-E3-REFINEMENT-ABLATION-001-RUN2`
is the clean complete record. Read RUN2; this record is the audit trail, and its
numbers agree with RUN2's.
