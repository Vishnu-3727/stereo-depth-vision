"""EXP-CORRESPONDENCE-INDEX-002 -- freeze the empirical permutation null.

Run ONCE, before PREREGISTRATION.md is finalised and before any experiment
output exists. Emits `permutations.json`, which is then hashed into the
preregistration.

The seed is fixed here and is the date, 20260911, following the convention used
by EXP-CORRESPONDENCE-INDEX-001 (which used 20260910). It was chosen with no
reference to any response, any previous result, any occupancy statistic or any
preliminary output.

Nothing in this file screens, filters, reorders or rejects a draw. Every
permutation the generator produces enters the null.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

N_CANDIDATES = 12
FROZEN_SEED = 20260911
M_TRIPLES = 512                      # number of null draws
N_RANDOM = 3 * M_TRIPLES             # 1536 permutations, partitioned into triples

# Ordered arms. Convention (identical to INDEX-001): V'(k) = V(pi(k)) with
# pi_m(k) = (k - m) mod 12, so content originally at candidate j moves to
# j + m mod 12; positive m moves content toward larger candidate indices.
ORDERED = {
    "identity": [(k - 0) % N_CANDIDATES for k in range(N_CANDIDATES)],
    "m_plus_1": [(k - 1) % N_CANDIDATES for k in range(N_CANDIDATES)],
    "m_plus_2": [(k - 2) % N_CANDIDATES for k in range(N_CANDIDATES)],
    "m_minus_1": [(k + 1) % N_CANDIDATES for k in range(N_CANDIDATES)],
}


def inverse(p: list[int]) -> list[int]:
    q = [0] * len(p)
    for k, v in enumerate(p):
        q[v] = k
    return q


def main() -> int:
    rng = np.random.default_rng(FROZEN_SEED)
    randoms = [[int(x) for x in rng.permutation(N_CANDIDATES)] for _ in range(N_RANDOM)]

    # Descriptive metadata only. Recorded per D/K of the protocol. NOT used to
    # screen, weight, reorder or reject any permutation, and NOT an input to the
    # decision rule.
    meta = []
    for i, p in enumerate(randoms):
        q = inverse(p)
        meta.append({
            "index": i,
            "permutation": p,
            "inverse_permutation": q,
            "n_fixed_points": sum(1 for k in range(N_CANDIDATES) if p[k] == k),
            "fixed_points": [k for k in range(N_CANDIDATES) if p[k] == k],
            # mean over candidate indices of the position a signature at d would
            # move to, minus d. Uniform over all 12 candidates this is 0 by
            # construction; over the retained band it is not.
            "mean_displacement_all12": float(np.mean([q[d] - d for d in range(N_CANDIDATES)])),
            "mean_displacement_band_2_8_uniform":
                float(np.mean([q[d] - d for d in range(2, 9)])),
            "inverse_on_band_2_8": [q[d] for d in range(2, 9)],
        })

    triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(M_TRIPLES)]

    # duplicate check -- reported, never acted on (12! = 479001600, so collisions
    # are expected to be absent; if any appear they STAY in the null).
    seen: dict[tuple[int, ...], int] = {}
    dups = []
    for i, p in enumerate(randoms):
        t = tuple(p)
        if t in seen:
            dups.append({"first_index": seen[t], "duplicate_index": i, "permutation": p})
        else:
            seen[t] = i

    payload = {
        "experiment": "EXP-CORRESPONDENCE-INDEX-002",
        "n_candidates": N_CANDIDATES,
        "frozen_seed": FROZEN_SEED,
        "generator": "numpy.random.default_rng(20260911); "
                     "then rng.permutation(12) called 1536 times in sequence",
        "numpy_version": np.__version__,
        "convention": "V'(k) = V(pi(k)); pi_m(k) = (k - m) mod 12",
        "ordered_arms": ORDERED,
        "m_triples": M_TRIPLES,
        "n_random_permutations": N_RANDOM,
        "random_permutations": randoms,
        "random_permutation_metadata": meta,
        "null_triples": triples,
        "triple_role": ["stands_in_for_m_plus_2", "stands_in_for_m_plus_1",
                        "stands_in_for_m_minus_1"],
        "duplicates_found": dups,
        "n_duplicates": len(dups),
        "screening_applied": False,
        "rejection_applied": False,
    }

    body = json.dumps(payload, indent=2, sort_keys=False)
    (OUT / "permutations.json").write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (OUT / "permutations.sha256").write_text(digest + "  permutations.json\n",
                                             encoding="utf-8")

    print("seed                 :", FROZEN_SEED)
    print("numpy                :", np.__version__)
    print("random permutations  :", N_RANDOM)
    print("null triples (M)     :", M_TRIPLES)
    print("duplicates           :", len(dups))
    print("min two-sided p      : %.6f" % (2.0 / (M_TRIPLES + 1)))
    print("permutations.json sha256:", digest)
    print("first 4 random       :", randoms[:4])
    print("last 4 random        :", randoms[-4:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
