"""EXP-CORRESPONDENCE-ARCH-001 -- freeze the 16 random aggregation initialisations.

Run ONCE, before PREREGISTRATION.md is finalised and before any inference on
real data. Emits `random_initialization_metadata.json` and `generator.json`,
whose hashes are then written into the preregistration.

Instantiates ONLY the aggregation module (the repository class, identical
forward code). Loads no checkpoint, loads no scene, runs no inference, performs
no training. Seeds 0..15 are frozen here and were fixed in the authorised design
audit before any output existed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
REPO_ROOT = OUT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.stereonet.aggregation import Aggregation  # noqa: E402

SEEDS = list(range(16))                 # frozen: 0 .. 15
IN_CHANNELS, CHANNELS, NUM_LAYERS = 32, 32, 4
EXPECTED_PARAM_COUNT = 111585
N_CANDIDATES = 12


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_random_aggregation(seed: int) -> Aggregation:
    """Framework-default initialisation, exactly as the architecture receives it
    when instantiated from scratch. No custom init, no scaling, no training."""
    torch.manual_seed(seed)
    return Aggregation(in_channels=IN_CHANNELS, channels=CHANNELS, num_layers=NUM_LAYERS)


def main() -> int:
    # ---- generator: c(k) = (k-1) mod 12, powers c^m -> pi_m(k) = (k-m) mod 12
    perms = {str(m): [(k - m) % N_CANDIDATES for k in range(N_CANDIDATES)]
             for m in range(N_CANDIDATES)}
    gen = {
        "experiment": "EXP-CORRESPONDENCE-ARCH-001",
        "n_candidates": N_CANDIDATES,
        "generator": "c(k) = (k - 1) mod 12",
        "convention": "V'(k) = V(pi_m(k)); pi_m(k) = (k - m) mod 12; content at j moves to j+m",
        "orbit_positions": list(range(N_CANDIDATES)),
        "permutations_by_m": perms,
        "identity_m": 0,
        "minus1_equivalent_m": 11,
        "note": "m=0 is identity; m=11 is the permutation INDEX-001/002 called -1",
    }
    gen_body = json.dumps(gen, indent=2)
    (OUT / "generator.json").write_text(gen_body, encoding="utf-8")
    gen_digest = hashlib.sha256(gen_body.encode("utf-8")).hexdigest()

    # ---- 16 random aggregations
    entries = []
    for s in SEEDS:
        agg = build_random_aggregation(s)
        n = sum(p.numel() for p in agg.parameters())
        if n != EXPECTED_PARAM_COUNT:
            raise RuntimeError("HARD STOP: aggregation parameter count %d != %d"
                               % (n, EXPECTED_PARAM_COUNT))
        tensors = []
        for name, p in agg.named_parameters():
            tensors.append({"name": name, "shape": list(p.shape),
                            "dtype": str(p.dtype), "sha256": tensor_sha256(p)})
        combined = hashlib.sha256(
            "".join(t["sha256"] for t in tensors).encode("utf-8")).hexdigest()
        entries.append({"seed": s, "param_count": n,
                        "tensors": tensors, "combined_sha256": combined})

    # ---- K5: every randomised tensor must differ across all seed pairs
    collisions = []
    for ti in range(len(entries[0]["tensors"])):
        seen: dict[str, int] = {}
        for e in entries:
            h = e["tensors"][ti]["sha256"]
            if h in seen:
                collisions.append({"tensor": e["tensors"][ti]["name"],
                                   "seed_a": seen[h], "seed_b": e["seed"]})
            else:
                seen[h] = e["seed"]
    if collisions:
        raise RuntimeError("HARD STOP: identical parameter tensors across seeds: %r"
                           % collisions)

    payload = {
        "experiment": "EXP-CORRESPONDENCE-ARCH-001",
        "module_randomised": "aggregation",
        "architecture": "4 x [Conv3d(32->32, 3x3x3, padding=1) + LeakyReLU(0.01)] "
                        "then Conv3d(32->1, 3x3x3, padding=1)",
        "constructor": "Aggregation(in_channels=32, channels=32, num_layers=4)",
        "initialisation": "framework default - torch.nn.modules.conv._ConvNd.reset_parameters: "
                          "kaiming_uniform_(weight, a=sqrt(5)); "
                          "bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))",
        "seed_procedure": "torch.manual_seed(seed) immediately before construction",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "param_count_each": EXPECTED_PARAM_COUNT,
        "training_performed": False,
        "modules_not_randomised": ["feature_extractor", "cost_volume", "regression/readout",
                                   "refinement (never invoked)"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "seed_tensor_collisions": collisions,
        "initialisations": entries,
    }
    body = json.dumps(payload, indent=2)
    (OUT / "random_initialization_metadata.json").write_text(body, encoding="utf-8")
    init_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    (OUT / "frozen.sha256").write_text(
        "%s  random_initialization_metadata.json\n%s  generator.json\n"
        % (init_digest, gen_digest), encoding="utf-8")

    print("seeds                    :", SEEDS)
    print("param count per seed     :", EXPECTED_PARAM_COUNT)
    print("tensor-hash collisions   :", len(collisions))
    print("torch / cuda / numpy     : %s / %s / %s"
          % (torch.__version__, torch.version.cuda, np.__version__))
    print("generator.json sha256    :", gen_digest)
    print("random_init_meta sha256  :", init_digest)
    for e in entries[:3]:
        print("  seed %2d combined sha256 %s" % (e["seed"], e["combined_sha256"]))
    print("  ...")
    print("  seed %2d combined sha256 %s" % (entries[-1]["seed"], entries[-1]["combined_sha256"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
