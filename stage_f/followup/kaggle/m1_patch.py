"""The M1 intervention: distribution supervision on the aggregation cost.

Surgical edits to the bundle's copy of `scripts/finetune_pilot.py`, and nothing
else. The E0 recipe (optimizer, scheduler, data, augmentation, validation,
checkpointing, init, seed, determinism, epochs) is unchanged; the only training
change is an added cross-entropy term between the full-resolution cost softmax
and a normalised Laplacian target, using the same valid mask as the smooth-L1.

Verification notes (checked against the E0 bundle before freezing this patch):

- Normalisation is EXACTLY `regression.soft_argmin(normalize=True)`:
  `(c - c.mean(1, keepdim)) / (c.std(1, keepdim) + 1e-6)` followed by
  `softmax(-c)`. See `src/models/stereonet/regression.py::soft_argmin`, and the
  E0 config sets `regression_normalize=True`, so the M1 softmax matches the
  readout the model itself uses.
- Valid mask `(disparity > 0) & (disparity < max_disparity_px)` is the same rule
  as `masked_smooth_l1` in `src/losses/disparity.py` (with `max_disparity`
  passed, as the training step does).
- Candidate spacing 8.0 == `config.feature_stride` (E0 config:
  `downsample_levels=3` -> 2**3 = 8) and 24 == `config.num_disparities`;
  both are read from `config` (already in scope in the training loop), not
  hard-coded, and equal the frozen values (24 candidates, max 184 px).

Imported by build_bundle_f1m.py; not executed directly.
"""
from __future__ import annotations

M1_LAMBDA = 0.1
M1_BANDWIDTH = 0.5  # Laplacian b, in candidate units (4 px)

EDITS = [
    # 1. Functional import for the M1 term.
    ("""import torch
from torch.utils.data import DataLoader, Dataset
""",
     """import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
""",
     "M1 needs torch.nn.functional (interpolate / log_softmax); nothing else "
     "uses it, so it is added on its own line."),

    # 2. The training step: read the aggregation cost via return_stages and add
    # the Laplacian CE term. The smooth-L1 line is kept byte-identical.
    ("""            out = model(left, right)
            loss, n = masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))
""",
     """            out, _st = model(left, right, return_stages=True)
            loss, n = masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))
            # M1 distribution supervision: CE between the full-res cost softmax
            # and a normalised Laplacian target, same valid mask as above.
            _m1_cost = _st["aggregated_cost"]
            _m1_up = F.interpolate(_m1_cost, size=disparity.shape[-2:], mode="bilinear", align_corners=True)
            _m1_norm = (_m1_up - _m1_up.mean(1, keepdim=True)) / (_m1_up.std(1, keepdim=True) + 1e-6)
            _m1_logp = F.log_softmax(-_m1_norm, dim=1)
            _m1_valid = (disparity > 0) & (disparity < float(config.max_disparity_px))
            _m1_t = disparity / float(config.feature_stride)
            _m1_k = torch.arange(config.num_disparities, dtype=_m1_up.dtype, device=_m1_up.device).view(1, -1, 1, 1)
            _m1_q = torch.softmax(-torch.abs(_m1_k - _m1_t) / 0.5, dim=1)
            _m1_ce_map = -(_m1_q * _m1_logp).sum(1, keepdim=True)
            _m1_ce = _m1_ce_map[_m1_valid].mean() if int(_m1_valid.sum()) > 0 else _m1_up.sum() * 0.0
            loss = loss + 0.1 * _m1_ce
""",
     "M1 training term: forward with return_stages, upsample the aggregated "
     "cost, standardise exactly as soft_argmin(normalize=True), Laplacian "
     "(b=0.5) target over the config's candidates, lambda=0.1."),

    # 3. Per-epoch accumulator for the CE term.
    ("""        epoch_loss, epoch_pixels, batches = 0.0, 0, 0
""",
     """        epoch_loss, epoch_pixels, batches = 0.0, 0, 0
        epoch_ce = 0.0
""",
     "Accumulate the per-batch M1 CE so the epoch mean can be logged."),

    # 4. Accumulate each batch's CE.
    ("""            epoch_loss += float(loss)
""",
     """            epoch_loss += float(loss)
            epoch_ce += float(_m1_ce.detach())
""",
     "Add this batch's M1 CE to the epoch accumulator."),

    # 5. Log row: epoch mean of the CE term.
    ("""        row = {"epoch": epoch, "mean_loss": epoch_loss / max(batches, 1),
               "valid_pixels": epoch_pixels, "lr": float(scheduler.get_last_lr()[0])}
""",
     """        row = {"epoch": epoch, "mean_loss": epoch_loss / max(batches, 1),
               "m1_ce": epoch_ce / max(batches, 1),
               "valid_pixels": epoch_pixels, "lr": float(scheduler.get_last_lr()[0])}
""",
     "Log row carries m1_ce (epoch mean of the CE term)."),

    # 6. Printed line: show the CE term.
    ("""        print("epoch {:>3} loss {:8.4f} lr {:.2e}{}".format(
            epoch, row["mean_loss"], row["lr"],
""",
     """        print("epoch {:>3} loss {:8.4f} ce {:8.4f} lr {:.2e}{}".format(
            epoch, row["mean_loss"], row["m1_ce"], row["lr"],
""",
     "Printed line shows the epoch-mean M1 CE next to the loss."),
]


def apply(text: str, lam: float = 0.1) -> str:
    """Apply every edit exactly once, failing loudly if a target is missing.

    lam selects the M1 lambda literal inserted in the patched loss line
    (`loss = loss + <lam> * _m1_ce`); allowed values are 0.1 and 1.0.
    lam=0.1 reproduces the original pre-registered patch byte-identically.
    """
    if float(lam) not in (0.1, 1.0):
        raise SystemExit(
            f"M1 PATCH FAIL: lam must be 0.1 or 1.0, got {lam!r}")
    for old, new, why in EDITS:
        if old not in text:
            raise SystemExit(
                "M1 PATCH FAIL: target not found -> " + why + "\n" + repr(old[:90]))
        if text.count(old) != 1:
            raise SystemExit(
                f"M1 PATCH FAIL: target appears {text.count(old)} times -> {why}")
        text = text.replace(old, new, 1)
    if float(lam) == 1.0:
        old_line = "loss = loss + 0.1 * _m1_ce"
        new_line = "loss = loss + 1.0 * _m1_ce"
        if text.count(old_line) != 1:
            raise SystemExit(
                "M1 PATCH FAIL: lambda literal line not found exactly once "
                "for lam=1.0 substitution")
        text = text.replace(old_line, new_line, 1)
    return text
