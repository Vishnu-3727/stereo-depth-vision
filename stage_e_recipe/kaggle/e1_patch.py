"""The E1 intervention: weight EMA, decay 0.999.

Four surgical edits to the bundle's copy of `finetune_pilot.py`. Nothing else
changes: same architecture, initialization, data, augmentation, loss,
optimizer, learning rate, batch size, epochs and scheduler as E0.

The EMA is a training-time shadow copy. The exported model has the same 397,954
parameters; no architectural parameter is added. The model has no buffers and
all 70 state entries are floating point (verified), so averaging the whole
state dict is exact and has no buffer-copy edge case.

Per the spec: the monitor evaluates and selects on the EMA weights - the same
weights that will be scored - and both checkpoints store EMA weights.

Imported by build_bundle.py; not executed directly.
"""
from __future__ import annotations

DECAY = 0.999

EDITS = [
    # 1. Construct the shadow copy, right after optimizer and scheduler.
    ("""    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
""",
     """    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # STAGE E / E1 INTERVENTION: weight EMA, decay 0.999. Training-time shadow
    # copy initialised from the Stage-1 weights; adds no model parameter.
    EMA_DECAY = 0.999
    ema_model = copy.deepcopy(model)
    for _p in ema_model.parameters():
        _p.requires_grad_(False)
    ema_model.eval()
""",
     "Create the EMA shadow copy from the initialised weights."),

    # 2. Update it after every optimizer step.
    ("""            optimizer.step()
            epoch_loss += float(loss)
""",
     """            optimizer.step()
            # E1: EMA update, every optimizer step.
            with torch.no_grad():
                _msd = model.state_dict()
                for _k, _v in ema_model.state_dict().items():
                    _v.mul_(EMA_DECAY).add_(_msd[_k], alpha=1.0 - EMA_DECAY)
            epoch_loss += float(loss)
""",
     "EMA update every optimizer step, over the full state dict."),

    # 3. The monitor selects on the EMA weights - the ones that get scored.
    ("""            m = validate(model, val_base, device, limit=10)
""",
     """            m = validate(ema_model, val_base, device, limit=10)
            ema_model.eval()  # validate() leaves its argument in train mode
""",
     "Monitor evaluates the EMA weights, so selection and scoring agree."),

    # 4. Both checkpoints store EMA weights.
    ("""                torch.save({"model": model.state_dict(), "config": exp_config},
                           OUT_DIR / "p2a_best.pth")
""",
     """                torch.save({"model": ema_model.state_dict(), "config": exp_config},
                           OUT_DIR / "p2a_best.pth")
""",
     "Best checkpoint stores EMA weights."),

    ("""    torch.save({"model": model.state_dict(), "config": exp_config}, final_ckpt)
""",
     """    torch.save({"model": ema_model.state_dict(), "config": exp_config}, final_ckpt)
""",
     "Final checkpoint stores EMA weights."),

    # `copy` is needed for the deepcopy above.
    ("""import argparse
import hashlib
""",
     """import argparse
import copy
import hashlib
""",
     "Import copy for the shadow model."),
]


def apply(text: str) -> str:
    """Apply every edit exactly once, failing loudly if a target is missing."""
    for old, new, why in EDITS:
        if old not in text:
            raise SystemExit(
                "E1 PATCH FAIL: target not found -> " + why + "\n" + repr(old[:90]))
        if text.count(old) != 1:
            raise SystemExit(
                f"E1 PATCH FAIL: target appears {text.count(old)} times -> {why}")
        text = text.replace(old, new, 1)
    return text
