"""Q1a probe: fp16 per-module finiteness on ARM-P scene 0, cuda (DIAGNOSIS ONLY)."""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
import torch
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
import armp_depth as AD


def stats(t: torch.Tensor):
    f = t.detach().float().cpu().numpy()
    import numpy as np
    finite = np.isfinite(f)
    with np.errstate(all="ignore"):
        mx = float(np.max(np.abs(np.where(finite, f, 0.0)))) if finite.any() else float("nan")
    return mx, bool((f == float("inf")).any() or (f == float("-inf")).any()), bool(np.isnan(f).any()), str(t.dtype).replace("torch.", "")


def main() -> None:
    assert torch.cuda.is_available(), "need cuda"
    net, dev = AD.load_frozen_net("cuda")
    net = net.half().eval()
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    sm = ds[0]
    print(f"scene0: {sm.name}")
    tl = torch.from_numpy(normalize(sm.left)).to("cuda").half()
    tr = torch.from_numpy(normalize(sm.right)).to("cuda").half()
    print(f"input tl: {stats(tl)[:3]} max_abs={stats(tl)[0]:.6g} dtype={tl.dtype}")
    print(f"input tr: {stats(tr)[:3]} max_abs={stats(tr)[0]:.6g} dtype={tr.dtype}")

    order = []
    def hook(name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                mx, has_inf, has_nan, dt = stats(out)
                order.append((name, tuple(out.shape), mx, has_inf, has_nan, dt))
            elif isinstance(out, (tuple, list)):
                for k, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        mx, has_inf, has_nan, dt = stats(o)
                        order.append((f"{name}[{k}]", tuple(o.shape), mx, has_inf, has_nan, dt))
        return fn
    for n, m in net.named_modules():
        if n:  # skip top-level (reported via return_stages final)
            m.register_forward_hook(hook(n))

    with torch.no_grad():
        out = net(tl, tr)
        disp, stages = net(tl, tr, return_stages=True) if False else (out, None)
    # return_stages second pass would double-run hooks; instead call once more cleanly:
    # (hooks already recorded module order for the plain forward above)
    with torch.no_grad():
        # fresh run with return_stages to get named stage tensors
        for h in list(net._forward_hooks.values()):
            pass
        disp2, st = net(tl, tr, return_stages=True)

    print("\n--- per-module forward order (first pass, plain forward) ---")
    first_bad = None
    for name, shape, mx, hi, hn, dt in order:
        flag = ""
        if hi or hn:
            flag = "  <-- NON-FINITE"
            if first_bad is None:
                first_bad = name
        print(f"{name:45s} shape={str(shape):28s} max_abs={mx:.6g} inf={hi} nan={hn} dtype={dt}{flag}")

    print("\n--- return_stages named tensors (second pass) ---")
    import numpy as np
    for k in ("left_features", "right_features", "cost_volume", "aggregated_cost",
              "disparity_initial", "refinement_residual", "disparity_final"):
        t = st[k]
        mx, hi, hn, dt = stats(t)
        print(f"{k:22s} shape={str(tuple(t.shape)):28s} max_abs={mx:.6g} inf={hi} nan={hn} dtype={dt}")

    f = disp.detach().float().cpu().numpy()
    import numpy as np
    print(f"\nfinal out: shape={tuple(disp.shape)} max_abs={float(np.max(np.abs(np.where(np.isfinite(f), f, 0.0)))):.6g} "
          f"inf={bool(np.isinf(f).any())} nan={bool(np.isnan(f).any())} "
          f"finite_frac={float(np.isfinite(f).mean()):.6f}")
    print(f"\nFIRST non-finite module in forward order: {first_bad}")


if __name__ == "__main__":
    main()
