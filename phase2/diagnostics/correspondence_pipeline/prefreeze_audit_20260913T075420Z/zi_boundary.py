"""Measured location of the seam's influence at the disparity_initial stage.

Finds, per column, whether the output is bit-identical:
  ZI2 : zone arm vs GLOBAL arm        -> largest x with 0.0 for all x' <= x
  ZI1 : zone arm at Delta vs Delta=0  -> smallest x with 0.0 for all x' >= x
Random weights, 3 seeds: this is a receptive-field SUPPORT property.
"""
import sys, json, numpy as np, torch
sys.path.insert(0,"C:/Users/vishn/stereo_depth_vision")
torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
from phase2.viz import core
from src.models.stereonet import StereoNet, StereoNetConfig
from src.datasets.kitti2015 import normalize

Y0, XL, CH, CW, ST, SEAM = 96, 0, 272, 1136, 16, 568
def crop(img,x0): return img[Y0:Y0+CH, x0:x0+CW]
sc = core.load_scene(1, split="hailo_val")
L = torch.from_numpy(normalize(crop(sc.left, XL)))
def right(delta, mode):
    r = crop(sc.right, XL).copy()
    if delta:
        a = crop(sc.right, XL+ST*delta)
        if mode=="global": r = a
        else: r[:, :SEAM] = a[:, :SEAM]
    return torch.from_numpy(normalize(r))

out={"design_bounds":{"zone":[0,250],"comp":[886,1136]}, "per_seed":[]}
for seed in (11,12,13):
    torch.manual_seed(seed)
    m = StereoNet(StereoNetConfig(cost_volume_shift="left")).eval()
    def di(rt):
        with torch.no_grad():
            lf=m.feature_extractor(L); rf=m.feature_extractor(rt)
            return m.regression(m.aggregation(m.cost_volume(lf,rf)),(CH,CW))[0,0].numpy()
    base = di(right(0,"zone"))
    rows=[]
    for d in (1,3,6):
        z, g = di(right(d,"zone")), di(right(d,"global"))
        c2 = np.abs(z-g).max(axis=0)          # zone vs global, per column
        c1 = np.abs(z-base).max(axis=0)       # zone vs Delta=0, per column
        nz2 = np.nonzero(c2)[0]; nz1 = np.nonzero(c1)[0]
        rows.append({
          "delta": d,
          "ZI2_last_clean_x": int(nz2[0]-1) if len(nz2) else CW-1,
          "ZI1_first_clean_x": int(nz1[-1]+1) if len(nz1) else 0,
          "ZI2_at_design_bound_x249": float(c2[249]),
          "ZI1_at_design_bound_x886": float(c1[886]),
        })
    out["per_seed"].append({"seed":seed,"rows":rows})
print(json.dumps(out, indent=1))
