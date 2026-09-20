"""ZI-1 / ZI-2: receptive-field support checks at the disparity_initial stage.
MODEL-FREE in the sense that matters: random weights, 3 seeds.  Bit-identity of
outputs under an input change confined outside the dependency window is a
support property of the architecture, not of the weights (same justification as
HS-Z / HS-BAND)."""
import sys, json, numpy as np, torch
sys.path.insert(0,"C:/Users/vishn/stereo_depth_vision")
torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
from phase2.viz import core
from src.models.stereonet import StereoNet, StereoNetConfig
from src.datasets.kitti2015 import normalize

Y0, XL, CH, CW, ST, SEAM = 96, 0, 272, 1136, 16, 568
ZONE=(0,224); COMP=(912,1136)
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

out={}
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
        z = di(right(d,"zone")); g = di(right(d,"global"))
        rows.append({"delta":d,
          "ZI2_zone_interior_zone_vs_global": float(np.abs(z[:,ZONE[0]:ZONE[1]]-g[:,ZONE[0]:ZONE[1]]).max()),
          "ZI1_comp_interior_zone_vs_delta0": float(np.abs(z[:,COMP[0]:COMP[1]]-base[:,COMP[0]:COMP[1]]).max()),
          "zone_interior_response_mean": float((z[:,ZONE[0]:ZONE[1]]-base[:,ZONE[0]:ZONE[1]]).mean()),
          "max_diff_anywhere_zone_vs_delta0": float(np.abs(z-base).max())})
    out[str(seed)]=rows
print(json.dumps(out, indent=1))
