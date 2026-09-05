"""Stereo geometry: calibration, disparity to depth, and error propagation.

This is the deterministic half of a stereo depth system. The network predicts
disparity; everything in this module is fixed geometry that turns disparity
into metres and tells us how much a disparity error is worth in depth.

The core relation for a rectified pair is

    Z = f * B / d

with Z the depth along the optical axis in metres, f the focal length in
pixels, B the baseline in metres, and d the disparity in pixels. Differentiating
with respect to disparity,

    dZ/dd = -f * B / d**2   and, substituting d = f*B/Z,
    dZ/dd = -Z**2 / (f * B)

so a fixed disparity error of one pixel costs depth error proportional to the
square of the depth. That single fact drives most of the far-range behaviour of
any stereo system, learned or classical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    """A rectified stereo pair, reduced to what disparity-to-depth needs."""

    focal_px: float
    baseline_m: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = ""

    @property
    def fB(self) -> float:
        """The product that converts disparity to depth, in pixel-metres."""
        return self.focal_px * self.baseline_m

    def depth_from_disparity(
        self, disparity: np.ndarray, min_disparity: float = 1e-3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a disparity map to metric depth.

        Returns (depth, valid). Pixels with disparity at or below
        ``min_disparity`` are not converted -- their depth is meaningless, not
        merely large -- and are reported as invalid rather than being clipped to
        some finite number. Silently turning an invalid disparity into a valid
        depth is the single easiest way to corrupt a depth benchmark.
        """
        disparity = np.asarray(disparity, dtype=np.float64)
        valid = disparity > min_disparity
        depth = np.full(disparity.shape, np.nan, dtype=np.float64)
        np.divide(self.fB, disparity, out=depth, where=valid)
        return depth, valid

    def disparity_from_depth(
        self, depth: np.ndarray, min_depth: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`depth_from_disparity`, with the same validity rule."""
        depth = np.asarray(depth, dtype=np.float64)
        valid = depth > min_depth
        disparity = np.full(depth.shape, np.nan, dtype=np.float64)
        np.divide(self.fB, depth, out=disparity, where=valid)
        return disparity, valid

    def depth_error_from_disparity_error(
        self, depth_m: np.ndarray | float, disparity_error_px: float = 1.0
    ) -> np.ndarray:
        """Depth error implied by a disparity error, via |dZ/dd| = Z^2 / (fB).

        This is the first-order (linearised) figure. It understates the true
        error for large disparity errors at long range, because Z(d) is convex;
        use :meth:`depth_error_exact` when that matters.
        """
        depth_m = np.asarray(depth_m, dtype=np.float64)
        return depth_m**2 / self.fB * disparity_error_px

    def depth_error_exact(
        self, depth_m: np.ndarray | float, disparity_error_px: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact depth shift for a disparity over- and under-estimate.

        Returns (depth_if_disparity_underestimated, depth_if_overestimated).
        Underestimating disparity pushes the point further away, and the two
        directions are not symmetric -- the far side blows up while the near
        side is bounded. That asymmetry is invisible in the linearised figure.

        When the error is large enough to drive disparity to zero or below, the
        far side is reported as ``inf``: the depth is not enormous, it is
        undefined, and the point has been pushed behind the camera or to
        infinity. Clamping disparity to a small positive number instead would
        produce a finite but meaningless figure.
        """
        depth_m = np.asarray(depth_m, dtype=np.float64)
        d = self.fB / depth_m
        d_low = d - disparity_error_px
        far = np.where(d_low > 0.0, self.fB / np.where(d_low > 0.0, d_low, 1.0), np.inf)
        return far, self.fB / (d + disparity_error_px)

    def max_range_for_disparity(self, disparity_px: float) -> float:
        """Depth at which disparity falls to the given value.

        With ``disparity_px = 1`` this is the range beyond which the whole
        remaining scene is compressed into less than one pixel of disparity.
        """
        return self.fB / disparity_px


def parse_kitti_cam_to_cam(
    path: str | Path, left_cam: int = 2, right_cam: int = 3
) -> StereoCalibration:
    """Read a KITTI ``calib_cam_to_cam`` file into a StereoCalibration.

    KITTI's rectified projection matrices have the form

        P_rect_0x = [[f, 0, cx, tx], [0, f, cy, ty], [0, 0, 1, 0]]

    where ``tx = -f * b_x`` and ``b_x`` is the camera's x position in the
    rectified frame. The baseline between two cameras is therefore
    ``|tx_right - tx_left| / f``. The default pair 2/3 is the colour pair the
    KITTI stereo benchmark is defined on.
    """
    values: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()])
        except ValueError:
            continue  # non-numeric fields such as calib_time

    p_left = values["P_rect_{:02d}".format(left_cam)].reshape(3, 4)
    p_right = values["P_rect_{:02d}".format(right_cam)].reshape(3, 4)
    size = values["S_rect_{:02d}".format(left_cam)]

    focal = float(p_left[0, 0])
    if not np.isclose(focal, p_right[0, 0]):
        raise ValueError(
            "Rectified focal lengths differ between the two cameras "
            "({} vs {}); the pair is not properly rectified.".format(
                focal, p_right[0, 0]
            )
        )
    baseline = float(abs(p_right[0, 3] - p_left[0, 3]) / focal)

    return StereoCalibration(
        focal_px=focal,
        baseline_m=baseline,
        cx=float(p_left[0, 2]),
        cy=float(p_left[1, 2]),
        width=int(size[0]),
        height=int(size[1]),
        source=str(path),
    )


def demo() -> None:
    """Self-check: geometry round-trips, invalid pixels stay invalid, and the
    linearised error matches a numerical derivative."""
    calib = StereoCalibration(
        focal_px=721.5377, baseline_m=0.5372, cx=609.56, cy=172.85,
        width=1242, height=375, source="synthetic",
    )

    # round trip
    depths = np.array([5.0, 10.0, 20.0, 50.0, 80.0])
    disp, ok = calib.disparity_from_depth(depths)
    assert ok.all()
    back, ok2 = calib.depth_from_disparity(disp)
    assert ok2.all()
    assert np.allclose(back, depths), back

    # an invalid disparity must not become a valid depth
    d = np.array([10.0, 0.0, -1.0, 1e-9])
    z, valid = calib.depth_from_disparity(d)
    assert valid.tolist() == [True, False, False, False]
    assert np.isnan(z[1:]).all(), z

    # linearised error agrees with a numerical derivative of Z(d)
    for depth in depths:
        d0 = calib.fB / depth
        h = 1e-4
        numeric = abs((calib.fB / (d0 + h) - calib.fB / (d0 - h)) / (2 * h))
        analytic = calib.depth_error_from_disparity_error(depth, 1.0)
        assert abs(numeric - analytic) / analytic < 1e-6, (depth, numeric, analytic)

    # the exact figures bracket the linearised one, and are asymmetric
    far, near = calib.depth_error_exact(80.0, 1.0)
    lin = calib.depth_error_from_disparity_error(80.0, 1.0)
    assert far - 80.0 > lin > 80.0 - near, (far, near, lin)

    # an error large enough to zero out the disparity yields undefined depth,
    # not a huge finite number
    d80 = calib.fB / 80.0
    far2, near2 = calib.depth_error_exact(80.0, d80 + 1.0)
    assert np.isinf(far2), far2
    assert np.isfinite(near2) and near2 < 80.0, near2

    print("stereo geometry self-check passed")


if __name__ == "__main__":
    demo()
