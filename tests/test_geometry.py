"""Stereo geometry checks.

The disparity-to-depth conversion is the deterministic half of the system, so
its failures are silent rather than loud: a wrong baseline or a clamped invalid
pixel produces a depth map that looks entirely plausible and is wrong. These
tests pin the behaviour that matters.
"""

import numpy as np
import pytest

from src.geometry.stereo import StereoCalibration, demo, parse_kitti_cam_to_cam

KITTI_CALIB = "data/kitti2015/training/calib_cam_to_cam/000000.txt"


def test_geometry_self_check():
    demo()


def test_kitti_calibration_matches_published_values():
    """KITTI's rectified colour pair is documented as roughly 721 px focal
    length and a 0.54 m baseline. Parsing must land there, or the whole depth
    evaluation is silently scaled wrong."""
    from pathlib import Path

    path = Path(KITTI_CALIB)
    if not path.exists():
        pytest.skip("KITTI 2015 calibration not downloaded")
    calib = parse_kitti_cam_to_cam(path)
    assert 700.0 < calib.focal_px < 730.0, calib.focal_px
    assert 0.52 < calib.baseline_m < 0.55, calib.baseline_m
    assert (calib.width, calib.height) == (1242, 375)


def test_invalid_disparity_never_becomes_valid_depth():
    calib = StereoCalibration(721.5377, 0.5327, 609.6, 172.9, 1242, 375)
    disparity = np.array([[40.0, 0.0], [-3.0, 1e-9]])
    depth, valid = calib.depth_from_disparity(disparity)
    assert valid.sum() == 1
    assert np.isfinite(depth[valid]).all()
    assert np.isnan(depth[~valid]).all()


def test_depth_error_grows_with_the_square_of_depth():
    calib = StereoCalibration(721.5377, 0.5327, 609.6, 172.9, 1242, 375)
    e10 = calib.depth_error_from_disparity_error(10.0, 1.0)
    e20 = calib.depth_error_from_disparity_error(20.0, 1.0)
    e80 = calib.depth_error_from_disparity_error(80.0, 1.0)
    assert np.isclose(e20 / e10, 4.0)
    assert np.isclose(e80 / e10, 64.0)
