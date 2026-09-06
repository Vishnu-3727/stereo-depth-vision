"""Metric and dataset protocol checks.

The evaluation protocol is the thing most likely to be silently wrong, and a
wrong protocol produces a plausible number rather than an error. These tests
pin the two conventions apart.
"""

from src.datasets.kitti2015 import demo as kitti_demo
from src.evaluation.metrics import demo as metrics_demo


def test_metrics_self_check():
    metrics_demo()


def test_kitti_preprocessing_self_check():
    kitti_demo()
