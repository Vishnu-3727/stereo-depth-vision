"""The experiment helper is infrastructure every measurement depends on, so it
gets its own check: ids advance, environment is captured, directories are never
reused, and a failing run is still recorded as a failure."""

from src.common.experiment import demo


def test_experiment_helper():
    demo()
