"""Unit tests for tartan_imu.utils.error_handling.is_empty_trajectory_error.

Guards the eval-skip fix: a single trajectory that yields no valid windows must
be recognisable so the tester can skip it instead of aborting the whole run.
"""

from tartan_imu.utils.error_handling import is_empty_trajectory_error


def test_matches_airlab_dataloader_message():
    exc = ValueError(
        "No valid data files found in the provided paths. Please check your data directory."
    )
    assert is_empty_trajectory_error(exc) is True


def test_matches_humanoid_dataloader_message():
    assert is_empty_trajectory_error(ValueError("No valid data files found.")) is True


def test_rejects_unrelated_value_error():
    assert is_empty_trajectory_error(ValueError("shape mismatch")) is False


def test_rejects_non_value_error_even_with_marker():
    assert is_empty_trajectory_error(RuntimeError("No valid data files found")) is False
