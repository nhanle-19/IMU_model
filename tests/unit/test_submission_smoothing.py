import numpy as np

from starter.tartanimu_submission import _moving_average_time


def test_moving_average_time_preserves_shape_and_smooths_2d_sequence():
    values = np.array(
        [
            [0.0, 0.0],
            [9.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    smoothed = _moving_average_time(values, 3)

    assert smoothed.shape == values.shape
    np.testing.assert_allclose(smoothed[:, 0], [3.0, 3.0, 3.0], atol=1e-6)


def test_moving_average_time_smooths_logits_inside_each_sequence():
    logits = np.array(
        [
            [
                [4.0, 0.0],
                [0.0, 6.0],
                [4.0, 0.0],
            ],
            [
                [0.0, 4.0],
                [6.0, 0.0],
                [0.0, 4.0],
            ],
        ],
        dtype=np.float32,
    )

    smoothed = _moving_average_time(logits, 3)
    predicted = np.argmax(smoothed, axis=-1)

    assert smoothed.shape == logits.shape
    np.testing.assert_array_equal(predicted[0], [0, 0, 0])
    np.testing.assert_array_equal(predicted[1], [1, 1, 1])


def test_moving_average_time_kernel_one_is_noop():
    values = np.random.default_rng(0).normal(size=(2, 5, 3)).astype(np.float32)

    smoothed = _moving_average_time(values, 1)

    assert smoothed is values
