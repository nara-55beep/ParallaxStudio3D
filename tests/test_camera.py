import math

from parallax3d.renderer import camera_envelope, scale_for_depth


def test_camera_returns_to_start() -> None:
    first, _ = camera_envelope(0, 101)
    middle, _ = camera_envelope(50, 101)
    last, _ = camera_envelope(100, 101)
    assert first == 0.0
    assert math.isclose(middle, 1.0)
    assert math.isclose(last, 0.0, abs_tol=1e-12)


def test_near_layers_move_more_than_far_layers() -> None:
    far = scale_for_depth(0.1, 0.17, 1.0)
    near = scale_for_depth(0.9, 0.17, 1.0)
    assert 1.0 < far < near


def test_velocity_reverses() -> None:
    _, going_in = camera_envelope(20, 101)
    _, going_out = camera_envelope(80, 101)
    assert going_in > 0
    assert going_out < 0
