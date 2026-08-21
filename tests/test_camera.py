import math

from parallax3d.renderer import camera_envelope, camera_path, scale_for_depth


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


def test_camera_trucks_both_directions_and_loops() -> None:
    start = camera_path(0, 101)
    right = camera_path(25, 101)
    middle = camera_path(50, 101)
    left = camera_path(75, 101)
    end = camera_path(100, 101)
    assert math.isclose(start.truck, 0.0, abs_tol=1e-12)
    assert right.truck > 0.9
    assert abs(right.sky) < 0.1
    assert math.isclose(middle.truck, 0.0, abs_tol=1e-12)
    assert left.truck < -0.9
    assert math.isclose(end.truck, 0.0, abs_tol=1e-12)
