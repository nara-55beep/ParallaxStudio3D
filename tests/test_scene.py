from dataclasses import replace
from pathlib import Path

import numpy as np

import cv2

from parallax3d.analyzer import create_auto_project
from parallax3d.renderer import DepthCameraRenderer, render_scene
from parallax3d.scene import load_scene
from parallax3d.segmentation import DetectedLayer


ROOT = Path(__file__).resolve().parents[1]


def _fake_detector(image, maximum_layers, status_callback):
    height, width = image.shape[:2]
    back = np.zeros((height, width), dtype=np.uint8)
    back[: height // 2] = 255
    front = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(front, (width // 2, height // 2), min(height, width) // 4, 255, -1)
    return [
        DetectedLayer("sky_01", "sky", 0.98, 0.035, "sky", 1.35, False, back),
        DetectedLayer(
            "object_02", "object", 0.95, 0.8, "camera", 1.0, True, front
        ),
    ]


def test_demo_scene_loads() -> None:
    scene = load_scene(ROOT / "scenes" / "child_mother.json")
    assert scene.source.exists()
    assert scene.clean_background.exists()
    assert [region.name for region in scene.regions] == [
        "yurt",
        "left_family",
        "mother",
        "foreground_child",
    ]
    assert sorted(region.depth for region in scene.regions) == [
        region.depth for region in scene.regions
    ]


def test_renderer_creates_real_depth_change(tmp_path: Path) -> None:
    scene = load_scene(ROOT / "scenes" / "child_mother.json")
    scene = replace(scene, render=replace(scene.render, width=320, height=180))
    renderer = DepthCameraRenderer(scene)
    rest = renderer.render_frame(0, 5)
    pushed = renderer.render_frame(2, 5)
    assert rest.shape == (180, 320, 3)
    assert pushed.shape == rest.shape
    assert np.mean(np.abs(rest.astype(np.int16) - pushed.astype(np.int16))) > 3.0
    renderer.save_diagnostics(tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "05_foreground_child.png").exists()


def test_one_image_render_outputs_only_video_and_gif(tmp_path: Path) -> None:
    source = np.full((180, 320, 3), 210, dtype=np.uint8)
    cv2.circle(source, (160, 95), 42, (40, 55, 70), -1)
    input_path = tmp_path / "input.png"
    cv2.imwrite(str(input_path), source)
    scene = load_scene(
        create_auto_project(
            input_path,
            tmp_path / "project",
            layer_detector=_fake_detector,
            background_builder=lambda image, layers, callback: image.copy(),
        )
    )
    scene = replace(
        scene,
        render=replace(
            scene.render,
            width=160,
            height=90,
            fps=2,
            duration=1.0,
            particles=0,
            motion_blur=0.0,
        ),
    )
    output = tmp_path / "render"
    result = render_scene(scene, output)
    assert result.video.name == "parallax.mp4"
    assert result.gif.name == "parallax.gif"
    assert {path.name for path in output.iterdir()} == {
        "parallax.mp4",
        "parallax.gif",
    }
