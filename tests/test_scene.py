from dataclasses import replace
from pathlib import Path

import numpy as np

from parallax3d.renderer import DepthCameraRenderer
from parallax3d.scene import load_scene


ROOT = Path(__file__).resolve().parents[1]


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
