import json
from pathlib import Path

import cv2
import numpy as np

from parallax3d.analyzer import create_auto_project
from parallax3d.scene import load_scene
from parallax3d.segmentation import DetectedLayer


def _fake_detector(image, maximum_layers, status_callback):
    height, width = image.shape[:2]
    sky = np.zeros((height, width), dtype=np.uint8)
    sky[: height // 3] = 255
    subject = np.zeros((height, width), dtype=np.uint8)
    subject[height // 4 : height - 12, width // 3 : width * 2 // 3] = 255
    return [
        DetectedLayer("sky_01", "sky", 0.98, 0.035, "sky", 1.35, False, sky),
        DetectedLayer(
            "person_02", "person", 0.96, 0.82, "camera", 1.0, True, subject
        ),
    ]


def test_auto_project_writes_portable_scene(tmp_path: Path) -> None:
    source = np.full((180, 320, 3), 220, dtype=np.uint8)
    cv2.rectangle(source, (115, 35), (210, 160), (20, 30, 45), -1)
    input_path = tmp_path / "input.png"
    cv2.imwrite(str(input_path), source)

    scene_path = create_auto_project(
        input_path,
        tmp_path / "project",
        layer_detector=_fake_detector,
        background_builder=lambda image, layers, callback: image.copy(),
    )
    data = json.loads(scene_path.read_text(encoding="utf-8"))
    assert data["mode"] == "planes"
    assert [region["label"] for region in data["regions"]] == ["sky", "person"]
    assert (scene_path.parent.parent / "assets" / "depth_map.png").exists()
    assert (scene_path.parent.parent / "assets" / "background.png").exists()
    scene = load_scene(scene_path)
    assert scene.depth_map is not None
    assert scene.depth_map.exists()
    assert all(region.mask and region.mask.exists() for region in scene.regions)
    assert scene.regions[0].motion == "sky"
