import json
from pathlib import Path

import cv2
import numpy as np

from parallax3d.analyzer import create_auto_project
from parallax3d.scene import load_scene


def test_auto_project_writes_portable_scene(tmp_path: Path) -> None:
    source = np.full((180, 320, 3), 220, dtype=np.uint8)
    cv2.rectangle(source, (115, 35), (210, 160), (20, 30, 45), -1)
    input_path = tmp_path / "input.png"
    cv2.imwrite(str(input_path), source)

    scene_path = create_auto_project(input_path, tmp_path / "project")
    data = json.loads(scene_path.read_text(encoding="utf-8"))
    assert data["mode"] == "dense"
    assert data["regions"] == []
    assert (scene_path.parent.parent / "assets" / "depth_map.png").exists()
    assert not (
        scene_path.parent.parent / "assets" / "subject_proposals.png"
    ).exists()
    scene = load_scene(scene_path)
    assert scene.depth_map is not None
    assert scene.depth_map.exists()
