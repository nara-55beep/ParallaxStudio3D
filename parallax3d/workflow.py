from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

from .analyzer import create_auto_project
from .renderer import RenderResult, render_scene
from .scene import load_scene


def render_image(
    image_path: str | Path,
    output_dir: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RenderResult:
    """Analyze and render one image while keeping only MP4 and GIF outputs."""
    with tempfile.TemporaryDirectory(prefix="parallaxstudio3d_") as temporary:
        scene_path = create_auto_project(image_path, Path(temporary) / "project")
        return render_scene(load_scene(scene_path), output_dir, progress_callback)
