from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    focus: tuple[float, float] = (0.56, 0.43)
    push: float = 0.08
    truck: float = 0.040
    rise: float = 0.016
    sky_drift: float = 0.040


@dataclass(frozen=True)
class RegionSpec:
    name: str
    depth: float
    rect: tuple[float, float, float, float]
    foreground_points: tuple[tuple[float, float], ...] = ()
    background_points: tuple[tuple[float, float], ...] = ()
    clip_polygon: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class RenderSpec:
    width: int = 1280
    height: int = 720
    fps: int = 24
    duration: float = 6.0
    particles: int = 150
    motion_blur: float = 0.55


@dataclass(frozen=True)
class SceneSpec:
    source: Path
    clean_background: Path | None = None
    depth_map: Path | None = None
    mode: str = "dense"
    camera: CameraSpec = field(default_factory=CameraSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    regions: tuple[RegionSpec, ...] = ()


def _pairs(values: list[list[float]] | None) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in (values or []))


def load_scene(path: str | Path) -> SceneSpec:
    scene_path = Path(path).resolve()
    data: dict[str, Any] = json.loads(scene_path.read_text(encoding="utf-8"))
    root = scene_path.parent

    def asset(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    def optional_asset(value: str | None) -> Path | None:
        return asset(value) if value else None

    camera_data = data.get("camera", {})
    render_data = data.get("render", {})
    regions = tuple(
        RegionSpec(
            name=item["name"],
            depth=float(item["depth"]),
            rect=tuple(float(v) for v in item["rect"]),
            foreground_points=_pairs(item.get("foreground_points")),
            background_points=_pairs(item.get("background_points")),
            clip_polygon=_pairs(item.get("clip_polygon")),
        )
        for item in data.get("regions", [])
    )
    return SceneSpec(
        source=asset(data["source"]),
        clean_background=optional_asset(data.get("clean_background")),
        depth_map=optional_asset(data.get("depth_map")),
        mode=str(data.get("mode", "planes" if regions else "dense")),
        camera=CameraSpec(
            focus=tuple(float(v) for v in camera_data.get("focus", [0.56, 0.43])),
            push=float(camera_data.get("push", 0.08)),
            truck=float(camera_data.get("truck", 0.040)),
            rise=float(camera_data.get("rise", 0.016)),
            sky_drift=float(camera_data.get("sky_drift", 0.040)),
        ),
        render=RenderSpec(
            width=int(render_data.get("width", 1280)),
            height=int(render_data.get("height", 720)),
            fps=int(render_data.get("fps", 24)),
            duration=float(render_data.get("duration", 6.0)),
            particles=int(render_data.get("particles", 150)),
            motion_blur=float(render_data.get("motion_blur", 0.55)),
        ),
        regions=regions,
    )


def read_bgr(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def read_depth(path: Path, size: tuple[int, int]) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth map: {path}")
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_CUBIC)
    return cv2.bilateralFilter(depth, 9, 28, 28).astype(np.float32) / 255.0


def _pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(point[0] * width, 0, width - 1)),
        int(np.clip(point[1] * height, 0, height - 1)),
    )


def extract_mask(image: np.ndarray, region: RegionSpec) -> np.ndarray:
    """Extract a soft object matte using a constrained GrabCut proposal."""
    height, width = image.shape[:2]
    x, y, rw, rh = region.rect
    left = int(np.clip(x * width, 0, width - 2))
    top = int(np.clip(y * height, 0, height - 2))
    right = int(np.clip((x + rw) * width, left + 1, width))
    bottom = int(np.clip((y + rh) * height, top + 1, height))

    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    mask[top:bottom, left:right] = cv2.GC_PR_FGD

    if region.clip_polygon:
        allowed = np.zeros_like(mask)
        polygon = np.array(
            [_pixel(point, width, height) for point in region.clip_polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(allowed, [polygon], 255)
        mask[allowed == 0] = cv2.GC_BGD

    radius = max(4, round(min(width, height) * 0.009))
    for point in region.foreground_points:
        cv2.circle(mask, _pixel(point, width, height), radius, cv2.GC_FGD, -1)
    for point in region.background_points:
        cv2.circle(mask, _pixel(point, width, height), radius, cv2.GC_BGD, -1)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bg_model, fg_model, 7, cv2.GC_INIT_WITH_MASK)
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.GaussianBlur(binary, (0, 0), 1.25)


def build_planes(scene: SceneSpec) -> tuple[np.ndarray, np.ndarray, list[tuple[RegionSpec, np.ndarray]]]:
    size = (scene.render.width, scene.render.height)
    source = read_bgr(scene.source, size)
    background = (
        read_bgr(scene.clean_background, size)
        if scene.clean_background is not None
        else source.copy()
    )
    planes: list[tuple[RegionSpec, np.ndarray]] = []
    for region in scene.regions:
        alpha = extract_mask(source, region)
        rgba = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        planes.append((region, rgba))
    return source, background, planes
