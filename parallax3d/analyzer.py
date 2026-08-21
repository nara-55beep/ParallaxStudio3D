from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Callable

import cv2
import numpy as np

from .segmentation import DetectedLayer, detect_layers
from .inpainting import inpaint_background


LayerDetector = Callable[[np.ndarray, int, Callable[[str], None] | None], list[DetectedLayer]]


def _spectral_saliency(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    average = cv2.blur(log_amplitude, (5, 5))
    residual = log_amplitude - average
    restored = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(restored) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.resize(
        saliency.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )


def estimate_depth(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Estimate a smooth support depth map for the reconstructed background."""
    height, width = image.shape[:2]
    saliency = _spectral_saliency(image).astype(np.float32) / 255.0
    saliency = cv2.GaussianBlur(
        saliency,
        (0, 0),
        sigmaX=max(width, height) * 0.035,
        sigmaY=max(width, height) * 0.035,
    )
    if float(saliency.max()) > float(saliency.min()):
        saliency = cv2.normalize(saliency, None, 0.0, 1.0, cv2.NORM_MINMAX)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    perspective = np.interp(
        y,
        [0.0, 0.20, 0.42, 0.65, 1.0],
        [0.03, 0.07, 0.22, 0.53, 0.96],
    ).astype(np.float32)
    perspective = np.repeat(perspective, width, axis=1)
    depth = np.clip(0.76 * perspective + 0.24 * saliency, 0.0, 1.0)
    depth_u8 = cv2.bilateralFilter(
        np.round(depth * 255.0).astype(np.uint8), 15, 32, 32
    )
    weights = np.maximum(saliency - np.percentile(saliency, 62), 0.0)
    total = float(weights.sum())
    if total > 1e-6:
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        focus = (
            float(np.clip((grid_x * weights).sum() / total / width, 0.30, 0.70)),
            float(np.clip((grid_y * weights).sum() / total / height, 0.30, 0.62)),
        )
    else:
        focus = (0.5, 0.45)
    return depth_u8, focus


def _background_plate(
    image: np.ndarray,
    layers: list[DetectedLayer],
    status_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    erase = np.zeros(image.shape[:2], dtype=np.uint8)
    for layer in layers:
        if layer.erase_from_background:
            erase = np.maximum(erase, layer.mask)
    if not np.any(erase > 24):
        return image.copy()
    radius = max(3, round(min(image.shape[:2]) * 0.007))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    holes = cv2.dilate((erase > 24).astype(np.uint8) * 255, kernel, iterations=1)
    return inpaint_background(image, holes, status_callback)


def create_auto_project(
    image_path: str | Path,
    output_dir: str | Path,
    maximum_layers: int = 12,
    status_callback: Callable[[str], None] | None = None,
    layer_detector: LayerDetector | None = None,
    background_builder: Callable[
        [np.ndarray, list[DetectedLayer], Callable[[str], None] | None], np.ndarray
    ]
    | None = None,
) -> Path:
    """Create an editable, portable scene with AI-cut transparent planes."""
    source_path = Path(image_path).resolve()
    output = Path(output_dir).resolve()
    assets = output / "assets"
    masks_dir = assets / "masks"
    scenes = output / "scenes"
    masks_dir.mkdir(parents=True, exist_ok=True)
    scenes.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    detector = layer_detector or detect_layers
    layers = detector(image, maximum_layers, status_callback)
    if not layers:
        raise RuntimeError("No independent parallax layers were detected.")

    source_copy = assets / f"source{source_path.suffix.lower() or '.png'}"
    shutil.copy2(source_path, source_copy)
    depth, focus = estimate_depth(image)
    depth_path = assets / "depth_map.png"
    background_path = assets / "background.png"
    cv2.imwrite(str(depth_path), depth)
    build_background = background_builder or _background_plate
    cv2.imwrite(
        str(background_path),
        build_background(image, layers, status_callback),
    )

    regions: list[dict[str, object]] = []
    for layer in layers:
        mask_path = masks_dir / f"{layer.name}.png"
        cv2.imwrite(str(mask_path), layer.mask)
        regions.append(
            {
                "name": layer.name,
                "label": layer.label,
                "depth": round(layer.depth, 5),
                "rect": [0.0, 0.0, 1.0, 1.0],
                "mask": f"../assets/masks/{mask_path.name}",
                "motion": layer.motion,
                "motion_strength": round(layer.motion_strength, 4),
                "score": round(layer.score, 5),
            }
        )

    data = {
        "mode": "planes",
        "source": f"../assets/{source_copy.name}",
        "clean_background": "../assets/background.png",
        "depth_map": "../assets/depth_map.png",
        "camera": {
            "focus": [round(focus[0], 5), round(focus[1], 5)],
            "push": 0.065,
            "truck": 0.028,
            "rise": 0.012,
            "sky_drift": 0.032,
        },
        "render": {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "duration": 6.0,
            "particles": 55,
            "motion_blur": 0.24,
        },
        "regions": regions,
    }
    scene_path = scenes / "auto_scene.json"
    scene_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return scene_path
