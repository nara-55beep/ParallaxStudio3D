from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


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


def _regions_from_saliency(image: np.ndarray, maximum: int = 4) -> tuple[list[dict], np.ndarray]:
    height, width = image.shape[:2]
    saliency = _spectral_saliency(image)
    threshold = int(np.percentile(saliency, 78))
    binary = np.where(saliency >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(5, width // 100), max(5, height // 100))
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates: list[tuple[int, int]] = []
    min_area = width * height * 0.006
    max_area = width * height * 0.42
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            candidates.append((area, index))
    candidates.sort(reverse=True)

    regions: list[dict] = []
    combined = np.zeros((height, width), dtype=np.uint8)
    for order, (_, index) in enumerate(candidates[:maximum], start=1):
        component = np.where(labels == index, 255, 0).astype(np.uint8)
        component = cv2.dilate(component, kernel, iterations=1)
        x, y, w, h, _ = stats[index]
        pad_x, pad_y = round(w * 0.08), round(h * 0.08)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        cx, cy = centroids[index]
        depth = float(np.clip(0.2 + 0.75 * cy / height, 0.2, 0.95))
        regions.append(
            {
                "name": f"subject_{order}",
                "depth": round(depth, 3),
                "rect": [
                    round(x0 / width, 5),
                    round(y0 / height, 5),
                    round((x1 - x0) / width, 5),
                    round((y1 - y0) / height, 5),
                ],
                "foreground_points": [[round(cx / width, 5), round(cy / height, 5)]],
                "background_points": [],
            }
        )
        combined = cv2.max(combined, component)
    return regions, combined


def create_auto_project(
    image_path: str | Path, output_dir: str | Path, maximum_layers: int = 4
) -> Path:
    """Create a free local draft scene. AI mattes/inpainting can replace its assets."""
    source_path = Path(image_path).resolve()
    output = Path(output_dir).resolve()
    assets = output / "assets"
    scenes = output / "scenes"
    assets.mkdir(parents=True, exist_ok=True)
    scenes.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")

    regions, combined_mask = _regions_from_saliency(image, maximum_layers)
    if not regions:
        height, width = image.shape[:2]
        regions = [{
            "name": "main_subject",
            "depth": 0.72,
            "rect": [0.2, 0.15, 0.6, 0.7],
            "foreground_points": [[0.5, 0.5]],
            "background_points": [],
        }]
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(
            combined_mask,
            (round(width * 0.2), round(height * 0.15)),
            (round(width * 0.8), round(height * 0.85)),
            255,
            -1,
        )

    source_copy = assets / f"source{source_path.suffix.lower() or '.png'}"
    source_copy.write_bytes(source_path.read_bytes())
    inpaint_mask = cv2.dilate(
        combined_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    )
    clean = cv2.inpaint(image, inpaint_mask, 7, cv2.INPAINT_TELEA)
    clean_path = assets / "clean_background.png"
    cv2.imwrite(str(clean_path), clean)
    cv2.imwrite(str(assets / "subject_proposals.png"), combined_mask)
    cv2.imwrite(str(assets / "saliency.png"), _spectral_saliency(image))

    data = {
        "source": f"../assets/{source_copy.name}",
        "clean_background": "../assets/clean_background.png",
        "camera": {"focus": [0.5, 0.45], "push": 0.15},
        "render": {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "duration": 6.0,
            "particles": 120,
            "motion_blur": 0.5,
        },
        "regions": regions,
    }
    scene_path = scenes / "auto_scene.json"
    scene_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return scene_path
