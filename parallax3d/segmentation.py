from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Callable

import cv2
import numpy as np
from PIL import Image


ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_swin_tiny"
SAM_MODEL = "facebook/sam2.1-hiera-small"

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class DetectedLayer:
    name: str
    label: str
    score: float
    depth: float
    motion: str
    motion_strength: float
    erase_from_background: bool
    mask: np.ndarray


@dataclass(frozen=True)
class _Candidate:
    label: str
    score: float
    mask: np.ndarray
    source: str


@dataclass(frozen=True)
class _SemanticResult:
    label_map: np.ndarray
    id_to_label: dict[int, str]
    candidates: tuple[_Candidate, ...]


_SKY_WORDS = ("sky", "cloud")
_SCENIC_WORDS = (
    "sky",
    "cloud",
    "mountain",
    "hill",
    "earth",
    "ground",
    "field",
    "grass",
    "plant",
    "tree",
    "road",
    "sea",
    "water",
    "river",
    "sand",
    "snow",
    "building",
    "wall",
    "house",
)
_VEGETATION_WORDS = ("tree", "plant", "grass", "flower", "bush")


def _status(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _ai_modules():
    try:
        import torch
        from transformers import (
            OneFormerForUniversalSegmentation,
            OneFormerProcessor,
            Sam2Processor,
            pipeline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "AI cutout support is not installed. Run start_parallax_studio.bat "
            "again, or run: python -m pip install -e \".[ai]\""
        ) from exc
    return (
        torch,
        OneFormerProcessor,
        OneFormerForUniversalSegmentation,
        Sam2Processor,
        pipeline,
    )


def _label(config: object, label_id: int) -> str:
    labels = getattr(config, "id2label", {})
    return str(labels.get(label_id, labels.get(str(label_id), f"region_{label_id}")))


def _semantic_scene(image: Image.Image, callback: StatusCallback | None) -> _SemanticResult:
    torch, processor_type, model_type, _, _ = _ai_modules()
    _status(callback, "Understanding people, sky, vegetation, terrain and objects...")
    processor = processor_type.from_pretrained(ONEFORMER_MODEL)
    model = model_type.from_pretrained(ONEFORMER_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
    )[0]
    segment_map = result["segmentation"].detach().cpu().numpy().astype(np.int32)
    label_map = np.full(segment_map.shape, -1, dtype=np.int32)
    id_to_label: dict[int, str] = {}
    candidates: list[_Candidate] = []
    for info in result["segments_info"]:
        label_id = int(info["label_id"])
        name = _label(model.config, label_id).lower()
        mask = segment_map == int(info["id"])
        label_map[mask] = label_id
        id_to_label[label_id] = name
        area = float(mask.mean())
        if 0.004 <= area <= 0.65 and any(word in name for word in _SCENIC_WORDS):
            candidates.append(
                _Candidate(name, float(info.get("score", 0.8)), mask, "semantic")
            )
    del outputs, model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _SemanticResult(label_map, id_to_label, tuple(candidates))


def _majority_label(mask: np.ndarray, semantic: _SemanticResult) -> str:
    values = semantic.label_map[mask]
    values = values[values >= 0]
    if values.size == 0:
        return "object"
    ids, counts = np.unique(values, return_counts=True)
    label_id = int(ids[int(np.argmax(counts))])
    label = semantic.id_to_label.get(label_id, "object")
    if any(word in label for word in _SKY_WORDS) and not _edge_flags(mask)[2]:
        return "distant terrain"
    return label


def _sam_candidates(
    image: Image.Image,
    semantic: _SemanticResult,
    callback: StatusCallback | None,
) -> list[_Candidate]:
    torch, _, _, sam_processor_type, pipeline = _ai_modules()
    _status(callback, "Cutting clean object masks with SAM 2...")
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("mask-generation", model=SAM_MODEL, device=device)
    points_per_side = int(os.environ.get("PARALLAX_SAM_POINTS", "20"))
    points_per_side = int(np.clip(points_per_side, 12, 32))
    result = generator(
        image,
        points_per_batch=64 if device >= 0 else 32,
        points_per_side=points_per_side,
        pred_iou_thresh=0.78,
        stability_score_thresh=0.86,
        crops_n_layers=0,
    )
    candidates: list[_Candidate] = []
    for raw_mask, score in zip(result["masks"], result["scores"]):
        if hasattr(raw_mask, "detach"):
            mask = raw_mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(raw_mask, dtype=bool)
        area = float(mask.mean())
        if not 0.0035 <= area <= 0.60:
            continue
        candidates.append(
            _Candidate(
                _majority_label(mask, semantic),
                float(score),
                mask,
                "sam2",
            )
        )

    prompt_points = _color_prompt_points(np.asarray(image))
    if prompt_points:
        _status(callback, "Refining small salient props and details...")
        processor = sam_processor_type.from_pretrained(SAM_MODEL)
        points = [[[[x, y]] for x, y in prompt_points]]
        labels = [[[1] for _ in prompt_points]]
        device_object = next(generator.model.parameters()).device
        inputs = processor(
            images=image,
            input_points=points,
            input_labels=labels,
            return_tensors="pt",
        ).to(device_object)
        with torch.inference_mode():
            prompted = generator.model(**inputs)
        masks = processor.post_process_masks(
            prompted.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        scores = prompted.iou_scores.detach().cpu()[0]
        for object_index in range(masks.shape[0]):
            best = int(torch.argmax(scores[object_index]).item())
            mask = masks[object_index, best].numpy().astype(bool)
            area = float(mask.mean())
            score = float(scores[object_index, best])
            if 0.002 <= area <= 0.20 and score >= 0.52:
                candidates.append(
                    _Candidate(
                        _majority_label(mask, semantic),
                        score,
                        mask,
                        "sam2_prompt",
                    )
                )
        del prompted, processor
    del generator, result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return candidates


def _color_prompt_points(image_rgb: np.ndarray) -> list[tuple[int, int]]:
    """Find small vivid regions that a coarse automatic point grid can miss."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    threshold = max(68, int(np.percentile(saturation, 88)))
    vivid = (
        (saturation >= threshold)
        & (value >= 35)
        & (value <= 250)
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vivid = cv2.morphologyEx(vivid, cv2.MORPH_OPEN, kernel, iterations=1)
    vivid = cv2.morphologyEx(vivid, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(vivid, 8)
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    components: list[tuple[float, int, tuple[int, int]]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        fraction = area / image_area
        if not 0.0007 <= fraction <= 0.10:
            continue
        component = labels == index
        center_x = float(
            stats[index, cv2.CC_STAT_LEFT]
            + stats[index, cv2.CC_STAT_WIDTH] / 2
        )
        center_y = float(
            stats[index, cv2.CC_STAT_TOP]
            + stats[index, cv2.CC_STAT_HEIGHT] / 2
        )
        ys, xs = np.nonzero(component)
        nearest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
        point = (int(xs[nearest]), int(ys[nearest]))
        components.append((float(saturation[component].mean()), area, point))
    components.sort(reverse=True)
    points: list[tuple[int, int]] = []
    minimum_distance = max(12, round(min(image_rgb.shape[:2]) * 0.035))
    for _, _, point in components:
        if any(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2
            < minimum_distance**2
            for other in points
        ):
            continue
        points.append(point)
        if len(points) == 8:
            break
    return points


def _edge_flags(mask: np.ndarray) -> tuple[bool, bool, bool, bool]:
    height, width = mask.shape
    edge_x = max(2, width // 100)
    edge_y = max(2, height // 100)
    threshold_x = height * edge_x * 0.08
    threshold_y = width * edge_y * 0.08
    return (
        np.count_nonzero(mask[:, :edge_x]) > threshold_x,
        np.count_nonzero(mask[:, -edge_x:]) > threshold_x,
        np.count_nonzero(mask[:edge_y, :]) > threshold_y,
        np.count_nonzero(mask[-edge_y:, :]) > threshold_y,
    )


def _is_scenic(candidate: _Candidate) -> bool:
    left, right, top, _ = _edge_flags(candidate.mask)
    label = candidate.label.lower()
    return (left and right) or (top and any(word in label for word in _SKY_WORDS)) or any(
        word in label for word in _SCENIC_WORDS
    )


def _duplicate(first: np.ndarray, second: np.ndarray) -> bool:
    intersection = int(np.count_nonzero(first & second))
    if intersection == 0:
        return False
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    containment = intersection / max(1, min(first_area, second_area))
    union = first_area + second_area - intersection
    return containment >= 0.82 or intersection / max(1, union) >= 0.72


def _deduplicate(candidates: list[_Candidate]) -> list[_Candidate]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.score + (0.015 if item.source.startswith("sam2") else 0.0),
            float(item.mask.mean()),
        ),
        reverse=True,
    )
    kept: list[_Candidate] = []
    for candidate in ranked:
        if any(_duplicate(candidate.mask, other.mask) for other in kept):
            continue
        kept.append(candidate)
    return kept


def _clean_binary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 2:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        minimum = max(12, int(largest * 0.008))
        keep = np.zeros_like(binary)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
                keep[labels == index] = 1
        binary = keep
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)


def _depth_for(candidate: _Candidate) -> float:
    ys = np.flatnonzero(np.any(candidate.mask, axis=1))
    bottom = float(ys[-1] / max(candidate.mask.shape[0] - 1, 1)) if ys.size else 0.5
    label = candidate.label.lower()
    if any(word in label for word in _SKY_WORDS):
        return 0.035
    if "mountain" in label or "hill" in label:
        return float(np.clip(0.10 + 0.24 * bottom, 0.12, 0.32))
    if any(word in label for word in ("building", "house", "wall")):
        return float(np.clip(0.20 + 0.35 * bottom, 0.28, 0.58))
    if _is_scenic(candidate):
        return float(np.clip(0.08 + 0.55 * bottom, 0.12, 0.66))
    return float(np.clip(0.26 + 0.72 * bottom, 0.38, 0.96))


def _alpha(mask: np.ndarray, scenic: bool) -> np.ndarray:
    binary = mask.astype(np.uint8) * 255
    if not scenic:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)
    return cv2.GaussianBlur(binary, (0, 0), 0.85)


def _safe_name(label: str, index: int) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"
    return f"{stem}_{index:02d}"


def select_layers(candidates: list[_Candidate], maximum_layers: int) -> list[DetectedLayer]:
    maximum_layers = max(2, int(maximum_layers))
    cleaned = [
        _Candidate(item.label, item.score, _clean_binary(item.mask), item.source)
        for item in candidates
    ]
    cleaned = [item for item in cleaned if float(item.mask.mean()) >= 0.0035]
    unique = _deduplicate(cleaned)
    compact = [item for item in unique if not _is_scenic(item)]
    scenic = [item for item in unique if _is_scenic(item)]

    compact_limit = max(2, int(round(maximum_layers * 0.65)))
    chosen = compact[:compact_limit]
    chosen.extend(scenic[: max(1, maximum_layers - len(chosen))])
    if len(chosen) < maximum_layers:
        chosen_ids = {id(item) for item in chosen}
        remaining = [item for item in unique if id(item) not in chosen_ids]
        chosen.extend(remaining[: maximum_layers - len(chosen)])

    layers: list[DetectedLayer] = []
    for index, candidate in enumerate(chosen[:maximum_layers], start=1):
        scenic_layer = _is_scenic(candidate)
        label = candidate.label.lower()
        if any(word in label for word in _SKY_WORDS):
            motion = "sky"
            strength = 1.35
        elif any(word in label for word in _VEGETATION_WORDS):
            motion = "sway"
            strength = 1.08
        else:
            motion = "camera"
            strength = 0.78 if scenic_layer else 1.0
        layers.append(
            DetectedLayer(
                name=_safe_name(candidate.label, index),
                label=candidate.label,
                score=candidate.score,
                depth=_depth_for(candidate),
                motion=motion,
                motion_strength=strength,
                erase_from_background=not scenic_layer,
                mask=_alpha(candidate.mask, scenic_layer),
            )
        )
    return sorted(layers, key=lambda layer: layer.depth)


def detect_layers(
    image_bgr: np.ndarray,
    maximum_layers: int = 12,
    status_callback: StatusCallback | None = None,
) -> list[DetectedLayer]:
    """Find semantic scene regions and clean object instances in one image."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    semantic = _semantic_scene(image, status_callback)
    candidates = list(semantic.candidates)
    candidates.extend(_sam_candidates(image, semantic, status_callback))
    layers = select_layers(candidates, maximum_layers)
    if len(layers) < 2:
        raise RuntimeError(
            "The AI could not find enough separate objects in this image. "
            "Try a clearer or higher-resolution source image."
        )
    _status(status_callback, f"Prepared {len(layers)} independent parallax layers.")
    return layers
