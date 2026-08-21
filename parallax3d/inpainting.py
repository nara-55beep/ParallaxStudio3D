from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# TorchScript packaging by simple-lama-inpainting (Apache-2.0), based on the
# official LaMa project (Apache-2.0): https://github.com/advimman/lama
LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/"
    "download/v0.1.0/big-lama.pt"
)
LAMA_MODEL_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


def _model_path(torch_module) -> Path:
    path = Path(torch_module.hub.get_dir()) / "checkpoints" / "big-lama.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        torch_module.hub.download_url_to_file(
            LAMA_MODEL_URL,
            str(path),
            progress=True,
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != LAMA_MODEL_SHA256:
        raise RuntimeError(
            f"LaMa model checksum failed: {path}. Delete that one cache file "
            "and run the application again."
        )
    return path


def _pad(array: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.pad(
        array,
        ((0, 0), (0, height - array.shape[1]), (0, width - array.shape[2])),
        mode="symmetric",
    )


def inpaint_background(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    status_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Reconstruct large hidden regions with the local LaMa TorchScript model."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "LaMa background reconstruction requires PyTorch. Run "
            "start_parallax_studio.bat to install the AI dependencies."
        ) from exc
    if status_callback is not None:
        status_callback("Reconstructing hidden background with LaMa...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(_model_path(torch)), map_location=device)
    model.eval().to(device)

    height, width = image_bgr.shape[:2]
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    mask_chw = (mask.astype(np.float32) / 255.0)[None, :, :]
    image_tensor = torch.from_numpy(
        _pad(image_chw, padded_height, padded_width)
    ).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(
        _pad(mask_chw, padded_height, padded_width)
    ).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor > 0.05).to(image_tensor.dtype)
    with torch.inference_mode():
        prediction = model(image_tensor, mask_tensor)
    result_rgb = (
        prediction[0, :, :height, :width]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    result_bgr = cv2.cvtColor(
        np.clip(result_rgb * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    feather = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
    feather = feather[:, :, None]
    blended = (
        result_bgr.astype(np.float32) * feather
        + image_bgr.astype(np.float32) * (1.0 - feather)
    )
    del model, prediction, image_tensor, mask_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.clip(blended, 0, 255).astype(np.uint8)
