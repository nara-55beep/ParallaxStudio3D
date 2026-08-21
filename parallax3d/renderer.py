from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .scene import RegionSpec, SceneSpec, build_planes, read_depth


ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class RenderResult:
    video: Path
    gif: Path


@dataclass(frozen=True)
class CameraState:
    progress: float
    velocity: float
    truck: float
    sky: float


def camera_envelope(frame_index: int, frame_count: int) -> tuple[float, float]:
    """Return reversible push amount and signed velocity."""
    t = frame_index / max(frame_count - 1, 1)
    progress = math.sin(math.pi * t) ** 2
    velocity = math.pi * math.sin(2.0 * math.pi * t)
    return progress, velocity


def camera_path(frame_index: int, frame_count: int) -> CameraState:
    """A looping dolly plus side-to-side truck and independent sky phase."""
    progress, velocity = camera_envelope(frame_index, frame_count)
    t = frame_index / max(frame_count - 1, 1)
    phase = math.sin(2.0 * math.pi * t)
    sky_phase = math.sin(4.0 * math.pi * t)
    return CameraState(progress, velocity, phase, sky_phase)


def scale_for_depth(depth: float, push: float, progress: float) -> float:
    return 1.0 + push * progress * (0.10 + 0.95 * float(depth))


def _focus_px(scene: SceneSpec) -> tuple[float, float]:
    return (
        scene.camera.focus[0] * scene.render.width,
        scene.camera.focus[1] * scene.render.height,
    )


class DepthCameraRenderer:
    def __init__(self, scene: SceneSpec):
        self.scene = scene
        self.source, self.background, self.planes = build_planes(scene)
        self.height, self.width = self.background.shape[:2]
        self.focus = _focus_px(scene)
        grid_y, grid_x = np.indices((self.height, self.width), dtype=np.float32)
        self.grid_x = grid_x
        self.grid_y = grid_y
        y_norm = grid_y / max(self.height - 1, 1)
        self.y_norm = y_norm
        if scene.depth_map is not None:
            self.depth = read_depth(scene.depth_map, (self.width, self.height))
        else:
            self.depth = np.interp(
                y_norm,
                [0.00, 0.18, 0.35, 0.55, 0.75, 1.00],
                [0.02, 0.05, 0.16, 0.33, 0.65, 0.98],
            ).astype(np.float32)
        self.particles = self._make_particles(scene.render.particles)

    def _make_particles(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(90421)
        particles = np.empty((count, 5), dtype=np.float32)
        particles[:, 0] = rng.uniform(-0.08, 1.08, count)
        particles[:, 1] = rng.uniform(-0.08, 1.08, count)
        particles[:, 2] = rng.uniform(0.08, 1.0, count)
        particles[:, 3] = rng.uniform(0.55, 1.75, count)
        particles[:, 4] = rng.uniform(0.12, 0.55, count)
        return particles

    def _warp_background(self, state: CameraState) -> np.ndarray:
        scale = 1.0 + self.scene.camera.push * state.progress * (
            0.10 + 0.95 * self.depth
        )
        focus_x, focus_y = self.focus
        map_x = focus_x + (self.grid_x - focus_x) / scale
        map_y = focus_y + (self.grid_y - focus_y) / scale
        response = 0.12 + 0.88 * self.depth
        map_x += self.scene.camera.truck * self.width * state.truck * response
        map_y += self.scene.camera.rise * self.height * state.progress * response

        # Clouds/sky have their own faster lateral motion. The smooth weight
        # avoids a hard horizon seam while retaining stronger movement aloft.
        sky_weight = np.clip((0.62 - self.y_norm) / 0.62, 0.0, 1.0)
        sky_weight = sky_weight * sky_weight * (3.0 - 2.0 * sky_weight)
        background_sky_strength = 0.28 if self.scene.mode == "planes" else 1.0
        map_x += (
            self.scene.camera.sky_drift
            * self.width
            * state.sky
            * sky_weight
            * background_sky_strength
        )
        return cv2.remap(
            self.source if self.scene.mode == "dense" else self.background,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _warp_plane(
        self, plane: np.ndarray, region: RegionSpec, state: CameraState
    ) -> np.ndarray:
        strength = max(0.0, region.motion_strength)
        if region.motion == "sky":
            scale = 1.0 + self.scene.camera.push * state.progress * 0.08
            shift_x = (
                self.scene.camera.sky_drift
                * self.width
                * state.sky
                * strength
            )
            shift_y = 0.0
        else:
            scale = scale_for_depth(
                region.depth,
                self.scene.camera.push * strength,
                state.progress,
            )
            response = (0.12 + 0.88 * region.depth) * strength
            shift_x = (
                self.scene.camera.truck
                * self.width
                * state.truck
                * response
            )
            shift_y = (
                self.scene.camera.rise
                * self.height
                * state.progress
                * response
            )
            if region.motion == "sway":
                shift_x += (
                    self.scene.camera.sky_drift
                    * self.width
                    * state.sky
                    * 0.16
                    * strength
                )
        focus_x, focus_y = self.focus
        matrix = np.array(
            [[scale, 0.0, (1.0 - scale) * focus_x - shift_x],
             [0.0, scale, (1.0 - scale) * focus_y - shift_y]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            plane,
            matrix,
            (self.width, self.height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def _over(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(
            foreground[:, :, :3].astype(np.float32) * alpha
            + background.astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)

    def _draw_particles(
        self, frame: np.ndarray, progress: float, velocity: float, frame_index: int
    ) -> np.ndarray:
        if self.particles.size == 0:
            return frame
        overlay = frame.copy()
        focus_x, focus_y = self.focus
        time = frame_index / self.scene.render.fps
        speed = min(abs(velocity) / math.pi, 1.0)
        for px, py, depth, radius, opacity in self.particles:
            scale = scale_for_depth(float(depth), self.scene.camera.push, progress)
            x0 = float(px) * self.width
            y0 = ((float(py) + time * (0.012 + 0.016 * float(depth))) % 1.16 - 0.08) * self.height
            x = int(round(focus_x + (x0 - focus_x) * scale))
            y = int(round(focus_y + (y0 - focus_y) * scale))
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            r = max(1, int(round(float(radius) * (0.8 + 1.6 * float(depth)))))
            color = (224, 222, 215)
            if speed > 0.12 and depth > 0.45:
                vx = x - focus_x
                vy = y - focus_y
                length = math.hypot(vx, vy) or 1.0
                streak = int(2 + 11 * speed * float(depth))
                sign = 1 if velocity >= 0 else -1
                end = (
                    int(x + sign * streak * vx / length),
                    int(y + sign * streak * vy / length),
                )
                cv2.line(overlay, (x, y), end, color, r, cv2.LINE_AA)
            else:
                cv2.circle(overlay, (x, y), r, color, -1, cv2.LINE_AA)
        amount = float(np.mean(self.particles[:, 4])) * 0.42
        return cv2.addWeighted(overlay, amount, frame, 1.0 - amount, 0)

    def _radial_motion_blur(self, frame: np.ndarray, velocity: float) -> np.ndarray:
        strength = min(abs(velocity) / math.pi, 1.0) * self.scene.render.motion_blur
        if strength < 0.03:
            return frame
        focus_x, focus_y = self.focus
        direction = 1.0 if velocity >= 0 else -1.0
        result = frame.astype(np.float32) * 0.68
        weights = (0.16, 0.10, 0.06)
        for index, weight in enumerate(weights, start=1):
            scale = 1.0 + direction * strength * 0.0035 * index
            matrix = np.array(
                [[scale, 0.0, (1.0 - scale) * focus_x],
                 [0.0, scale, (1.0 - scale) * focus_y]],
                dtype=np.float32,
            )
            sample = cv2.warpAffine(
                frame,
                matrix,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            result += sample.astype(np.float32) * weight
        return np.clip(result, 0, 255).astype(np.uint8)

    def render_frame(self, frame_index: int, frame_count: int) -> np.ndarray:
        state = camera_path(frame_index, frame_count)
        frame = self._warp_background(state)
        for region, plane in sorted(self.planes, key=lambda item: item[0].depth):
            frame = self._over(frame, self._warp_plane(plane, region, state))
        frame = self._draw_particles(
            frame, state.progress, state.velocity, frame_index
        )
        return self._radial_motion_blur(frame, state.velocity)

    def save_diagnostics(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "00_background.png"), self.background)
        cv2.imwrite(str(directory / "01_depth_map.png"), (self.depth * 255).astype(np.uint8))
        composite = self.background.copy()
        for index, (region, plane) in enumerate(
            sorted(self.planes, key=lambda item: item[0].depth), start=2
        ):
            layer_name = f"{index:02d}_{region.name}.png"
            cv2.imwrite(str(directory / layer_name), plane)
            composite = self._over(composite, plane)
        cv2.imwrite(str(directory / "composite_at_rest.png"), composite)
        manifest = {
            "format": "parallax-studio-3d/1",
            "width": self.width,
            "height": self.height,
            "focus": list(self.scene.camera.focus),
            "push": self.scene.camera.push,
            "background": "00_background.png",
            "depth_map": "01_depth_map.png",
            "layers": [
                {
                    "name": region.name,
                    "label": region.label,
                    "file": f"{index:02d}_{region.name}.png",
                    "depth": region.depth,
                    "motion": region.motion,
                    "motion_strength": region.motion_strength,
                    "score": region.score,
                }
                for index, (region, _) in enumerate(
                    sorted(self.planes, key=lambda item: item[0].depth), start=2
                )
            ],
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def render_scene(
    scene: SceneSpec,
    output_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> RenderResult:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer = DepthCameraRenderer(scene)

    frame_count = max(2, int(round(scene.render.duration * scene.render.fps)))
    video_path = output / "parallax.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        scene.render.fps,
        (scene.render.width, scene.render.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize an MP4 video writer")

    gif_frames: list[Image.Image] = []
    gif_step = max(1, scene.render.fps // 12)
    try:
        for index in range(frame_count):
            frame = renderer.render_frame(index, frame_count)
            writer.write(frame)
            if index % gif_step == 0:
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            if progress_callback:
                progress_callback(index + 1, frame_count)
    finally:
        writer.release()

    gif_path = output / "parallax.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 * gif_step / scene.render.fps),
        loop=0,
        optimize=False,
    )
    return RenderResult(video_path, gif_path)
