from __future__ import annotations

import argparse
from pathlib import Path

from .renderer import DepthCameraRenderer, render_scene
from .scene import load_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parallax3d",
        description="Render a single image as a unified-camera 2.5D scene.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="Render only MP4 and GIF from a scene")
    render.add_argument("scene", type=Path)
    render.add_argument("--output", type=Path, default=Path("output/demo"))
    inspect = sub.add_parser("inspect", help="Extract and save scene layers only")
    inspect.add_argument("scene", type=Path)
    inspect.add_argument("--output", type=Path, default=Path("output/inspection"))
    auto = sub.add_parser("auto", help="Create a free local draft scene from one image")
    auto.add_argument("image", type=Path)
    auto.add_argument("--project", type=Path, default=Path("output/auto_project"))
    auto.add_argument("--layers", type=int, default=4)
    image = sub.add_parser("image", help="Analyze and render one image to MP4 + GIF")
    image.add_argument("source", type=Path)
    image.add_argument("--output", type=Path, default=Path("output/render"))
    sub.add_parser("app", help="Open the desktop interface")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "app":
        from .app import launch

        launch()
        return 0
    if args.command == "auto":
        from .analyzer import create_auto_project

        scene_path = create_auto_project(args.image, args.project, args.layers)
        print(f"Draft scene: {scene_path}")
        print(f"Render with: python -m parallax3d render \"{scene_path}\"")
        return 0
    if args.command == "image":
        from .workflow import render_image

        def report(done: int, total: int) -> None:
            if done == 1 or done == total or done % max(1, total // 10) == 0:
                print(f"Rendering {done}/{total}")

        result = render_image(args.source, args.output, report)
        print(f"MP4: {result.video}")
        print(f"GIF: {result.gif}")
        return 0
    scene = load_scene(args.scene)
    if args.command == "inspect":
        renderer = DepthCameraRenderer(scene)
        renderer.save_diagnostics(args.output)
        print(f"Layers saved to {args.output.resolve()}")
        return 0

    def report(done: int, total: int) -> None:
        if done == 1 or done == total or done % max(1, total // 10) == 0:
            print(f"Rendering {done}/{total}")

    result = render_scene(scene, args.output, report)
    print(f"MP4: {result.video}")
    print(f"GIF: {result.gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
