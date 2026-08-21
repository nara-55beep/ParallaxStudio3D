"""Run from Blender's Scripting workspace and choose a generated manifest.json."""
import json
from pathlib import Path

import bpy
from bpy_extras.image_utils import load_image


def import_manifest(manifest_path: str) -> None:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [{"name": "background", "file": data["background"], "depth": 0.02}]
    entries.extend(data["layers"])
    for item in entries:
        image = load_image(str(path.parent / item["file"]))
        bpy.ops.mesh.primitive_plane_add(
            size=2, location=(0, float(item["depth"]) * -2, 0)
        )
        plane = bpy.context.object
        plane.name = item["name"]
        plane.scale.x = data["width"] / data["height"]
        material = bpy.data.materials.new(item["name"] + "_material")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(
            texture.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"]
        )
        material.node_tree.links.new(
            texture.outputs["Alpha"], nodes["Principled BSDF"].inputs["Alpha"]
        )
        material.surface_render_method = "DITHERED"
        plane.data.materials.append(material)


if __name__ == "__main__":
    manifest = bpy.path.abspath("//output/demo/layers/manifest.json")
    import_manifest(manifest)
