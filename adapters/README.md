# Editor adapters

- After Effects: run after_effects_import.jsx, select the generated layers
  folder, and it creates a composition with ordered transparent layers.
- Blender: open blender_import.py in the Scripting workspace. Point manifest
  at a generated layer manifest and run it.
- CapCut, Premiere Pro, DaVinci Resolve, Final Cut, mobile editors, and web
  editors can import the rendered MP4. Editors that accept PNG layers can also
  use the files and depth values in manifest.json.

The Python renderer is the common engine. An editor-specific extension should be
a thin importer/exporter around the same manifest instead of reimplementing the
depth analysis and camera mathematics.
