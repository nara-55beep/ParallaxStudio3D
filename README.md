# ParallaxStudio3D

ParallaxStudio3D turns one still image into a looping 2.5D parallax shot. The
automatic path creates actual transparent scene planes instead of applying one
zoom or bending the complete image as a single sheet.

It combines two free local models:

- OneFormer identifies scene meaning such as people, sky, vegetation, terrain,
  buildings and water.
- SAM 2 cuts both recognized and unlabeled objects with detailed boundaries.
- LaMa reconstructs scenery hidden behind the selected foreground planes.

The detected planes receive depth estimates and independent motion. Compact
foreground objects are removed from a reconstructed background plate so they do
not leave a second copy behind when the camera moves.

![ParallaxStudio3D demo](output/demo/parallax.gif)

## Use it on Windows

1. Double-click `start_parallax_studio.bat`.
2. Choose any JPG, PNG or WebP under **Single image**.
3. Leave **Scene (optional)** blank.
4. Choose an output folder.
5. Click **AI Cut + Create MP4 + GIF**.

The first launch installs the free AI dependencies if needed. The first render
downloads about 600 MB of model files once, then reuses the local cache. No API
key, subscription or cloud upload is required.

The selected output folder receives exactly:

- `parallax.mp4`
- `parallax.gif`

Masks, depth maps and background plates are made in the Windows temporary
directory and removed automatically after a normal render.

## What moves independently

- People, animals, props and other compact objects become separate alpha planes.
- Sky and clouds use a separately phased, faster horizontal drift.
- Trees, grass and plants receive a small secondary sway.
- Distant terrain moves less than near subjects and foreground objects.
- Every layer follows the same reversible camera path, so the GIF loops cleanly.

The edge cleanup closes tiny mask gaps, preserves narrow limbs, expands compact
objects by one pixel and adds a sub-pixel feather. This avoids the chopped hands
and hard sticker edges produced by the old GrabCut proposal system.

## Terminal

Install everything:

    python -m pip install -e ".[ai,dev]"

Render any image:

    python -m parallax3d image "path\to\image.jpg" --output output\render

Create a persistent editable layer project:

    python -m parallax3d auto "path\to\image.jpg" --project output\project

Render the included demo:

    render_demo.bat

## Advanced scenes and editors

The optional Scene field opens a hand-authored or previously generated JSON
scene. `parallax3d inspect` exports its transparent PNG planes and manifest.

- Any editor, including CapCut, can import the finished MP4.
- After Effects can use `adapters/after_effects_import.jsx` with inspected layers.
- Blender can use `adapters/blender_import.py` with the same manifest.

## Models, cost and privacy

Automatic segmentation uses
[OneFormer ADE20K Swin Tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny)
and [SAM 2.1 Hiera Small](https://huggingface.co/facebook/sam2.1-hiera-small).
Background reconstruction uses the
[official Apache-2.0 LaMa architecture](https://github.com/advimman/lama) through
an Apache-2.0 TorchScript package. The model repositories identify their code
or weights as MIT or Apache-2.0 licensed. Inference happens on the user's
computer and uses no paid service.

No single-image system can reconstruct truly hidden detail with certainty.
ParallaxStudio3D therefore uses restrained camera travel and background
inpainting. Difficult transparent objects, motion blur or extremely low
resolution sources can still need manual mask correction in an editor.

## Development

    python -m pip install -e ".[ai,dev]"
    python -m pytest -q

The application's own source code is MIT licensed. See `docs/ARCHITECTURE.md`
for the pipeline design.
