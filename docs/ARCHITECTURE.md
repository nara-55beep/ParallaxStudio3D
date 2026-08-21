# Architecture

## Pipeline

1. Estimate a smooth dense depth field and a stable focal point.
2. Keep the complete image connected; automatic mode does not create binary
   person cutouts.
3. Project every pixel through one dolly/truck camera path.
4. Add separately phased sky drift with a soft horizon transition.
5. Add depth-aware particles and velocity-driven radial blur.
6. Export only MP4 and GIF during normal use.

## Why this looks different from sticker animation

All camera motion is derived from a shared envelope and focal point. Near pixels
have a larger projection scale and truck response than far pixels. Sky drift is
layered on top with a smooth spatial weight, preventing a hard horizon seam.
The move returns to its first frame for a clean GIF loop.

## Production AI backends

The core deliberately has no paid dependency. A production backend can replace
the draft analyzer with:

- semantic instance segmentation for meaningful object groups;
- monocular depth prediction for dense geometry;
- edge-aware alpha matting for hair and fine boundaries;
- generative inpainting for large hidden areas.

Those backends should emit the same scene JSON and PNG layers. The renderer and
editor adapters do not need to change.

## Host integration

The generated manifest is the compatibility boundary:

- native script adapters reconstruct editable layers where a host SDK exists;
- a rendered MP4 works in editors without a supported extension API;
- PNG layers allow manual or third-party compositor integration.
