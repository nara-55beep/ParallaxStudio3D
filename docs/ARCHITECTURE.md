# Architecture

## Pipeline

1. Analyze the image and propose meaningful objects plus a dense depth field.
2. Refine object masks and reconstruct the pixels hidden behind them.
3. Store normalized depth, camera focus, render settings, and matte guidance in
   a JSON scene.
4. Project every pixel and object plane through one virtual camera.
5. Add depth-aware particles and velocity-driven radial blur.
6. Export MP4/GIF plus transparent layers and a universal manifest.

## Why this looks different from sticker animation

All motion is derived from a shared camera envelope and focal point. Near pixels
have a larger projection scale than far pixels. Objects therefore separate in
depth while staying spatially coherent. The move is exactly reversible because
the second half uses the same camera path in reverse.

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
