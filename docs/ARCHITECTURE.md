# Architecture

## Automatic pipeline

1. OneFormer performs panoptic scene parsing and supplies semantic labels for
   broad scene regions.
2. SAM 2 runs in segment-everything mode and proposes detailed masks for both
   known and unlabeled objects.
3. The selector removes nested duplicate masks, balances compact objects with
   scenic regions, estimates depth from semantic priors and image position, and
   assigns camera, sky or vegetation motion.
4. Compact masks are closed, expanded by one pixel and softly feathered. Their
   union is dilated and reconstructed by the local LaMa model to create a
   background plate.
5. The renderer composites far-to-near transparent planes over the background.
   Sky uses a faster phase, vegetation adds subtle secondary drift, and camera
   response grows with depth.
6. Temporary analysis assets are deleted after normal use. Only MP4 and GIF are
   exported.

## Why two segmentation models

Semantic-only models can understand sky or terrain but may merge stylized
figures into one class. Prompt-free mask generators draw cleaner object
boundaries but do not know whether a region is sky, a person or a bush. The
combined result provides scene meaning and fine cutouts without requiring the
image to contain a fixed set of objects.

## Scene format

Each generated region records:

- a portable grayscale alpha-mask path;
- its semantic label and confidence score;
- normalized depth;
- motion kind (`camera`, `sky`, or `sway`);
- a motion-strength multiplier.

Legacy hand-authored regions without mask files continue to use constrained
GrabCut points. This keeps the existing After Effects and Blender interchange
format compatible.

## Motion and compositing

All planes share a looping dolly/truck envelope, preserving a coherent camera.
Depth changes the scale and translation response. Sky planes use twice the
camera's horizontal phase frequency. Plant planes add a restrained portion of
that secondary phase.

The generated background uses local inpainting only for compact foreground
objects. Broad sky and terrain planes retain the original scene underneath,
which provides stable coverage during their smaller movement and avoids trying
to invent an entire landscape band.

## Host integration

The diagnostic manifest is the compatibility boundary:

- native script adapters reconstruct editable planes where a host SDK exists;
- PNG alpha layers can be adjusted manually in any compositor;
- MP4 and GIF work without host-specific extensions.
