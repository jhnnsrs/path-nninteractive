# path-nninteractive

nnInteractive along a drawn path, in 3D. You trace a rough PATH along an axon, a
vessel or a fibre in the orkestrator 3D scene, run **nnInteractive Along Path**
from the shape's card, and get back a label crop (and, if you ask, a surface).

The model is [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive), a 3D
promptable segmenter trained on more than 120 volumetric datasets: CT, MRI, PET
and 3D microscopy.

## How it works

1. The path is snapped onto the intensity ridge.
2. The path is scribbled along wherever the structure stands out of its
   surroundings by at least `min_contrast` of its peak. With `prompt = points`,
   positive points every `prompt_every_um` are used instead.
3. nnInteractive segments the raw crop around the path from that prompt, in 3D.
   It sees the voxels exactly as stored: no smoothing, no normalisation, no 8-bit
   conversion, as its authors ask.
4. Only what lies within `radius_um` of the prompted stretch and connects to the
   path is kept, so a neighbour the model also picked up stays out. The result is
   a label crop placed on the image.

nnInteractive predicts one patch (192³, zoomed out at most 4×) around its latest
prompt. So the scribble goes in as strokes of at most 160 voxels, in path order,
each its own prediction. A single scribble along a 500-voxel tube was found only
in its middle 200 voxels.

Measured on an RTX 4070, at 0.3 × 0.1 × 0.1 µm voxels, using the test volumes:

| volume | prompt | voxels | dice | time |
| --- | --- | --- | --- | --- |
| synthetic AIS (40×170×170) | scribble | raw | 0.65 (0.83 with `min_contrast` 0.5) | 0.25 s |
| synthetic AIS | points (7) | raw | 0.61 | 1.7 s |
| synthetic AIS | scribble | resampled to cubic | 0.54 | 0.35 s |
| tube, 500 voxels long | scribble (strokes) | raw | 0.87 | 1.8 s |
| tube, 500 voxels long | points (26) | raw | 0.82 | 6.3 s |
| tube, 1200 voxels long | scribble (strokes) | raw | 0.87 | 3.8 s |
| tube, 1200 voxels long | points (61) | raw | 0.82 | 15 s |

Peak GPU memory was 3.9 GiB throughout.

Both prompts find essentially all of the AIS. What lowers the dice at the default
`min_contrast` is that the model rightly follows the dimmer axon past the AIS.
`min_contrast` is the knob for where to stop.

## Tuning by re-running

The agent keeps the voxels it read, the smoothed crop, and the image it last gave
the model, preprocessed on the GPU. A re-run on the same path with other prompts
only resets nnInteractive's interactions.

## Flavour

| flavour | hardware | torch |
| --- | --- | --- |
| `cuda` | NVIDIA GPU, 8 GB of VRAM (about 4 GB measured on a small crop); host driver ≥ 580 | PyPI's CUDA 13 build |

There is no CPU flavour: nnInteractive needs a GPU to be usable.

The image bakes the weights into `/opt/nninteractive` and sets `HF_HUB_OFFLINE=1`,
so a pod needs no network.

## Licence of the model weights

The **nnInteractive weights** are © Division of Medical Image Computing, German
Cancer Research Center (DKFZ). They are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
- non-commercial use only;
- with attribution;
- derivatives under the same licence.

The image ships their `LICENSE` next to them, in
`/opt/nninteractive/models/nnInteractive_v1.0/`. The nnInteractive **code** is
Apache-2.0.

When you use results from this app, cite:

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T.,
> Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D
> Promptable Segmentation. https://arxiv.org/abs/2503.08373

## Development

```bash
uv sync
nninteractive-download-model nnInteractive_v1.0   # once, into ~/.nninteractive
uv run pytest            # offline: geometry, the action against a stub mikro, and real
                         # nnInteractive if its weights are downloaded and a GPU is there
uv run pytest -m hub     # against a konstruktor hub (Docker, about a minute to start)
arkitekt run dev
```

Part of the code is copied from `path-sam`, and from the testing scripts before it:
- placement (`placement.py`);
- the voxel cache (`cache.py`);
- the ridge snap, profile and corridor (`geometry.py`).
