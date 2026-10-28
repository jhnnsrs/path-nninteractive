"""nnInteractive along a drawn PATH, in 3D: a promptable 3D segmenter guided by a rough tracing.

You draw a rough PATH through a volume in the orkestrator 3D scene, along an axon,
a vessel or a fibre. Then you run **nnInteractive Along Path** from the shape's
card. The action:

- snaps the path onto the intensity ridge;
- draws a scribble along it, where the structure stands out of its surroundings
  (or places positive points every `prompt_every_um`);
- has nnInteractive (https://github.com/MIC-DKFZ/nnInteractive) segment the raw
  volume from that prompt, in 3D;
- keeps what lies within `radius_um` of the path and connects to it, and returns
  it as a label crop, which the scene adds as a LABEL layer. A surface comes too,
  with `with_mesh`.

nnInteractive is 3D through and through, so unlike Segment Anything nothing is
cut into planes: the crop around the path goes in as it is. The geometry is in
``src/path_nninteractive/geometry.py``, the model in ``src/path_nninteractive/nni.py``.

**Re-running is the adjustment loop.** The agent keeps the voxels it read (bricks,
per lens), the smoothed crop, and the image it last gave the model, preprocessed
on the GPU. A re-run on the same crop (other prompts, another `min_contrast`,
negatives) only resets the interactions.

The model weights are CC BY-NC-SA 4.0: non-commercial use, share-alike. See the
README for the licence and the citation.

Placement is the one ``testing/path_segment_app.py`` proved: vertices go from the
path's drawing space onto the lens' voxels along the coordinate graph. The label
crop is a CATEGORIZED translation of the lens with no pyramid.

Run:  arkitekt run dev
"""

from __future__ import annotations

import contextlib
import io
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Optional, Tuple

import numpy as np
import xarray as xr
from arkitekt import App, Description, Task, run
from mikro import Mikro, mikro_service
from mikro.api.schema import (
    AnnotationDerivedFromInput,
    AxisType,
    MeshLens,
    ValueRelation,
)
from mikro.arkitekt.specs import LabelMask, PathAnnotation, SingleChannelVolume
from mikro.arkitekt.widgets import reachable_from

from path_nninteractive import geometry
from path_nninteractive.cache import VoxelCache
from path_nninteractive.nni import MODEL_ID, NnInteractive, load_session
from path_nninteractive.placement import _ON_A_FACE, dataset_name_of, lens_volume, path_frame, spacing_um_of

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "1.0.0"


# The app is a declaration: who it is, what it offers, and the services it uses.
# `arkitekt run` finds it in this module (as `app`) and runs it; so does
# `python app.py`, below.
app = App(
    "path-nninteractive",
    __version__,
    description="nnInteractive 3D segmentation along a drawn path",
    author="jhnnsrs",
    scopes=["read", "write"],
    services=[mikro_service],
)


@app.context
@dataclass
class NniCache:
    """What every task of this agent shares, built at startup. The model's image is in `nni`'s session."""

    voxels: VoxelCache = field(
        default_factory=lambda: VoxelCache(int(float(os.environ.get("PATH_NNINTERACTIVE_CACHE_GB", "4")) * 1024**3))
    )


@app.startup
def start_cache() -> NniCache:
    """A fresh voxel cache for each start of the agent; an action asks for it as `cache: NniCache`."""
    return NniCache()


COLLECTIONS = ["segment"]


# A plain Enum, not (str, Enum): a str member as a default turns the port into free text.
class Prompt(Enum):
    """How the path is shown to nnInteractive."""

    SCRIBBLE = "scribble"
    POINTS = "points"


@dataclass
class Outputs:
    label: Any
    mesh: Any = None


def segment_path(
    mikro: Mikro,
    task: Task,
    cache: NniCache,
    path: Any,
    lens: Any,
    params: geometry.NniParams,
    interactor: geometry.Interactor,
    *,
    with_mesh: bool,
) -> Outputs:
    """Segment along `path` in what `lens` shows, and register what was found."""
    started = time.perf_counter()
    voxels = cache.voxels
    space_axes = tuple(lens.axes_of_type(AxisType.SPACE))
    if len(space_axes) != 3:
        raise ValueError(f"Segmenting along a path needs three space axes, the image has {list(space_axes)}.")
    vertices, frame = path_frame(mikro, path, lens, space_axes)
    if frame.warning:
        task.log(frame.warning)

    # A path pinned to a timepoint is segmented at that timepoint. Everything else
    # beside the space axes has to be a single position (a pinned channel, say).
    lens_axes = tuple(lens.axis_names)
    pinned = {
        str(c.name): int(c.value)
        for c in path.coordinates or ()
        if c.name in lens_axes and c.name not in space_axes and c.name not in lens.axes_of_type(AxisType.CHANNEL)
    }
    key, open_ = lens_volume(lens, space_axes, pinned)

    data = voxels.handle(key, open_)
    shape = np.array([data.sizes[axis] for axis in space_axes])
    if (vertices < -0.5 - _ON_A_FACE).any() or (vertices > shape - 0.5 + _ON_A_FACE).any():
        raise ValueError(
            f"Annotation '{path.name}' leaves the image: its vertices span "
            f"{vertices.min(axis=0).round(1).tolist()} to {vertices.max(axis=0).round(1).tolist()} "
            f"over {list(space_axes)}, the image is {shape.tolist()}."
        )

    spacing_um = voxels.remember(str(lens.id), "spacing", lambda: spacing_um_of(mikro, lens))
    calibrated = all(axis in spacing_um for axis in space_axes)
    spacing = np.array([spacing_um[axis] for axis in space_axes]) if calibrated else np.ones(3)
    if not calibrated:
        task.log("The image is not calibrated: every length below is in voxels, not micrometres.")
    unit = "µm" if calibrated else "px"

    low, high = geometry.required_box(vertices, spacing, shape, params)
    fetched_before = voxels.fetched
    task.progress(10, f"Reading {(high - low).tolist()} voxels around '{path.name}'")
    crop = voxels.read(key, open_, low, high)
    warming = voxels.warm(key, data)
    read_s = time.perf_counter() - started

    smoothed = voxels.field(
        (key, tuple(low), tuple(high), params.sigma_um, tuple(spacing)),
        lambda: geometry.smooth(crop, spacing, params.sigma_um),
    )
    task.progress(30, "Segmenting with nnInteractive")
    # nnInteractive prints its progress; the task log says what matters instead.
    with contextlib.redirect_stdout(io.StringIO()):
        result = geometry.segment(
            crop, vertices - low, spacing, params, interactor, field=smoothed, key=(key, tuple(low), tuple(high))
        )
    if result.empty:
        raise ValueError(
            f"Nothing along '{path.name}' stands out of its surroundings, or nnInteractive found "
            f"nothing at the prompts: no stretch of the path has {params.min_contrast:g} of the peak "
            "contrast. Lower min_contrast, widen the radius, or check that the path runs along the stain."
        )
    summary = result.summary
    prompted = "a scribble" if params.prompt == "scribble" else f"{len(result.prompts.positive)} points"
    task.log(
        f"{summary['volume_um3']:.2f} {unit}³ over {summary['segmented_length_um']:.1f} of "
        f"{summary['path_length_um']:.1f} {unit} (prompted with {prompted}), "
        f"mean diameter {summary['mean_diameter_um']:.2f} {unit}"
    )

    name = path.name or f"path {path.id}"
    source = dataset_name_of(mikro, lens)
    task.progress(70, "Uploading the label crop")
    # The crop keeps every axis of the lens, the non-space ones as single positions, so
    # its edge back to the lens is a plain translation with one entry per lens axis.
    origin_space = dict(zip(space_axes, (low + result.origin).astype(float)))
    origin = [origin_space.get(axis, float(pinned.get(axis, 0))) for axis in lens_axes]
    label_array = (
        xr.DataArray(result.mask.astype(np.uint8), dims=list(space_axes))
        .expand_dims([axis for axis in lens_axes if axis not in space_axes])
        .transpose(*lens_axes)
    )
    label_ds = mikro.create_array_dataset(
        data=label_array,
        scales=[],  # a mean of label ids is another id, and a subsample pyramid drifts
        name=f"{source} · nnInteractive along {name} (r {params.radius_um:g} {unit})",
        axes=lens.carried_axes(list(lens_axes)),
        derived_from=[
            lens.derive_translation(origin, value_relation=ValueRelation.CATEGORIZED),
            AnnotationDerivedFromInput(annotation=path.id, value_relation=ValueRelation.TRANSFORMED),
        ],
    )
    outputs = Outputs(label=label_ds.lens())

    if with_mesh:
        task.progress(85, "Meshing")
        outputs.mesh = upload_mesh(mikro, lens, space_axes, result, low, name, source)

    total = time.perf_counter() - started
    stages = ", ".join(f"{k} {v * 1000:.0f}" for k, v in result.timings.items())
    task.log(
        f"{total:.2f} s: read {read_s:.2f} s ({voxels.fetched - fetched_before} bricks fetched, "
        f"{voxels.bytes / 1024**2:.0f} MiB cached{', warming the rest' if warming else ''}); segment ms: {stages}"
    )
    task.progress(100, f"Segmented '{name}'")
    return outputs


def upload_mesh(
    mikro: Mikro,
    lens: Any,
    space_axes: tuple[str, ...],
    result: geometry.Segmentation,
    low: np.ndarray,
    name: str,
    source: str,
) -> MeshLens:
    """The surface, in the lens' voxels with xyz components, as a one-object mesh collection."""
    from fabriks import build_collection

    verts, faces = geometry.surface(result.mask, low + result.origin)
    # Components reversed, faces left alone: the reversal is a reflection and turns the
    # marching-cubes faces outward by itself. The octree wants non-negative cells, and a
    # cap on a volume face sits at -0.5, so it is clipped rather than moved.
    xyz_axes = list(reversed(space_axes))
    verts_xyz = np.clip(verts[:, ::-1], 0.0, None)
    # `axes` names the vertex slots. Left out, the viewer guesses them as the
    # collection's last three axes reversed, which transposes this (x, y, z) one.
    built = build_collection({1: (verts_xyz, faces)}, levels=2, axes=tuple(xyz_axes))
    collection = mikro.create_mesh_collection(
        name=f"{source} · nnInteractive surface along {name}",
        store=built,
        axes=xyz_axes,
        # A projection: BY_DIMENSION naming x, y and z, by axis name -- the lens may
        # carry axes the surface has not (a channel), which a MAP_AXIS (a pure
        # permutation) is refused for, and an axis no one names is not claimed.
        derived_from=[lens.derive_projection(xyz_axes, value_relation=ValueRelation.TRANSFORMED)],
    )
    return collection.lens()


# --------------------------------------------------------------------------- #
# The actions
# --------------------------------------------------------------------------- #


@app.action(name="nnInteractive Along Path", collections=COLLECTIONS, concurrency="serial")
def nninteractive_along_path(
    path: Annotated[PathAnnotation, Description("The PATH to segment along: a rough tracing is enough")],
    lens: Annotated[SingleChannelVolume, reachable_from("path"), Description("The channel the structure is stained in")],
    radius_um: Annotated[float, Description("How far from the path the structure may reach")] = 1.5,
    prompt: Annotated[Prompt, Description("A scribble along the path (faster), or points along it")] = Prompt.SCRIBBLE,
    prompt_every_um: Annotated[float, Description("Distance between points, when prompting with points")] = 2.0,
    n_negative: Annotated[int, Description("Background points around each prompted position (slower)")] = 0,
    min_contrast: Annotated[
        float, Description("Only where the structure has this fraction of its peak contrast is prompted")
    ] = 0.15,
    sigma_um: Annotated[float, Description("Smoothing for the ridge snap (the model sees the raw voxels)")] = 0.3,
    refine_path: Annotated[bool, Description("Snap the path onto the intensity ridge first")] = True,
    isotropic: Annotated[bool, Description("Resample to cubic voxels for the model")] = False,
    with_mesh: Annotated[bool, Description("Also upload a surface (slower)")] = False,
    *,
    mikro: Mikro,
    task: Task,
    cache: NniCache,
) -> Tuple[
    Annotated[LabelMask, Description("The segmentation, as a label crop")],
    Annotated[Optional[MeshLens], Description("The surface, if asked for")],
]:
    """nnInteractive Along Path

    Prompts nnInteractive with a scribble along a rough path and keeps the 3D
    structure it finds within the radius of it. Fast to re-run on the same
    path: the image stays on the GPU
    """
    params = geometry.NniParams(
        radius_um=radius_um,
        sigma_um=sigma_um,
        refine=refine_path,
        prompt_every_um=prompt_every_um,
        min_contrast=min_contrast,
        prompt=prompt.value,
        n_negative=n_negative,
        isotropic=isotropic,
    )
    out = segment_path(mikro, task, cache, path, lens, params, NnInteractive(), with_mesh=with_mesh)
    return out.label, out.mesh


@app.action(name="Warm nnInteractive", collections=COLLECTIONS)
def warm_nninteractive(*, task: Task) -> str:
    """Warm nnInteractive

    Loads the model onto the GPU before the first segmentation, so that one is
    as fast as the rest
    """
    task.progress(10, f"Loading {MODEL_ID}")
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        held = load_session()
    return f"{MODEL_ID} is loaded on {held.session.device} in {time.perf_counter() - started:.1f} s"


if __name__ == "__main__":
    run(app)
