"""A mikro that answers what the action asks, and a SAM that thresholds, for offline tests.

The coordinate graph is the real mikro model classes (so `path_frame` walks it
with mikro's own `axis_map_to`): a lens' voxel space, calibrated into a physical
one, which a scene's drawing space maps onto by axis name. A path drawn on that
scene is in micrometres, as the orkestrator stores it.
"""

from __future__ import annotations

from types import SimpleNamespace as ns
from typing import Any

import numpy as np
import xarray as xr
from mikro.api.schema import (
    Axis,
    AxisInput,
    AxisType,
    CoordinateSystem,
    TransformationByDimensionTransformation,
    TransformationPart,
    TransformationScaleTransformation,
)

from path_nninteractive import geometry
from path_nninteractive.placement import _value


def system(names: str, system_id: str, unit: str | None = None) -> CoordinateSystem:
    return CoordinateSystem(
        id=system_id,
        name=system_id,
        axes=tuple(
            Axis(
                id=f"{system_id}-{n}",
                order=i,
                name=n,
                unit=None if n == "c" else unit,
                type="CHANNEL" if n == "c" else "SPACE",
            )
            for i, n in enumerate(names)
        ),
    )


IMAGE = system("czyx", "image")
PHYSICAL = system("czyx", "physical", "micrometer")
DRAWING = system("zyx", "drawing")


def graph(spacing: np.ndarray) -> list[Any]:
    calibration = TransformationScaleTransformation(
        id="calibration", kind="SCALE", version=1, input=IMAGE, output=PHYSICAL, scale=(1.0, *spacing)
    )
    scene = TransformationByDimensionTransformation(
        id="scene", kind="BY_DIMENSION", version=1, input=DRAWING, output=PHYSICAL,
        inputAxes=tuple("zyx"), outputAxes=tuple("zyx"),
        parts=(
            TransformationPart(
                kind="IDENTITY", inputAxes=tuple("zyx"), outputAxes=tuple("zyx"), scale=(), translation=(), affine=()
            ),
        ),
    )
    return [calibration, scene]


class StubLens:
    """A (c, z, y, x) lens over a volume, chunked like a zarr store; counts how often it is opened."""

    id = "lens-1"
    coordinate_system = IMAGE
    axis_names = ("c", "z", "y", "x")

    def __init__(self, volume: np.ndarray):
        self.volume = volume
        self.opened = 0

    @property
    def data(self) -> xr.DataArray:
        import dask.array as da

        self.opened += 1
        return xr.DataArray(da.from_array(self.volume[None], chunks=(1, 16, 64, 64)), dims=list("czyx"))

    def axes_of_type(self, axis_type: Any) -> tuple[str, ...]:
        return {"SPACE": ("z", "y", "x"), "CHANNEL": ("c",)}.get(_value(axis_type), ())

    def carried_axes(self, dims: list[str]) -> list[AxisInput]:
        return [AxisInput(name=d, type=AxisType.CHANNEL if d == "c" else AxisType.SPACE) for d in dims]

    def derive_translation(self, offset: list[float], *, value_relation: Any = None) -> Any:
        return ns(kind="translation", offset=list(offset), value_relation=value_relation)

    def derive(self, **kwargs: Any) -> Any:
        return ns(**{"kind": "derive", **kwargs})

    def derive_projection(self, axes: list[str], *, value_relation: Any = None) -> Any:
        return ns(kind="BY_DIMENSION", axes=list(axes), value_relation=value_relation)


class StubMikro:
    """Records what the action creates."""

    def __init__(self, edges: list[Any]):
        self.edges = edges
        self.rath = ns(query=self.query)
        self.datasets: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []

    def query(self, document: str, variables: dict[str, Any]) -> Any:
        return ns(data={"lens": {"dataset": {"name": "Synthetic AIS"}}})

    def get_transformations(self, filters: Any) -> list[Any]:
        return [edge for edge in self.edges if str(edge.input.id) == str(filters.input)]

    def get_coordinate_graph(self, coordinate_system: Any, max_depth: Any = None) -> Any:
        return ns(transformations=self.edges)

    def create_array_dataset(self, **kwargs: Any) -> Any:
        self.datasets.append(kwargs)
        return ns(lens=lambda: ns(id=f"label-{len(self.datasets)}", kwargs=kwargs))

    def create_mesh_collection(self, **kwargs: Any) -> Any:
        self.meshes.append(kwargs)
        return ns(lens=lambda: ns(id="mesh-lens"))


def drawn_path(rough_index: np.ndarray, spacing: np.ndarray) -> Any:
    """A PATH as the orkestrator stores one drawn on a scene: micrometres, columns (z, y, x)."""
    return ns(
        id="path-1",
        name="axon",
        kind="PATH",
        coordinates=[],
        coordinate_system=DRAWING,
        vectors=[tuple(row) for row in rough_index * spacing],
    )


class ThresholdInteractor(geometry.Interactor):
    """Segments what is brighter than half the prompts' intensity and connects to them.

    It answers in 3D from the prompts, as nnInteractive does, and follows the whole
    bright structure, past the corridor, so the clip is what keeps the result on the
    path. Counts how often it was given a new image.
    """

    def __init__(self) -> None:
        self.images_set = 0
        self.calls = 0
        self._key: object = None

    def segment(self, key, image, positive, negative, scribble):
        from scipy.ndimage import label

        self.calls += 1
        if key is None or key != self._key:
            self.images_set += 1
            self._key = key
        if scribble is not None:
            seeds = np.vstack([np.argwhere(stroke.crop) + stroke.low for stroke in scribble])
        else:
            seeds = np.rint(positive).astype(int)
        seeds = np.clip(seeds, 0, np.asarray(image.shape) - 1)
        level = 0.5 * (float(np.median(image)) + float(np.median(image[tuple(seeds.T)])))
        parts, _ = label(image > level, structure=np.ones((3, 3, 3)))
        touched = np.unique(parts[tuple(seeds.T)])
        return np.isin(parts, touched[touched > 0])
