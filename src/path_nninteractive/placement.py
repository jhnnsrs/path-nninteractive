"""Where a path lands on a lens' voxels, and what the lens' voxels are.

Copied from ``testing/straighten_path_app.py`` and ``testing/path_segment_app.py``
(neither is installable), where they are tested end-to-end against the mikro
stack. The rules they encode:

- A vertex is three components: the axes of a three-axis collection in order, or
  the space axes of a wider one.
- A shape drawn on a scene is in the world's units, not pixel indices. It is carried
  onto the lens along the coordinate graph (``axis_map_to``), and kept as an affine
  so a result can go back by its inverse.
- A freestanding collection (no composable edge at all) is trusted as pixel indices
  of the lens, and the frame says so. An UNMAPPABLE edge is not a route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pint
import xarray as xr
from mikro import Mikro
from mikro.api.schema import TransformationFilter
from mikro.vocabulary import COMPOSABLE_KINDS

_UREG = pint.UnitRegistry()
_LENGTH = _UREG.micrometer.dimensionality

_PATH_KINDS = {"PATH", "LINE"}

#: How far past a face, in voxels, a vertex is still read as lying on it.
_ON_A_FACE = 1e-6


def _value(kind: Any) -> str:
    """An enum or its value, as the bare string."""
    return str(getattr(kind, "value", kind))


def _affine_diagonal(affine: Any) -> list[float] | None:
    """The per-axis factors of an axis-aligned affine, or None if it is not one."""
    if affine is None:
        return None
    rows = [list(row) for row in affine]
    if not rows or any(len(row) != len(rows) + 1 for row in rows):
        return None
    for i, row in enumerate(rows):
        if any(abs(value) > 1e-9 for j, value in enumerate(row[:-1]) if j != i):
            return None
    return [float(row[i]) for i, row in enumerate(rows)]


def _is_length(unit: str | None) -> bool:
    if not unit:
        return False
    try:
        return _UREG(unit).dimensionality == _LENGTH
    except Exception:
        return False


def spacing_um_of(mikro: Mikro, lens: Any, *, max_hops: int = 3) -> dict[str, float]:
    """Per-axis voxel size in micrometres, walked out of the lens' space.

    Follows SCALE (and axis-aligned AFFINE, which is what `calibrate` authors
    once a translation comes with the pixel size) edges out of the lens'
    coordinate system and stops at the first space whose axes carry a length
    unit. `{}` means the data is uncalibrated; a failing query raises.
    """
    system = getattr(lens, "coordinate_system", None)
    if system is None:
        return {}

    frontier: list[tuple[str, dict[str, float]]] = [(str(system.id), {})]
    seen = {str(system.id)}
    for _ in range(max_hops):
        nxt: list[tuple[str, dict[str, float]]] = []
        for system_id, carried in frontier:
            edges = mikro.get_transformations(
                filters=TransformationFilter(input=system_id)
            )
            for edge in edges:
                output = getattr(edge, "output", None)
                if output is None:
                    continue
                scale = getattr(edge, "scale", None)
                if scale is None:
                    scale = _affine_diagonal(getattr(edge, "affine", None))
                in_axes = sorted(getattr(getattr(edge, "input", None), "axes", []) or [], key=lambda a: a.order)
                out_axes = sorted(getattr(output, "axes", []) or [], key=lambda a: a.order)
                factors = dict(carried)
                if scale is not None and in_axes and len(scale) == len(in_axes):
                    for axis, factor in zip(in_axes, scale):
                        factors[axis.name] = factors.get(axis.name, 1.0) * float(factor)
                spacings: dict[str, float] = {}
                for axis in out_axes:
                    unit = str(axis.unit) if axis.unit else None
                    if not _is_length(unit):
                        continue
                    factor = factors.get(axis.name)
                    if factor is None:
                        continue
                    spacings[axis.name] = float((factor * _UREG(unit)).to("micrometer").magnitude)
                if spacings:
                    return spacings
                output_id = str(output.id)
                if output_id not in seen:
                    seen.add(output_id)
                    nxt.append((output_id, factors))
        frontier = nxt
    return {}


_DATASET_NAME_DOCUMENT = """
query DatasetNameOf($id: ID!) {
  lens(id: $id) {
    ... on ArrayLens {
      dataset {
        name
      }
    }
  }
}
"""


def dataset_name_of(mikro: Mikro, lens: Any) -> str:
    """The name of the dataset a lens looks at; not in the generated `ArrayLens` fragment either."""
    try:
        result = mikro.rath.query(_DATASET_NAME_DOCUMENT, {"id": str(lens.id)})
    except Exception:  # noqa: BLE001 - a missing name is cosmetic
        return f"lens {lens.id}"
    dataset = ((result.data or {}).get("lens") or {}).get("dataset") or {}
    return dataset.get("name") or f"lens {lens.id}"


def named_vertices(annotation: Any) -> tuple[np.ndarray, list[str]]:
    """The path's vertices as ``(n, k)`` and the axis each column is along.

    The columns of an annotation's vectors follow its coordinate system's axes.
    """
    if _value(annotation.kind) not in _PATH_KINDS:
        raise ValueError(
            f"Annotation '{annotation.name}' is a {_value(annotation.kind)}, not a PATH."
        )
    vectors = np.asarray(annotation.vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Annotation '{annotation.name}' has fewer than two vertices.")

    axes = sorted(annotation.coordinate_system.axes, key=lambda a: a.order)
    # A vertex has three components whatever it is drawn over. In a three-axis
    # space those are its axes; in a wider one (c, z, y, x) they are its space
    # axes, and the rest is said by the annotation's pinned coordinates.
    if vectors.shape[1] == len(axes):
        columns = [axis.name for axis in axes]
    else:
        columns = [axis.name for axis in axes if _value(axis.type) == "SPACE"]
    if vectors.shape[1] != len(columns):
        raise ValueError(
            f"Annotation '{annotation.name}' has {vectors.shape[1]} values per vertex, which "
            f"matches neither the axes {[axis.name for axis in axes]} it is drawn over nor "
            "the space axes among them."
        )
    return vectors, columns


@dataclass(frozen=True)
class PathFrame:
    """The affine from a path's stored vertices to the lens' voxel indices: ``index = A v + b``.

    Kept as a matrix rather than applied once, because the refined path goes back
    through its inverse. The round trip is then exact, and the refined path is
    stored in the same units and column order as the one that was drawn.
    """

    columns: tuple[str, ...]
    matrix: np.ndarray
    offset: np.ndarray
    warning: str | None = None

    def to_index(self, vertices: np.ndarray) -> np.ndarray:
        return np.asarray(vertices, dtype=np.float64) @ self.matrix.T + self.offset

    def to_drawn(self, index: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.matrix, (np.asarray(index) - self.offset).T).T


def path_frame(mikro: Mikro, annotation: Any, lens: Any, space_axes: tuple[str, ...]) -> tuple[np.ndarray, PathFrame]:
    """The path's vertices as lens voxel indices (in `space_axes` order), and the frame that took them there.

    The same placement as ``straighten_path_app.path_vertices``, including its rules:
    a freestanding collection is trusted as pixel indices and the frame says so, and
    an UNMAPPABLE edge does not count as a route.
    """
    drawn_in = annotation.coordinate_system
    target = lens.coordinate_system
    if drawn_in is None or target is None:
        raise ValueError("The path or the image has no coordinate system to place the one on the other.")
    vertices, columns = named_vertices(annotation)

    def as_they_are(warning: str | None = None) -> PathFrame:
        missing = [axis for axis in space_axes if axis not in columns]
        if missing:
            raise ValueError(
                f"Annotation '{annotation.name}' is drawn over axes {columns}, which do not name "
                f"the image's space axes {missing}."
            )
        matrix = np.zeros((3, 3))
        for row, axis in enumerate(space_axes):
            matrix[row, columns.index(axis)] = 1.0
        return PathFrame(tuple(columns), matrix, np.zeros(3), warning)

    if str(drawn_in.id) == str(target.id):
        frame = as_they_are()
    else:
        graph = drawn_in.graph(mikro=mikro)
        if not any(
            str(drawn_in.id) in (str(edge.input.id), str(edge.output.id))
            for edge in graph.transformations
            if _value(getattr(edge, "kind", "")) in COMPOSABLE_KINDS
            and getattr(edge, "input", None) is not None
            and getattr(edge, "output", None) is not None
        ):
            frame = as_they_are(
                f"The collection of '{annotation.name}' is freestanding: nothing records which "
                f"image it was drawn on. Its vertices are read as pixel indices of '{target.name}'."
            )
        else:
            try:
                onto = drawn_in.axis_map_to(target, graph=graph, mikro=mikro)
                # The affine read off the map itself: the image of the origin and of each basis vector.
                probes = np.vstack([np.zeros(3), np.eye(3)])
                mapped = onto.apply(probes, input_axes=columns, output_axes=space_axes)
            except ValueError as error:
                raise ValueError(
                    f"Annotation '{annotation.name}' cannot be placed on this image: its space "
                    f"'{drawn_in.name}' does not map onto '{target.name}' along {list(space_axes)}. "
                    f"{error} Pass an image that shares a space with the path."
                ) from error
            frame = PathFrame(tuple(columns), (mapped[1:] - mapped[0]).T, mapped[0])
    if abs(np.linalg.det(frame.matrix)) < 1e-12:
        raise ValueError(f"Annotation '{annotation.name}' maps onto the image degenerately; its path has no inverse.")
    return frame.to_index(vertices), frame


def lens_volume(
    lens: Any, space_axes: tuple[str, ...], pinned: dict[str, int]
) -> tuple[str, Callable[[], xr.DataArray]]:
    """The cache key of the 3D volume a lens shows at `pinned`, and how to open it lazily.

    The key is the lens id, never the lens object: a lens names one immutable
    selection of one immutable dataset, while the object is new on every task.
    Everything beside the space axes has to be a single position once `pinned`
    is applied, or the volume is refused rather than silently cut to its first plane.
    """
    key = f"{lens.id}|" + ",".join(f"{k}={v}" for k, v in sorted(pinned.items()))

    def open_() -> xr.DataArray:
        data = lens.data
        data = data.isel({name: value for name, value in pinned.items() if name in data.dims})
        wide = {str(d): int(data.sizes[d]) for d in data.dims if d not in space_axes and data.sizes[d] > 1}
        if wide:
            raise ValueError(
                f"The image still spans {wide} beside its space axes. Pass a lens of one "
                "channel, or draw the path at one timepoint."
            )
        return data.isel({d: 0 for d in data.dims if d not in space_axes}).transpose(*space_axes)

    return key, open_
