"""The geometry around nnInteractive: prompts from a drawn path, and the result clipped to it.

nnInteractive is a 3D promptable segmenter: it takes the volume as it is and a
prompt in it, and answers with a 3D mask. A drawn path becomes that prompt:

    rough path   ->  arclength-uniform centreline, snapped onto the intensity ridge
    prompts      ->  one scribble along the centreline where the structure stands
                     out of its surroundings (or a positive point every
                     `prompt_every_um` along it)
    voxels       ->  the raw crop, in its own voxels (or resampled to cubic ones)
    nnInteractive->  (nni.py) one prediction per prompt, each refining the last
    clip         ->  only what lies within `radius_um` of the path and connects to
                     it is kept, so a neighbour the model followed is dropped

The model is given the raw intensities, as its authors insist: no smoothing, no
normalisation, no conversion to 8 bit. The smoothed field is for the ridge snap
and the contrast profile only.

All geometry is in the volume's own axis order and, given a voxel size, in
micrometres. With ``spacing = (1, 1, 1)`` every "_um" reads as voxels.

The centreline, ridge snap, profile, corridor, finish and the synthetic AIS come
from ``path-sam`` (and before it ``testing/path_segment_core.py``), copied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    gaussian_filter1d,
    label,
    map_coordinates,
    zoom,
)
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

Prompt = Literal["points", "scribble"]


@dataclass(frozen=True)
class NniParams:
    """Every knob of one run. Physical units when the volume is calibrated, voxels otherwise."""

    #: How far from the centreline a voxel may be and still belong to the structure.
    radius_um: float = 1.5
    #: Gaussian smoothing for the ridge snap and the contrast profile (never for the model).
    sigma_um: float = 0.3
    #: Snap the centreline onto the intensity ridge before prompting.
    refine: bool = True
    #: How far a centreline sample may move in one ridge-snap iteration.
    search_um: float = 1.0
    #: Ridge-snap iterations.
    iterations: int = 2
    #: Width of the background ring outside `radius_um` the contrast is measured against.
    shell_um: float = 1.0
    #: Smoothing of shifts and profiles along the path.
    along_um: float = 1.0
    #: Distance between positive points.
    prompt_every_um: float = 2.0
    #: Only where the structure stands out by this fraction of its peak contrast is prompted.
    min_contrast: float = 0.15
    #: One scribble along the path (in strokes, see `strokes`), or positive points
    #: along it, one prediction each. Measured at 0.3 x 0.1 x 0.1 µm voxels on an RTX
    #: 4070: on the synthetic AIS the scribble has the better precision (0.72 vs 0.68 of
    #: what is found lies on the axon, both find all of it) in 0.25 s against 1.7 s; on
    #: 500- and 1200-voxel tubes, dice 0.87 vs 0.82, in 1.8 / 3.8 s against 6 / 15 s.
    #: The model was trained on scribbles drawn in one slice; a 3D one is accepted, and works.
    prompt: Prompt = "scribble"
    #: Negative points per prompted position, on the background ring.
    n_negative: int = 0
    #: Resample the crop to cubic voxels of the finest spacing for the model. It works
    #: in voxel space, but on the synthetic AIS the raw 3:1 voxels did better
    #: (dice 0.65 vs 0.54 with the scribble), so this is off unless a stack asks for it.
    isotropic: bool = False

    @property
    def margin_um(self) -> float:
        """How far past the path's vertices anything is read."""
        return (
            self.radius_um
            + self.shell_um
            + (self.iterations * self.search_um if self.refine else 0.0)
            + 3.0 * self.sigma_um
        )


@dataclass
class Prompts:
    """What the model is told, in crop-local index coordinates."""

    positive: np.ndarray  # (p, 3)
    negative: np.ndarray  # (q, 3)
    scribble: np.ndarray | None  # (k, 3) the centreline it is drawn along, or None for points


@dataclass
class Segmentation:
    """What one run found. Index coordinates are crop-local (the crop the volume was)."""

    mask: np.ndarray  # bool, the bounding box of the segmentation
    origin: np.ndarray  # (3,) int: crop-local index of mask[0, 0, 0]
    centreline: np.ndarray  # (m, 3) crop-local index coordinates
    arclength: np.ndarray  # (m,)
    prompts: Prompts
    summary: dict[str, float]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.mask.any()


# --------------------------------------------------------------------------- #
# Centreline, fields and the the stages before the model (copied)
# --------------------------------------------------------------------------- #


def resample_path(
    vertices: np.ndarray, spacing: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit tangents along the path, uniform in arc length.

    `vertices` is ``(n, 3)`` in index units, `spacing` the physical size of one
    index step per axis and `step` the physical distance between samples. A
    cubic spline is fitted through the vertices in *physical* space, so a step
    along a coarse axis counts for the distance it really covers. Both results
    are physical, ``(m, 3)``.
    """
    points = np.asarray(vertices, dtype=np.float64) * spacing
    # A repeated vertex has no direction and makes the spline fit singular.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    points = points[keep]
    if len(points) < 2:
        raise ValueError("A path needs at least two distinct vertices to be straightened.")

    chord = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    dense = int(max(2000, 8 * chord / step))
    u_dense = np.linspace(0.0, 1.0, dense)
    try:
        tck, _ = splprep(list(points.T), s=0, k=min(3, len(points) - 1))
        curve = np.column_stack(splev(u_dense, tck))
    except Exception:
        # The polyline itself, parametrised by chord length.
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        curve = np.column_stack(
            [np.interp(u_dense * along[-1], along, points[:, i]) for i in range(3)]
        )

    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    n_samples = max(2, int(round(length[-1] / step)) + 1)
    targets = np.linspace(0.0, length[-1], n_samples)
    centres = np.column_stack([np.interp(targets, length, curve[:, i]) for i in range(3)])

    tangents = np.gradient(curve, axis=0)
    tangents = np.column_stack([np.interp(targets, length, tangents[:, i]) for i in range(3)])
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    return centres, tangents


def transport_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A normal and binormal per tangent, carried along the path without twist.

    The first normal is arbitrary; each next one is the previous projected onto
    the new perpendicular plane. Rebuilding the frame per point from a fixed
    reference instead makes the cross-section jump wherever the tangent passes
    that reference.
    """
    first = tangents[0]
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(first)))] = 1.0
    normal = np.cross(first, reference)
    normal /= np.linalg.norm(normal)

    normals = np.empty_like(tangents)
    for i, tangent in enumerate(tangents):
        normal = normal - np.dot(normal, tangent) * tangent
        normal /= np.linalg.norm(normal)
        normals[i] = normal
    return normals, np.cross(tangents, normals)


class _Clock:
    """Per-stage wall time, so a slow run says where it was slow."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[stage] = self.timings.get(stage, 0.0) + now - self._last
        self._last = now


def smooth(volume: np.ndarray, spacing: np.ndarray, sigma_um: float) -> np.ndarray:
    """The volume as float32, blurred physically round (anisotropic voxels taken into account)."""
    values = np.asarray(volume, dtype=np.float32)
    if sigma_um <= 0:
        return values
    return gaussian_filter(values, sigma=sigma_um / np.asarray(spacing), mode="nearest", truncate=3.0)


def required_box(
    vertices: np.ndarray, spacing: np.ndarray, shape: np.ndarray, params: NniParams
) -> tuple[np.ndarray, np.ndarray]:
    """The ``[low, high)`` index box a run on these vertices reads, clipped to the volume."""
    margin = np.ceil(params.margin_um / np.asarray(spacing)).astype(int) + 2
    low = np.floor(vertices.min(axis=0)).astype(int) - margin
    high = np.ceil(vertices.max(axis=0)).astype(int) + margin + 1
    return np.clip(low, 0, shape), np.clip(high, 0, shape)


def disk_offsets(inner: float, outer: float, step: float) -> np.ndarray:
    """``(k, 2)`` offsets ``(v, u)`` on a grid of `step` in the annulus ``inner <= r <= outer``."""
    n = int(np.ceil(outer / step))
    v, u = np.meshgrid(np.arange(-n, n + 1) * step, np.arange(-n, n + 1) * step, indexing="ij")
    r = np.hypot(v, u)
    keep = (r >= inner) & (r <= outer + 1e-9)
    offsets = np.column_stack([v[keep], u[keep]])
    return offsets if len(offsets) else np.zeros((1, 2))


def cross_sections(field: np.ndarray, line: Centreline, offsets: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """``(m, k)``: the field at each offset of each sample's perpendicular plane, linearly."""
    points = (
        line.points[:, None, :]
        + offsets[None, :, 0, None] * line.binormals[:, None, :]
        + offsets[None, :, 1, None] * line.normals[:, None, :]
    )
    indices = (points / spacing).reshape(-1, 3).T
    return map_coordinates(field, indices, order=1, mode="nearest").reshape(points.shape[:2])


@dataclass
class Centreline:
    """Samples along a path, uniform in arclength, physical, in crop-local coordinates."""

    points: np.ndarray  # (m, 3)
    tangents: np.ndarray  # (m, 3)
    normals: np.ndarray  # (m, 3)
    binormals: np.ndarray  # (m, 3)
    step: float

    @property
    def arclength(self) -> np.ndarray:
        segments = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(segments)])


def centreline_of(vertices: np.ndarray, spacing: np.ndarray, step: float) -> Centreline:
    """A centreline through `vertices` (index coordinates), sampled every `step`."""
    points, tangents = resample_path(vertices, spacing, step)
    normals, binormals = transport_frames(tangents)
    return Centreline(points, tangents, normals, binormals, step)


def refine_centreline(
    field: np.ndarray, line: Centreline, spacing: np.ndarray, params: NniParams
) -> Centreline:
    """Move each sample to the intensity centroid of its cross-section, smoothly.

    Only what stands out of its own cross-section pulls: the median is subtracted
    first. Without that, a sample in a dim stretch is dragged towards whatever
    happens to be brightest within `search_um`. Shifts are clamped, then smoothed
    along the path, so one noisy plane cannot kink the line.
    """
    offsets = disk_offsets(0.0, params.search_um, line.step)
    for _ in range(max(0, params.iterations)):
        values = cross_sections(field, line, offsets, spacing)
        weights = np.clip(values - np.median(values, axis=1, keepdims=True), 0.0, None)
        total = weights.sum(axis=1, keepdims=True)
        shift_vu = np.divide(weights @ offsets, total, out=np.zeros((len(values), 2)), where=total > 0)
        shift = shift_vu[:, :1] * line.binormals + shift_vu[:, 1:] * line.normals
        norm = np.linalg.norm(shift, axis=1, keepdims=True)
        shift *= np.minimum(1.0, params.search_um / np.maximum(norm, 1e-12))
        shift = gaussian_filter1d(shift, sigma=params.along_um / line.step, axis=0, mode="nearest")
        line = centreline_of((line.points + shift) / spacing, spacing, line.step)
    return line


def profile(
    field: np.ndarray, line: Centreline, spacing: np.ndarray, params: NniParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(fg, bg)`` per sample: the core mean and the median of a ring around the corridor."""
    core = max(line.step, 0.5 * params.radius_um)
    fg = cross_sections(field, line, disk_offsets(0.0, core, line.step), spacing).mean(axis=1)
    ring = disk_offsets(params.radius_um, params.radius_um + params.shell_um, line.step)
    bg = np.median(cross_sections(field, line, ring, spacing), axis=1)
    sigma = params.along_um / line.step
    return (
        gaussian_filter1d(fg, sigma, mode="nearest"),
        gaussian_filter1d(bg, sigma, mode="nearest"),
    )


def corridor(
    shape: tuple[int, ...], line: Centreline, spacing: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(coords, distance, nearest)`` of every voxel within `radius` of the centreline.

    `coords` is ``(K, 3)`` in C order (so its raveled indices are sorted), `nearest`
    the index of the closest sample. Candidates are painted as boxes around each
    sample and then measured exactly. Memory scales with the corridor, not the
    crop, which matters for a long diagonal path whose bounding box is mostly empty.
    """
    candidate = np.zeros(shape, dtype=bool)
    half = np.ceil(radius / spacing).astype(int) + 1
    centres = np.rint(line.points / spacing).astype(int)
    for centre in centres[:: max(1, int(half.min() // 2) or 1)]:
        low = np.clip(centre - half, 0, shape)
        high = np.clip(centre + half + 1, 0, shape)
        candidate[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = True
    # The stride above can skip the last sample's box.
    low = np.clip(centres[-1] - half, 0, shape)
    high = np.clip(centres[-1] + half + 1, 0, shape)
    candidate[low[0] : high[0], low[1] : high[1], low[2] : high[2]] = True

    coords = np.argwhere(candidate)
    distance, nearest = cKDTree(line.points).query(coords * spacing, distance_upper_bound=radius)
    inside = np.isfinite(distance)
    return coords[inside], distance[inside], nearest[inside]


def _finish(coords: np.ndarray, reached: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Close one voxel and fill holes, inside the corridor, on the tight box of the result."""
    if not reached.any():
        return np.zeros((0, 0, 0), dtype=bool), np.zeros(3, dtype=int)
    hits = coords[reached]
    low = np.maximum(hits.min(axis=0) - 1, 0)
    high = np.minimum(hits.max(axis=0) + 2, shape)
    inside = coords[np.all((coords >= low) & (coords < high), axis=1)] - low
    allowed = np.zeros(high - low, dtype=bool)
    allowed[tuple(inside.T)] = True
    mask = np.zeros(high - low, dtype=bool)
    mask[tuple((hits - low).T)] = True
    mask = binary_fill_holes(binary_closing(mask, structure=np.ones((3, 3, 3)), border_value=0) | mask) & allowed
    # Tight again: the margin above was only for the closing.
    found = np.argwhere(mask)
    tight_low, tight_high = found.min(axis=0), found.max(axis=0) + 1
    mask = mask[tuple(slice(int(a), int(b)) for a, b in zip(tight_low, tight_high))]
    return mask, low + tight_low


def surface(mask: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes over `mask`: ``(verts, faces)``, verts in index coordinates of the crop.

    The pad is what lets the surface close where the mask touches its own box. It
    moves the origin by one voxel, so the offset is ``origin - 1``. The vertices are
    in array axis order. Reverse the components (``verts[:, ::-1]``) for xyz, and do
    not also flip the faces: the reversal is a reflection and already turns them
    outward. ``step_size`` stays 1, because coarser steps shift the surface.
    """
    padded = np.pad(mask, 1).astype(np.uint8)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=1)
    return verts + (np.asarray(origin, dtype=np.float64) - 1.0), faces


# --------------------------------------------------------------------------- #
# Prompts, resampling and the clip
# --------------------------------------------------------------------------- #


def prompted_samples(contrast: np.ndarray, step: float, params: NniParams) -> np.ndarray:
    """The centreline samples to prompt: about every `prompt_every_um` where the structure stands out.

    The first and last such samples are always among them, so the model is told
    where the structure ends on both sides, not only where it starts.
    """
    peak = float(contrast.max()) if len(contrast) else 0.0
    if peak <= 0:
        return np.zeros(0, dtype=int)
    valid = np.flatnonzero(contrast >= params.min_contrast * peak)
    every = max(1, int(round(params.prompt_every_um / step)))
    picked = [int(valid[0])]
    for index in valid[1:]:
        if index - picked[-1] >= every:
            picked.append(int(index))
    if picked[-1] != valid[-1]:
        picked.append(int(valid[-1]))
    return np.asarray(picked, dtype=int)


def prompts_for(
    line: Centreline, samples: np.ndarray, spacing: np.ndarray, shape: tuple[int, ...], params: NniParams
) -> Prompts:
    """The positive points (or scribble) at `samples`, and negatives on the ring around them."""
    positive = line.points[samples] / spacing
    negatives = []
    ring = params.radius_um + 0.5 * params.shell_um
    for i in samples:
        for k in range(max(0, params.n_negative)):
            angle = 2.0 * np.pi * (k + 0.5) / params.n_negative
            offset = ring * (np.sin(angle) * line.binormals[i] + np.cos(angle) * line.normals[i])
            negatives.append((line.points[i] + offset) / spacing)
    negative = np.asarray(negatives, dtype=float).reshape(-1, 3)
    limit = np.asarray(shape) - 1
    negative = negative[np.all((negative >= 0) & (negative <= limit), axis=1)]

    # The centreline between the first and last prompted sample. It is drawn on the
    # model's grid (`strokes`), one voxel thick: nnInteractive thickens it itself.
    scribble = line.points[samples[0] : samples[-1] + 1] / spacing if params.prompt == "scribble" else None
    return Prompts(positive=positive, negative=negative, scribble=scribble)


@dataclass(frozen=True)
class Grid:
    """How the crop is resampled for the model: index ``i`` of the crop is ``i * scale`` of the model's."""

    scale: np.ndarray  # (3,)
    shape: tuple[int, ...]  # the model's

    @classmethod
    def of(cls, shape: tuple[int, ...], spacing: np.ndarray, isotropic: bool) -> "Grid":
        factor = np.asarray(spacing, dtype=float) / float(np.min(spacing))
        if not isotropic or np.allclose(factor, 1.0, atol=0.01):
            return cls(scale=np.ones(3), shape=tuple(shape))
        target = tuple(int(max(1, round(n * f))) for n, f in zip(shape, factor))
        # Corner sample to corner sample, as scipy's zoom maps them (grid_mode=False).
        scale = np.array([(t - 1) / (n - 1) if n > 1 else 1.0 for n, t in zip(shape, target)])
        return cls(scale=scale, shape=target)

    @property
    def identity(self) -> bool:
        return bool(np.all(self.scale == 1.0))

    def image(self, volume: np.ndarray) -> np.ndarray:
        data = volume.astype(np.float32, copy=False)
        if self.identity:
            return data
        # zoom sizes its output round(n * factor), so the factor is the shape's ratio;
        # with grid_mode=False its samples then sit where `scale` puts them.
        factor = np.asarray(self.shape, dtype=float) / np.asarray(volume.shape, dtype=float)
        resampled = zoom(data, factor, order=1, grid_mode=False)
        assert resampled.shape == self.shape, (resampled.shape, self.shape)
        return resampled

    def points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float).reshape(-1, 3) * self.scale

    def back(self, mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """The model's mask on the crop's own grid, by nearest neighbour."""
        if self.identity:
            return np.asarray(mask, dtype=bool)
        axes = [np.arange(n) * s for n, s in zip(shape, self.scale)]
        grid = np.meshgrid(*axes, indexing="ij")
        return map_coordinates(np.asarray(mask, dtype=np.uint8), grid, order=0, mode="nearest").astype(bool)


def _dense(points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The voxels of a one-voxel line through `points` (on the grid of `shape`), in order, its gaps bridged.

    The centreline is sampled every finest voxel, so on a cubic grid of that voxel
    consecutive samples are at most one apart; the bridging only matters elsewhere.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) > 1:
        gaps = np.ceil(np.linalg.norm(np.diff(points, axis=0), axis=1)).astype(int).clip(1)
        points = np.vstack(
            [np.linspace(a, b, n, endpoint=False) for a, b, n in zip(points[:-1], points[1:], gaps)] + [points[-1:]]
        )
    return np.clip(np.rint(points).astype(int), 0, np.asarray(shape) - 1)


def rasterise(points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A one-voxel line through `points`, as a mask of `shape`."""
    out = np.zeros(shape, dtype=bool)
    out[tuple(_dense(points, shape).T)] = True
    return out


@dataclass(frozen=True)
class Stroke:
    """One piece of the scribble: a one-voxel line in the box ``[low, high)`` of the model's grid."""

    crop: np.ndarray  # bool, of shape high - low
    low: np.ndarray  # (3,) int
    high: np.ndarray  # (3,) int


#: A stroke is at most this many model voxels across. nnInteractive predicts one
#: patch (192³, zoomed out at most 4x) around the centre of its latest prompt, so one
#: scribble along a long path is cut to that window: measured on a 500-voxel tube, a
#: single scribble found only the middle 200 voxels (dice 0.52, points 0.82). In
#: strokes of 160, each its own prediction, the whole tube is found.
STROKE_EXTENT = 160


def strokes(points: np.ndarray, shape: tuple[int, ...], extent: int = STROKE_EXTENT) -> list[Stroke]:
    """The scribble along `points` as consecutive strokes, each at most `extent` voxels across.

    Neighbouring strokes share their end voxel, so together they are one unbroken line.
    """
    dense = _dense(points, shape)
    pieces: list[Stroke] = []
    start = 0
    while start < len(dense):
        low = high = dense[start]
        end = start + 1
        while end < len(dense):
            new_low, new_high = np.minimum(low, dense[end]), np.maximum(high, dense[end])
            if (new_high - new_low).max() >= extent:
                break
            low, high, end = new_low, new_high, end + 1
        crop = np.zeros(high - low + 1, dtype=bool)
        crop[tuple((dense[start:end] - low).T)] = True
        pieces.append(Stroke(crop=crop, low=low.copy(), high=high + 1))
        if end >= len(dense):
            break
        start = end - 1
    return pieces


def clip_to_path(
    found: np.ndarray, line: Centreline, samples: np.ndarray, spacing: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(coords, inside)``: the corridor voxels, and which of them are found and connect to the path.

    The model answers with everything the prompts suggest, which along a neurite
    can be the whole cell. What lies within `radius` of the prompted stretch of the
    path is kept, and of that only the parts touching the centreline: a neighbour
    crossing the corridor stays out.
    """
    shape = found.shape
    stretch = Centreline(
        points=line.points[samples[0] : samples[-1] + 1],
        tangents=line.tangents[samples[0] : samples[-1] + 1],
        normals=line.normals[samples[0] : samples[-1] + 1],
        binormals=line.binormals[samples[0] : samples[-1] + 1],
        step=line.step,
    )
    coords, _, _ = corridor(shape, stretch, spacing, radius)
    kept = np.zeros(shape, dtype=bool)
    kept[tuple(coords.T)] = found[tuple(coords.T)]
    parts, _ = label(kept, structure=np.ones((3, 3, 3)))
    on_path = np.clip(np.rint(stretch.points / spacing).astype(int), 0, np.asarray(shape) - 1)
    touching = np.unique(parts[tuple(on_path.T)])
    touching = touching[touching > 0]
    return coords, np.isin(parts[tuple(coords.T)], touching)


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


class Interactor:
    """What `segment` needs of nnInteractive. ``nni.NnInteractive`` is the real one; tests pass a fake.

    `segment` takes the model's image (raw intensities, (z, y, x)) and the prompts on
    its grid (points, or the strokes of a scribble, in path order), and answers with
    a bool mask of the image's shape. `key` names the
    image, so an implementation can keep it set between calls.
    """

    def segment(
        self,
        key: object,
        image: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        scribble: list[Stroke] | None,
    ) -> np.ndarray:
        raise NotImplementedError


def segment(
    volume: np.ndarray,
    vertices: np.ndarray,
    spacing: np.ndarray,
    params: NniParams,
    interactor: Interactor,
    *,
    field: np.ndarray | None = None,
    key: object = None,
) -> Segmentation:
    """Segment the structure along `vertices` (crop-local index coordinates) in `volume`.

    `field` is ``smooth(volume, spacing, params.sigma_um)``: pass it when it is
    cached. `key` names the crop for the interactor (it is the same image whenever
    the key is), so a re-run on the same crop skips setting the image again.
    """
    clock = _Clock()
    spacing = np.asarray(spacing, dtype=np.float64)
    shape = tuple(int(n) for n in volume.shape)
    if field is None:
        field = smooth(volume, spacing, params.sigma_um)
    clock.lap("smooth")

    line = centreline_of(vertices, spacing, float(spacing.min()))
    if params.refine:
        line = refine_centreline(field, line, spacing, params)
    fg, bg = profile(field, line, spacing, params)
    samples = prompted_samples(np.clip(fg - bg, 0.0, None), line.step, params)
    clock.lap("refine")

    empty = Prompts(np.zeros((0, 3)), np.zeros((0, 3)), None)
    if len(samples) == 0:
        return Segmentation(
            mask=np.zeros((0, 0, 0), dtype=bool), origin=np.zeros(3, dtype=int), centreline=line.points / spacing,
            arclength=line.arclength, prompts=empty, summary=summarise(np.zeros((0, 0, 0), bool), line, None, spacing),
            timings=clock.timings,
        )
    prompts = prompts_for(line, samples, spacing, shape, params)

    grid = Grid.of(shape, spacing, params.isotropic)
    image = grid.image(volume)
    clock.lap("resample")
    found = interactor.segment(
        (key, grid.shape),
        image,
        grid.points(prompts.positive),
        grid.points(prompts.negative),
        None if prompts.scribble is None else strokes(grid.points(prompts.scribble), grid.shape),
    )
    clock.lap("predict")
    found = grid.back(found, shape)

    coords, inside = clip_to_path(found, line, samples, spacing, params.radius_um)
    mask, origin = _finish(coords, inside, shape)
    clock.lap("finish")
    return Segmentation(
        mask=mask,
        origin=origin,
        centreline=line.points / spacing,
        arclength=line.arclength,
        prompts=prompts,
        summary=summarise(mask, line, origin, spacing),
        timings=clock.timings,
    )


def summarise(mask: np.ndarray, line: Centreline, origin: np.ndarray | None, spacing: np.ndarray) -> dict[str, float]:
    """Volume, the length of path the segmentation covers, and the diameter of a tube of both."""
    volume = float(mask.sum()) * float(np.prod(spacing))
    covered = 0.0
    if mask.any():
        index = np.rint(line.points / spacing).astype(int) - origin
        within = np.all((index >= 0) & (index < mask.shape), axis=1)
        hit = np.zeros(len(index), dtype=bool)
        hit[within] = mask[tuple(index[within].T)]
        if hit.any():
            covered = float(line.arclength[np.flatnonzero(hit)[-1]] - line.arclength[np.flatnonzero(hit)[0]])
    return {
        "path_length_um": float(line.arclength[-1]),
        "segmented_length_um": covered,
        "volume_um3": volume,
        "n_voxels": float(mask.sum()),
        "mean_diameter_um": float(2.0 * np.sqrt(volume / (np.pi * covered))) if covered > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# A synthetic AIS, for the tests (copied from path_segment_core)
# --------------------------------------------------------------------------- #


@dataclass
class Synthetic:
    volume: np.ndarray
    spacing: np.ndarray
    truth: np.ndarray  # (m, 3) index coordinates of the true centreline
    truth_arclength: np.ndarray
    truth_mask: np.ndarray  # bool, the AIS tube between `ais` bounds
    rough: np.ndarray  # (n, 3) index coordinates of a jittered hand-drawn path
    ais: tuple[float, float]  # true start/end, µm along the true centreline
    radius_um: float


def synthetic_ais(seed: int = 0) -> Synthetic:
    """A curved axon: bright AIS from 3 to 11 µm fading to 40 % distally, dim elsewhere.

    The volume is (z, y, x) at 0.3 x 0.1 x 0.1 µm, the tube 0.5 µm in radius. The
    rough path is six vertices on the true line, each knocked up to 0.3 µm sideways,
    and it runs 3 µm past the AIS on both ends, as a quick click-through would.
    """
    rng = np.random.default_rng(seed)
    spacing = np.array([0.3, 0.1, 0.1])
    shape = (40, 170, 170)
    radius, ais = 0.5, (3.0, 11.0)

    t = np.linspace(0.0, 1.0, 4000)
    line = np.column_stack(
        [6.0 + 2.0 * np.sin(np.pi * t), 3.0 + 11.0 * t, 3.0 + 9.0 * t + 2.0 * np.sin(2 * np.pi * t)]
    )
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])

    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing="ij"), -1)
    distance, nearest = cKDTree(line).query(grid.reshape(-1, 3))
    distance, along = distance.reshape(shape), length[nearest].reshape(shape)

    def strength(s: np.ndarray) -> np.ndarray:
        plateau = np.where(s < 6.0, 1.0, 1.0 - 0.6 * (s - 6.0) / (ais[1] - 6.0))
        edge = 1 / (1 + np.exp(-(s - ais[0]) / 0.1)) * 1 / (1 + np.exp((s - ais[1]) / 0.1))
        return plateau * edge

    tube = 1 / (1 + np.exp((distance - radius) / 0.05))
    volume = 100.0 + 1000.0 * strength(along) * tube + 80.0 * tube  # a faint axon all along
    volume = gaussian_filter(volume, sigma=0.12 / spacing) + rng.normal(0, 20.0, shape)
    truth_mask = (distance <= radius) & (along >= ais[0]) & (along <= ais[1])

    picks = np.interp(np.linspace(0.0, length[-1], 6), length, np.arange(len(line))).astype(int)
    rough = line[picks].copy()
    jitter = rng.normal(0, 1, rough.shape)
    rough += 0.3 * jitter / np.linalg.norm(jitter, axis=1, keepdims=True)
    return Synthetic(
        volume=np.clip(volume, 0, 65535).astype(np.uint16),
        spacing=spacing,
        truth=line / spacing,
        truth_arclength=length,
        truth_mask=truth_mask,
        rough=rough / spacing,
        ais=ais,
        radius_um=radius,
    )


def paste(mask: np.ndarray, origin: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A box mask placed back into a full volume of `shape`."""
    full = np.zeros(shape, dtype=bool)
    if mask.size:
        full[tuple(slice(int(o), int(o) + n) for o, n in zip(origin, mask.shape))] = mask
    return full


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def distance_to(points: np.ndarray, line: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Physical distance from each of `points` to the polyline `line` (both index coordinates)."""
    return cKDTree(line * spacing).query(points * spacing)[0]
