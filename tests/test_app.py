"""The app for itself, with no server.

The geometry is tested as the plain numpy it is, on a synthetic axon initial
segment: a curved tube, bright for 8 µm and fading, drawn over by a rough path.
The action runs against a stub mikro and a thresholding stand-in for the model,
so the placement, the clip, the upload and the caches are checked without a
GPU. Real nnInteractive runs last, and only where its weights are downloaded and
a GPU is there.

Run these with `uv run pytest`; the release workflow does too.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pytest
from arkitekt import Task
from arkitekt.runtime import Runtime
from mikro.api.schema import ValueRelation

import app as nni_app
from app import NniCache, Prompt, app, nninteractive_along_path
from path_nninteractive import geometry
from path_nninteractive.cache import VoxelCache

from stubs import StubLens, StubMikro, ThresholdInteractor, drawn_path, graph


@pytest.fixture(scope="module")
def synth() -> geometry.Synthetic:
    return geometry.synthetic_ais()


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> ThresholdInteractor:
    """The action's `NnInteractive` replaced by one thresholding stand-in for every call."""
    model = ThresholdInteractor()
    monkeypatch.setattr(nni_app, "NnInteractive", lambda: model)
    return model


# min_contrast 0.5 stops the prompts where the AIS fades (to 40 %), as the truth does.
PARAMS = dict(radius_um=1.0, sigma_um=0.15, min_contrast=0.5)


def full_mask(result: geometry.Segmentation, shape: tuple[int, ...]) -> np.ndarray:
    return geometry.paste(result.mask, result.origin, shape)


def long_tube(length: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(volume, spacing, truth, rough)``: a wobbling tube `length` voxels along x, uniformly bright.

    Longer than one of the model's patches, which is what a scribble has to be
    drawn in strokes for.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    spacing = np.array([0.3, 0.1, 0.1])
    shape = (40, 90, length)
    x = np.linspace(10, length - 10, 4000)
    line = np.column_stack(
        [20 + 4 * np.sin(x / length * 2 * np.pi), 45 + 15 * np.sin(x / length * 3 * np.pi + 0.5), x]
    )
    distance, _ = cKDTree(line * spacing).query(np.argwhere(np.ones(shape, dtype=bool)) * spacing)
    truth = (distance <= 0.5).reshape(shape)
    volume = 200 + 800 * gaussian_filter(truth.astype(float), [0.5, 1.5, 1.5]) + rng.normal(0, 40, shape)
    knots = line[np.linspace(0, len(line) - 1, 8).astype(int)]
    rough = knots + rng.normal(0, 0.2, knots.shape) / spacing * [1, 1, 0]
    return np.clip(volume, 0, None).astype(np.uint16), spacing, truth, rough


def tube(synth: geometry.Synthetic) -> np.ndarray:
    """The whole axon, AIS or not."""
    from scipy.spatial import cKDTree

    coords = np.argwhere(np.ones(synth.volume.shape, dtype=bool))
    distance, _ = cKDTree(synth.truth * synth.spacing).query(coords * synth.spacing)
    return (distance <= synth.radius_um).reshape(synth.volume.shape)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_the_cubic_grid_carries_points_and_masks_exactly(synth):
    """A point lands where the resampled image has it, and a mask comes back where it was."""
    grid = geometry.Grid.of(synth.volume.shape, synth.spacing, isotropic=True)
    assert grid.shape == (120, 170, 170)
    image = grid.image(synth.volume)
    assert image.shape == grid.shape

    # The brightest voxel of the crop is the brightest of the resampled image too.
    bright = np.array(np.unravel_index(np.argmax(synth.volume), synth.volume.shape))
    assert image[tuple(np.rint(grid.points(bright)[0]).astype(int))] == pytest.approx(synth.volume.max(), rel=0.02)

    # A mask drawn on the model's grid from the crop's own voxels comes back unchanged.
    model_mask = grid.image(synth.truth_mask.astype(np.float32)) >= 0.5
    back = grid.back(model_mask, synth.volume.shape)
    assert geometry.dice(back, synth.truth_mask) > 0.95


def test_a_long_scribble_is_drawn_in_strokes_that_make_one_line():
    """Each stroke fits the model's window, and together they are the whole, unbroken line.

    The model predicts around its latest prompt only, so one scribble along a long
    path would be found only around its middle.
    """
    from scipy.ndimage import label

    x = np.arange(5.0, 1195.0)
    points = np.column_stack([20 + 10 * np.sin(x / 200), 45 + 30 * np.cos(x / 150), x])
    shape = (40, 90, 1200)
    pieces = geometry.strokes(points, shape)

    assert len(pieces) >= 1190 // geometry.STROKE_EXTENT
    assert all((piece.high - piece.low).max() <= geometry.STROKE_EXTENT for piece in pieces)
    union = np.zeros(shape, dtype=bool)
    for piece in pieces:
        assert piece.crop.shape == tuple(piece.high - piece.low)
        box = tuple(slice(int(a), int(b)) for a, b in zip(piece.low, piece.high))
        union[box] |= piece.crop
    assert np.array_equal(union, geometry.rasterise(points, shape))
    assert label(union, structure=np.ones((3, 3, 3)))[1] == 1


def test_prompts_stay_where_the_structure_stands_out(synth):
    params = geometry.NniParams(**PARAMS, prompt="points")
    field = geometry.smooth(synth.volume, synth.spacing, params.sigma_um)
    line = geometry.centreline_of(synth.rough, synth.spacing, float(synth.spacing.min()))
    fg, bg = geometry.profile(field, line, synth.spacing, params)
    samples = geometry.prompted_samples(np.clip(fg - bg, 0, None), line.step, params)

    # The rough path starts and ends 3 µm off the AIS: neither end is prompted.
    assert line.arclength[samples[0]] > 1.5 and line.arclength[-1] - line.arclength[samples[-1]] > 1.5
    assert np.all(np.diff(samples)[:-1] * line.step >= params.prompt_every_um - 1e-9)


def test_what_leaves_the_path_is_clipped(synth):
    """A model that follows the whole axon is kept to the prompted stretch, and to the radius."""
    model = ThresholdInteractor()
    result = geometry.segment(synth.volume, synth.rough, synth.spacing, geometry.NniParams(**PARAMS), model)
    found = full_mask(result, synth.volume.shape)

    assert geometry.dice(found, synth.truth_mask) > 0.7
    assert (found & tube(synth)).sum() / found.sum() > 0.6
    assert 6.0 < result.summary["segmented_length_um"] < 10.5
    assert model.calls == 1


def test_a_neighbour_crossing_the_corridor_stays_out(synth):
    """A bright blob inside the radius but not touching the path is not part of the result."""
    volume = synth.volume.copy()
    line = geometry.centreline_of(synth.truth, synth.spacing, float(synth.spacing.min()))
    i = len(line.points) // 2
    # Sideways in the y-x plane, where voxels are fine: 0.3 µm of gap to the tube's surface.
    sideways = np.cross(line.tangents[i], [1.0, 0.0, 0.0])
    sideways /= np.linalg.norm(sideways)
    side = np.rint((line.points[i] + 0.9 * sideways) / synth.spacing).astype(int)
    z, y, x = side
    volume[z, y - 1 : y + 2, x - 1 : x + 2] = volume.max()

    class Everything(geometry.Interactor):
        """Answers with everything bright, the blob too, as a model following the stain might."""

        def segment(self, key, image, positive, negative, scribble):
            return image > 0.5 * image.max()

    params = geometry.NniParams(radius_um=1.0, sigma_um=0.15, min_contrast=0.5, refine=False)
    result = geometry.segment(volume, synth.truth, synth.spacing, params, Everything())
    found = full_mask(result, synth.volume.shape)
    assert found.any() and not found[tuple(side)]


def test_nothing_to_prompt_is_empty_not_an_error():
    flat = np.full((20, 40, 40), 100, dtype=np.uint16)
    path = np.array([[10.0, 5.0, 5.0], [10.0, 35.0, 35.0]])
    model = ThresholdInteractor()
    result = geometry.segment(flat, path, np.ones(3), geometry.NniParams(radius_um=3.0), model)
    assert result.empty and model.calls == 0


# --------------------------------------------------------------------------- #
# The action, against a stub mikro
# --------------------------------------------------------------------------- #


def test_the_action_uploads_a_label_crop_placed_on_the_axon(synth, fake_model):
    mikro = StubMikro(graph(synth.spacing))
    label, mesh = nninteractive_along_path(
        drawn_path(synth.rough, synth.spacing), StubLens(synth.volume), **PARAMS, with_mesh=True,
        mikro=mikro, task=Task.local(), cache=NniCache(VoxelCache(512 * 1024**2)),
    )

    created = mikro.datasets[-1]
    array = created["data"]
    assert label.id == "label-1"
    assert tuple(array.dims) == ("c", "z", "y", "x") and array.shape[0] == 1
    translation, parent = created["derived_from"]
    assert translation.value_relation == ValueRelation.CATEGORIZED
    assert str(parent.annotation) == "path-1"
    assert created["scales"] == []

    full = np.zeros(synth.volume.shape, dtype=bool)
    box = tuple(slice(int(o), int(o) + n) for o, n in zip(translation.offset[1:], array.shape[1:]))
    full[box] = array.values[0].astype(bool)
    assert geometry.dice(full, synth.truth_mask) > 0.7

    surface = mikro.meshes[0]
    assert mesh is not None and surface["axes"] == ["x", "y", "z"]
    assert surface["store"].manifest.axes == ("x", "y", "z")
    assert surface["derived_from"][0].kind == "BY_DIMENSION"


def test_a_re_run_on_the_same_crop_keeps_the_image_and_the_voxels(synth, fake_model):
    mikro = StubMikro(graph(synth.spacing))
    lens = StubLens(synth.volume)
    cache = NniCache(VoxelCache(512 * 1024**2))
    path = drawn_path(synth.rough, synth.spacing)
    nninteractive_along_path(path, lens, **PARAMS, mikro=mikro, task=Task.local(), cache=cache)
    cache.voxels.settle()  # the first run pulls the whole small lens in the background
    fetched, opened = cache.voxels.fetched, lens.opened

    nninteractive_along_path(
        path, lens, **{**PARAMS, "min_contrast": 0.3}, prompt=Prompt.POINTS, n_negative=2,
        mikro=mikro, task=Task.local(), cache=cache,
    )
    assert fake_model.calls == 2 and fake_model.images_set == 1, "the image was set again for other prompts"
    assert cache.voxels.fetched == fetched and lens.opened == opened

    # Resampling changes what the model is given: that is a new image.
    nninteractive_along_path(path, lens, **PARAMS, isotropic=True, mikro=mikro, task=Task.local(), cache=cache)
    assert fake_model.images_set == 2


def test_nothing_to_segment_is_said_not_returned_empty(synth, fake_model):
    flat = np.full_like(synth.volume, 100)
    with pytest.raises(ValueError, match="stands out"):
        nninteractive_along_path(
            drawn_path(synth.rough, synth.spacing), StubLens(flat), **PARAMS,
            mikro=StubMikro(graph(synth.spacing)), task=Task.local(), cache=NniCache(VoxelCache(512 * 1024**2)),
        )


def test_the_app_starts_and_offers_its_actions(local_runtime: Runtime[None]):
    assert local_runtime.app is app
    assert {"nninteractive_along_path", "warm_nninteractive"} <= set(app.registry.implementations)


# --------------------------------------------------------------------------- #
# Real nnInteractive, where its weights and a GPU are on this machine
# --------------------------------------------------------------------------- #


def _ready() -> str | None:
    from path_nninteractive.nni import device, model_folder

    try:
        model_folder()
    except FileNotFoundError:
        return "the nnInteractive weights are not downloaded"
    if device() != "cuda":
        return "no GPU"
    return None


@pytest.mark.skipif(_ready() is not None, reason=str(_ready()))
def test_nninteractive_follows_the_axon(synth):
    from path_nninteractive.nni import NnInteractive

    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        result = geometry.segment(
            synth.volume, synth.rough, synth.spacing, geometry.NniParams(**PARAMS), NnInteractive(), key="synth"
        )
    seconds = time.perf_counter() - started
    found = full_mask(result, synth.volume.shape)
    score = geometry.dice(found, synth.truth_mask)
    print(f"nnInteractive dice {score:.3f} in {seconds:.1f} s: {result.timings}")

    # Measured 0.83 on an RTX 4070: it finds all of the AIS, and a voxel more around it.
    assert score > 0.7
    assert (found & synth.truth_mask).sum() / synth.truth_mask.sum() > 0.95
    assert 6.0 < result.summary["segmented_length_um"] < 10.5


@pytest.mark.skipif(_ready() is not None, reason=str(_ready()))
def test_nninteractive_finds_a_tube_longer_than_its_window():
    """Every stretch of a 500-voxel tube is found, not only the middle of one patch.

    A single scribble found only the middle 200 voxels here (dice 0.52); in strokes,
    every 50-voxel slab is (measured dice 0.87, 1.8 s).
    """
    from path_nninteractive.nni import NnInteractive

    volume, spacing, truth, rough = long_tube(500)
    with contextlib.redirect_stdout(io.StringIO()):
        result = geometry.segment(
            volume, rough, spacing, geometry.NniParams(radius_um=1.0, sigma_um=0.15), NnInteractive(), key="long"
        )
    found = full_mask(result, volume.shape)
    slabs = [(found & truth)[..., a : a + 50].sum() / truth[..., a : a + 50].sum() for a in range(0, 500, 50)]
    assert min(slabs) > 0.9, slabs
    assert geometry.dice(found, truth) > 0.8
