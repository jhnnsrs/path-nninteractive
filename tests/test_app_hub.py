"""The app against the services it uses: a real volume, a real path, a real label crop.

`teststack` logs the app in to a hub made for the tests. It needs Docker and
konstruktor, and about a minute to start, so these are not part of a bare
`uv run pytest`: every test here is marked `hub`. Run them with
`uv run pytest -m hub`. The segmentation itself runs nnInteractive, so it
needs the model's weights and a GPU.

What needs no server is tested in `test_app.py`.
"""

import pytest
import xarray as xr
from arkitekt.testing import TestStack
from mikro import Mikro
from mikro.api.schema import AnnotationKind, AxisInput, AxisType

from path_nninteractive import geometry

# Every test in this file runs against a hub.
pytestmark = pytest.mark.hub


def test_a_path_drawn_on_a_volume_is_segmented_into_a_label_crop(teststack: TestStack[None]):
    mikro = teststack.runtime.require(Mikro)
    synth = geometry.synthetic_ais()
    stored = mikro.create_array_dataset(
        data=xr.DataArray(synth.volume[None], dims=("c", "z", "y", "x")),
        scales=[],
        name="Synthetic AIS",
        axes=[
            AxisInput(name="c", type=AxisType.CHANNEL),
            AxisInput(name="z", type=AxisType.SPACE),
            AxisInput(name="y", type=AxisType.SPACE),
            AxisInput(name="x", type=AxisType.SPACE),
        ],
    )
    lens = stored.lens()
    # Drawn in the lens' own space: voxel indices, (z, y, x). The volume is not
    # calibrated, so the radius is in voxels too.
    path = lens.draw(AnnotationKind.PATH, synth.rough.tolist(), name="axon")

    label, mesh = teststack.call(
        "nninteractive_along_path", path=path, lens=lens, radius_um=10.0, sigma_um=1.5, min_contrast=0.5
    )

    voxels = label.data.squeeze().compute().values
    assert voxels.ndim == 3 and voxels.any()
    assert mesh is None
    # The tube is 5 voxels in radius and about 80 long between the AIS bounds.
    assert 0.5 * synth.truth_mask.sum() < voxels.sum() < 3.0 * synth.truth_mask.sum()
