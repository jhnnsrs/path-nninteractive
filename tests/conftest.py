"""Loads arkitekt's test fixtures: `call` and `teststack`, and the app they call."""

import sys
from pathlib import Path

import pytest

# The app is a module of the project, not of tests/: a test imports it from there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest_plugins = ["arkitekt.testing"]


@pytest.fixture()
def arkitekt_context():
    """The app declares a context (its voxel cache); every started app gets a fresh one."""
    from app import NniCache

    return NniCache()
