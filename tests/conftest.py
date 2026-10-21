"""Loads arkitekt's test fixtures: `call` and `teststack`, and the app they call."""

import sys
from pathlib import Path

# The app is a module of the project, not of tests/: a test imports it from there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest_plugins = ["arkitekt.testing"]

