"""nnInteractive behind the `Interactor` the geometry asks for.

torch and nnInteractive are imported only when the model is first loaded, so
declaring the app (which `arkitekt plugin release` does inside the image, on a
runner with no GPU) never waits for them.

One inference session serves the whole process. It holds one image at a time,
preprocessed on the GPU, so a re-run on the same crop (another prompt spacing,
other negatives) only resets the interactions. Every call holds one lock: the
session is stateful, and two tasks prompting it at once would mix their prompts.

The weights are read from a folder, never fetched at run time:
``$NNINTERACTIVE_MODEL_DIR/models/nnInteractive_v1.0`` (the image bakes them into
``/opt/nninteractive``). nnInteractive's own `ensure_model_available` asks Hugging
Face for its manifest first, which in a pod without egress is a timeout on every
start.

The weights are CC BY-NC-SA 4.0 (non-commercial, share-alike), by the Division of
Medical Image Computing, DKFZ; see the README. The code is Apache-2.0.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Interactor, Stroke

logger = logging.getLogger(__name__)

MODEL_ID = "nnInteractive_v1.0"

_SESSION_LOCK = threading.Lock()


def model_folder(model_id: str = MODEL_ID) -> Path:
    """Where the weights of `model_id` are, without asking the network."""
    root = Path(os.environ.get("NNINTERACTIVE_MODEL_DIR", Path.home() / ".nninteractive")).expanduser()
    folder = root / "models" / model_id
    if not (folder / "fold_0" / "checkpoint_final.pth").exists():
        raise FileNotFoundError(
            f"The nnInteractive weights are not in {folder}. Download them once with "
            f"`nninteractive-download-model {model_id}` (into $NNINTERACTIVE_MODEL_DIR, default ~/.nninteractive)."
        )
    return folder


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _Session:
    """The one inference session, and which image it holds."""

    def __init__(self, session: Any):
        self.session = session
        self.image_key: object = None


@lru_cache(maxsize=1)
def load_session(model_id: str = MODEL_ID) -> _Session:
    """The inference session, with the model loaded and warmed up, once per process."""
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    where = torch.device(device())
    folder = model_folder(model_id)
    logger.info("Loading nnInteractive %s on %s from %s", model_id, where, folder)
    session = nnInteractiveInferenceSession(
        device=where,
        use_torch_compile=False,  # the one-off compile is minutes; a pod may live for one call
        verbose=False,
        torch_n_threads=min(os.cpu_count() or 1, 8),
        do_autozoom=True,
        enable_undo=False,
    )
    session.initialize_from_trained_model_folder(str(folder))
    return _Session(session)


class NnInteractive(Interactor):
    """An `Interactor` on the process' inference session."""

    def segment(
        self,
        key: object,
        image: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        scribble: list[Stroke] | None,
    ) -> np.ndarray:
        import torch

        held = load_session()
        with _SESSION_LOCK, torch.inference_mode():
            session = held.session
            # A key of None never matches, so an unnamed image is always set anew.
            if key is None or held.image_key != key:
                session.set_image(np.asarray(image, dtype=np.float32)[None])
                held.image_key = key
            else:
                session.reset_interactions()
            target = torch.zeros(image.shape, dtype=torch.uint8)
            session.set_target_buffer(target)
            # One prediction per prompt: nnInteractive predicts around the last one
            # it was given, each refining the result so far.
            if scribble is not None:
                # Each stroke as a crop with its box, the form the authors recommend:
                # the whole image's mask per stroke would be orders of magnitude larger.
                for stroke in scribble:
                    session.add_scribble_interaction(
                        stroke.crop.astype(np.uint8),
                        include_interaction=True,
                        interaction_bbox=[[int(a), int(b)] for a, b in zip(stroke.low, stroke.high)],
                    )
            else:
                for point in positive:
                    session.add_point_interaction(tuple(float(c) for c in point), include_interaction=True)
            for point in negative:
                session.add_point_interaction(tuple(float(c) for c in point), include_interaction=False)
            return target.numpy().astype(bool)
