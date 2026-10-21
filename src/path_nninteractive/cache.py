"""What every task of this agent shares: decoded bricks, array handles, derived arrays.

``VoxelCache`` is copied from ``testing/path_segment_app.py``. The SAM image
embeddings of a straightened stack are the costliest thing a run computes (the
encoder, once per frame), and only the prompts change between re-runs while tuning,
so they are kept too, in a small ``Lru`` of their own: a few hundred frames of
embeddings are close to a gigabyte.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from itertools import product
from typing import Any, Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: Bricks for an array that is not chunked (a plain numpy-backed one): the store's own layout.
_DEFAULT_BRICK = 64


class VoxelCache:
    """Decoded bricks, array handles and smoothed fields, shared by every task of this agent.

    Thread-safe: the lock guards the dictionaries, never a download, so two tasks
    can fetch at once. Two tasks fetching the *same* missing brick both fetch it.
    That costs a little bandwidth and never a wrong answer.
    """

    def __init__(self, max_bytes: int, handle_ttl_s: float = 600.0, max_fields: int = 4):
        self.max_bytes = max_bytes
        self.handle_ttl_s = handle_ttl_s
        self.max_fields = max_fields
        self._lock = threading.Lock()
        self._bricks: OrderedDict[tuple[str, tuple[int, ...]], np.ndarray] = OrderedDict()
        self._bytes = 0
        self._handles: dict[str, tuple[xr.DataArray, float]] = {}
        self._edges: dict[str, list[np.ndarray]] = {}
        self._meta: dict[tuple[str, str], tuple[Any, float]] = {}
        self._fields: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._warming: dict[str, threading.Thread] = {}
        #: Bricks pulled from the store since start, and reads answered without one.
        self.fetched = 0
        self.hits = 0

    @property
    def bytes(self) -> int:
        return self._bytes

    # -- handles ------------------------------------------------------------ #

    def handle(self, key: str, open_: Callable[[], xr.DataArray], *, fresh: bool = False) -> xr.DataArray:
        """The lazy array for `key`, opened by `open_` when absent, stale or `fresh` is asked."""
        now = time.monotonic()
        with self._lock:
            cached = self._handles.get(key)
            if cached is not None and not fresh and now - cached[1] < self.handle_ttl_s:
                return cached[0]
        data = open_()
        with self._lock:
            self._handles[key] = (data, now)
            self._edges.setdefault(key, _brick_edges(data))
        return data

    def remember(self, key: str, what: str, compute: Callable[[], Any]) -> Any:
        """A per-lens fact (the voxel size), kept as long as a handle."""
        now = time.monotonic()
        with self._lock:
            cached = self._meta.get((key, what))
            if cached is not None and now - cached[1] < self.handle_ttl_s:
                return cached[0]
        value = compute()
        with self._lock:
            self._meta[(key, what)] = (value, now)
        return value

    # -- bricks ------------------------------------------------------------- #

    def read(
        self, key: str, open_: Callable[[], xr.DataArray], low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """The ``[low, high)`` box of `key` as numpy, from cached bricks where it can."""
        data = self.handle(key, open_)
        edges = self._edges[key]
        wanted = list(product(*(_brick_range(e, int(lo), int(hi)) for e, lo, hi in zip(edges, low, high))))

        with self._lock:
            have = {}
            for brick in wanted:
                found = self._bricks.get((key, brick))
                if found is not None:
                    self._bricks.move_to_end((key, brick))
                    have[brick] = found
        missing = [brick for brick in wanted if brick not in have]
        self.hits += len(have)
        if missing:
            have.update(self._fetch(key, open_, missing))

        out = np.empty(tuple(int(h - l) for l, h in zip(low, high)), dtype=data.dtype)
        for brick, values in have.items():
            start = np.array([e[i] for e, i in zip(edges, brick)])
            stop = start + np.array(values.shape)
            a, b = np.maximum(start, low), np.minimum(stop, high)
            out[tuple(slice(int(x - l), int(y - l)) for x, y, l in zip(a, b, low))] = values[
                tuple(slice(int(x - s), int(y - s)) for x, y, s in zip(a, b, start))
            ]
        return out

    def _fetch(
        self, key: str, open_: Callable[[], xr.DataArray], bricks: list[tuple[int, ...]]
    ) -> dict[tuple[int, ...], np.ndarray]:
        """Pull `bricks` in one scheduler call, retrying once on a freshly opened store."""
        edges = self._edges[key]
        for attempt in (0, 1):
            data = self.handle(key, open_, fresh=attempt > 0)
            try:
                values = _compute([data.data[_brick_slices(edges, brick)] for brick in bricks])
                break
            except Exception as error:  # noqa: BLE001 - an expired grant surfaces as anything
                if attempt:
                    raise
                logger.warning("Reading %s failed (%s); reopening the store once", key, error)
        fetched = dict(zip(bricks, values))
        with self._lock:
            for brick, array in fetched.items():
                if (key, brick) not in self._bricks:
                    self._bricks[(key, brick)] = array
                    self._bytes += array.nbytes
            while self._bytes > self.max_bytes and self._bricks:
                _, evicted = self._bricks.popitem(last=False)
                self._bytes -= evicted.nbytes
            self.fetched += len(fetched)
        return fetched

    def warm(self, key: str, data: xr.DataArray) -> bool:
        """Pull the whole of `key` in the background if it fits a quarter of the budget.

        Uses the handle already opened by the task that asked. The thread outlives
        that task and has no client to open another, so if the handle stops working,
        the warm-up stops and later reads fetch what they need themselves.
        """
        if data.size * data.dtype.itemsize > self.max_bytes // 4:
            return False

        def pull() -> None:
            edges = self._edges[key]
            everything = list(product(*(range(len(e) - 1) for e in edges)))
            try:
                for start in range(0, len(everything), 32):
                    batch = [b for b in everything[start : start + 32] if (key, b) not in self._bricks]
                    if batch:
                        self._fetch(key, lambda: data, batch)
            except Exception as error:  # noqa: BLE001
                logger.warning("Warming %s stopped: %s", key, error)
            finally:
                with self._lock:
                    self._warming.pop(key, None)

        with self._lock:
            if key in self._warming:
                return False
            thread = threading.Thread(target=pull, name=f"warm-{key}", daemon=True)
            self._warming[key] = thread
        thread.start()
        return True

    def settle(self, timeout: float | None = None) -> None:
        """Wait for background warm-ups to finish."""
        with self._lock:
            threads = list(self._warming.values())
        for thread in threads:
            thread.join(timeout)

    # -- fields ------------------------------------------------------------- #

    def field(self, key: tuple[Any, ...], compute: Callable[[], np.ndarray]) -> np.ndarray:
        """A derived array (the smoothed crop), kept for the last few distinct keys."""
        with self._lock:
            found = self._fields.get(key)
            if found is not None:
                self._fields.move_to_end(key)
                return found
        value = compute()
        with self._lock:
            self._fields[key] = value
            while len(self._fields) > self.max_fields:
                self._fields.popitem(last=False)
        return value


def _brick_edges(data: xr.DataArray) -> list[np.ndarray]:
    """Per axis, the index where each brick starts, plus the end: the array's own chunking."""
    chunks = getattr(data, "chunks", None)
    if chunks:
        return [np.concatenate([[0], np.cumsum(c)]) for c in chunks]
    return [np.append(np.arange(0, n, _DEFAULT_BRICK), n) for n in data.shape]


def _brick_range(edges: np.ndarray, low: int, high: int) -> range:
    first = int(np.searchsorted(edges, low, side="right")) - 1
    last = int(np.searchsorted(edges, max(high - 1, low), side="right")) - 1
    return range(max(first, 0), min(last, len(edges) - 2) + 1)


def _brick_slices(edges: list[np.ndarray], brick: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(int(e[i]), int(e[i + 1])) for e, i in zip(edges, brick))


def _compute(arrays: list[Any]) -> list[np.ndarray]:
    """Many lazy slices in one dask call (parallel, one graph); plain arrays as they are."""
    if any(hasattr(a, "dask") for a in arrays):
        import dask

        return [np.asarray(a) for a in dask.compute(*arrays)]
    return [np.array(a) for a in arrays]


class Lru:
    """A thread-safe least-recently-used map of a few values, computed on a miss.

    The lock guards the map, never the computation: two tasks missing the same
    key both compute it. That costs time and never a wrong answer.
    """

    def __init__(self, size: int):
        self.size = size
        self._lock = threading.Lock()
        self._values: OrderedDict[Any, Any] = OrderedDict()
        #: How often a value was computed, and how often it was already there.
        self.misses = 0
        self.hits = 0

    def get(self, key: Any, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._values:
                self._values.move_to_end(key)
                self.hits += 1
                return self._values[key]
        value = compute()
        with self._lock:
            self.misses += 1
            self._values[key] = value
            while len(self._values) > self.size:
                self._values.popitem(last=False)
        return value
