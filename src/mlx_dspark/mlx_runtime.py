"""The process-wide MLX resources shared by a resident-model pool.

MLX arrays are thread/stream-affine.  A process with several resident targets must therefore
still have exactly one place that owns MLX execution, allocator-cache release and pressure
handling.  This module intentionally owns only those confirmed process-wide resources.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


class SerialExecutor:
    """One worker that safely accepts nested calls from that same worker.

    ``ThreadPoolExecutor.submit(...).result()`` deadlocks when code already running on its
    only worker tries to submit another MLX operation.  Returning an already-complete Future
    for that case preserves the normal executor interface without a second execution path.
    """

    def __init__(self, *, thread_name_prefix: str = "mlx-runtime"):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._local = threading.local()
        self._closed = False
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs) -> Future:
        if getattr(self._local, "active", False):
            future: Future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 -- Future preserves executor semantics.
                future.set_exception(exc)
            return future
        with self._lock:
            if self._closed:
                raise RuntimeError("MLX runtime is shut down")
            return self._executor.submit(self._run, fn, args, kwargs)

    def _run(self, fn: Callable[..., Any], args: tuple, kwargs: dict):
        self._local.active = True
        try:
            return fn(*args, **kwargs)
        finally:
            self._local.active = False

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait)


class _PoolPrefixCaches:
    """Small adapter so the established MemoryGuard can shed every pool cache at once."""

    def __init__(self, pool):
        self._pool = pool

    def shed(self, level: str) -> dict:
        return self._pool.shed_caches(level)


class MLXRuntime:
    """Owner of shared MLX state for ``ModelPool``.

    The pool attaches after construction so this module stays independent of model loading and
    HTTP routing.  Individual engines receive ``executor`` but never own or stop it.
    """

    def __init__(self, *, wired_limit: bool = False, memory_guard: bool = True):
        self.executor = SerialExecutor()
        self._closed = False
        self._pool = None
        self.memory_guard = None
        self._memory_guard_enabled = memory_guard
        if wired_limit:
            from .load import apply_wired_limit

            self.executor.submit(apply_wired_limit).result()

    def attach_pool(self, pool) -> None:
        if self._pool is not None:
            raise RuntimeError("an MLX runtime can only own one model pool")
        self._pool = pool
        if not self._memory_guard_enabled:
            return
        from .memory_guard import MemoryGuard

        self.memory_guard = MemoryGuard(
            prefix=_PoolPrefixCaches(pool),
            submit=self.executor.submit,
            is_busy=pool.has_active_leases,
        ).start()

    def on_pool_idle(self) -> None:
        """Let a pressure shed queued during generation run at the next safe point."""
        if self.memory_guard is not None:
            self.memory_guard.on_idle()

    def clear_allocator_cache(self) -> None:
        def clear() -> None:
            with contextlib.suppress(Exception):
                import mlx.core as mx

                mx.clear_cache()

        self.executor.submit(clear).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.memory_guard is not None:
            self.memory_guard.stop()
            self.memory_guard = None
        self.clear_allocator_cache()
        self.executor.shutdown(wait=True)
