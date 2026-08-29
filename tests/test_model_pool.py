"""Model-free coverage for the on-demand residency state machine."""

from __future__ import annotations

import threading
from concurrent.futures import Future

import pytest

from mlx_dspark.model_pool import ModelPool, PoolError, PreparedModel


class _InlineExecutor:
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # noqa: BLE001 -- test executor mirrors Future semantics.
            future.set_exception(error)
        return future


class _Runtime:
    def __init__(self):
        self.executor = _InlineExecutor()
        self.pool = None
        self.closed = False

    def attach_pool(self, pool):
        self.pool = pool

    def on_pool_idle(self):
        return None

    def clear_allocator_cache(self):
        return None

    def close(self):
        self.closed = True


class _Engine:
    def __init__(self, path):
        self.path = path
        self.closed = False
        self.load_notes = []

    def close(self):
        self.closed = True


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _prepared(model, profile, _local_only):
    return PreparedModel(model, f"/models/{model.rsplit('/', 1)[-1]}", model, None, None,
                         profile.options.get("mode", "auto"), profile,
                         estimated_bytes=1, kv_bytes=0)


def _pool(*, max_resident=2, clock=None, loader=None, ttl=900, prepare=_prepared):
    calls = []

    def default_loader(**kwargs):
        calls.append(kwargs)
        return _Engine(kwargs["model"])

    pool = ModelPool(
        runtime=_Runtime(), loader=loader or default_loader,
        load_defaults={"mode": "auto", "context_window": 65536},
        max_resident=max_resident, idle_ttl_s=ttl, clock=clock or _Clock(),
        inventory=lambda: ["org/One", "org/Two", "org/Three"],
        prepare=prepare, memory_snapshot=lambda: (0, None),
    )
    return pool, calls


def test_aliases_share_one_resident_engine_and_a_lease():
    pool, calls = _pool()
    try:
        with pool.lease("One") as first:
            assert first.model_id == "org/One"
            with pool.lease("org/One") as second:
                assert second is first
                assert pool.status()["models"][0]["leases"] == 2
        assert len(calls) == 1
        assert pool.status()["models"][0]["leases"] == 0
    finally:
        pool.close()


def test_waiting_aliases_singleflight_the_same_load():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        started.set()
        release.wait(timeout=2)
        return _Engine(kwargs["model"])

    pool, _ = _pool(loader=loader)
    results = []

    def acquire(model):
        with pool.lease(model) as engine:
            results.append(engine)

    first = threading.Thread(target=acquire, args=("One",))
    second = threading.Thread(target=acquire, args=("org/One",))
    first.start()
    assert started.wait(timeout=1)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    try:
        assert len(calls) == 1
        assert len(results) == 2 and results[0] is results[1]
    finally:
        pool.close()


def test_lru_never_evicts_pinned_or_leased_models():
    pool, _ = _pool(max_resident=1)
    try:
        pool.admin_load("org/One", keep_loaded=True)
        with pytest.raises(PoolError, match="inactive unpinned") as error, pool.lease("org/Two"):
            pass
        assert error.value.code == "pool_all_pinned"
        pool.admin_load("org/One", keep_loaded=False)
        with pool.lease("org/Two"):
            pass
        models = pool.status()["models"]
        assert [slot["model"] for slot in models if slot["state"] == "ready"] == ["org/Two"]
    finally:
        pool.close()


def test_unpin_restarts_ttl_and_reaper_only_unloads_idle_models():
    clock = _Clock()
    pool, _ = _pool(max_resident=1, clock=clock, ttl=15)
    try:
        pool.admin_load("org/One", keep_loaded=True)
        clock.now = 100
        pool.admin_load("org/One", keep_loaded=False)
        clock.now = 114
        assert pool.reap() is False
        clock.now = 115
        assert pool.reap() is True
        assert not pool.status()["models"]
    finally:
        pool.close()


def test_profile_change_requires_explicit_reload():
    pool, calls = _pool()
    try:
        pool.admin_load("org/One", keep_loaded=True)
        pool.register_profile("org/One", {"context_window": 32768, "mode": "lookup"})
        assert pool.status()["models"][0]["profile_pending_reload"] is True
        with pytest.raises(PoolError) as error:
            pool.admin_load("org/One", options={"context_window": 32768, "mode": "lookup"})
        assert error.value.code == "model_profile_conflict"
        pool.admin_load("org/One", options={"context_window": 32768, "mode": "lookup"},
                        reload=True)
        assert len(calls) == 2
        assert pool.status()["models"][0]["mode"] == "lookup"
    finally:
        pool.close()


def test_failed_replacement_best_effort_restores_victim_and_backs_off_target():
    calls = []

    def loader(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"].endswith("Two"):
            raise RuntimeError("broken checkpoint")
        return _Engine(kwargs["model"])

    pool, _ = _pool(max_resident=1, loader=loader)
    try:
        with pool.lease("org/One"):
            pass
        with pytest.raises(PoolError) as error, pool.lease("org/Two"):
            pass
        assert error.value.code == "model_load_failed"
        state = {slot["model"]: slot for slot in pool.status()["models"]}
        assert state["org/One"]["state"] == "ready"
        before = len(calls)
        with pytest.raises(PoolError), pool.lease("org/Two"):
            pass
        assert len(calls) == before
    finally:
        pool.close()


def test_full_unload_clears_a_failed_slot_too():
    def loader(**kwargs):
        if kwargs["model"].endswith("Two"):
            raise RuntimeError("broken checkpoint")
        return _Engine(kwargs["model"])

    pool, _ = _pool(loader=loader)
    try:
        with pytest.raises(PoolError), pool.lease("org/Two"):
            pass
        assert pool.status()["models"]
        pool.unload(all_models=True)
        assert pool.status()["models"] == []
    finally:
        pool.close()


def test_targeted_unload_rejects_an_active_lease():
    pool, _ = _pool()
    try:
        with pool.lease("org/One"):
            with pytest.raises(PoolError) as error:
                pool.unload("org/One")
            assert error.value.code == "model_active"
    finally:
        pool.close()


def test_shutdown_waits_for_an_active_streaming_lease():
    pool, _ = _pool()
    acquired = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def stream_request():
        with pool.lease("org/One"):
            acquired.set()
            release.wait(timeout=2)

    request = threading.Thread(target=stream_request)
    request.start()
    assert acquired.wait(timeout=1)

    def shutdown():
        pool.close()
        closed.set()

    stopping = threading.Thread(target=shutdown)
    stopping.start()
    assert not closed.wait(timeout=0.05)
    assert pool.is_closing is True
    release.set()
    request.join(timeout=2)
    stopping.join(timeout=2)
    assert closed.is_set()


def test_auto_mode_falls_back_without_fetching_a_missing_drafter(monkeypatch):
    pool, _ = _pool(prepare=None)

    def resolve(_model, *, mode, drafter=None, **_kwargs):
        if mode == "lookup":
            return "lookup", "org/One", None
        return "dflash", "org/One", drafter or "org/Missing-Drafter"

    calls = []

    def checkpoint(repo, *, kind, allow_download):
        calls.append((repo, kind, allow_download))
        if kind == "drafter":
            raise PoolError("model_not_local", "drafter is not local")
        return "/models/One"

    import mlx_dspark.load as load

    monkeypatch.setattr(load, "resolve_mode", resolve)
    monkeypatch.setattr(pool, "_checkpoint_path", checkpoint)
    monkeypatch.setattr(pool, "_estimate", lambda *_args: (1, 0))
    try:
        automatic = pool._prepare("org/One", pool._profile_for("org/One"), True)
        assert automatic.mode == "lookup"
        assert automatic.warning is not None
        assert all(not allow for _repo, _kind, allow in calls)

        explicit = pool._profile_from({"mode": "dflash"})
        with pytest.raises(PoolError) as error:
            pool._prepare("org/One", explicit, True)
        assert error.value.code == "model_not_local"
    finally:
        pool.close()
