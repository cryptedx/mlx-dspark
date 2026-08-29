"""JIT loading and residency management for several MLX targets on one HTTP port.

The normal server deliberately remains a one-model server.  ``ModelPool`` is an opt-in
owner for the on-demand mode: it serializes MLX through :class:`MLXRuntime`, protects every
accepted request with a lease and only evicts inactive, unpinned models.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
PROFILE_KEYS = {
    "mode", "drafter", "drafter_bits", "max_draft_tokens", "confidence_threshold",
    "enable_thinking", "reasoning_effort", "prefix_cache", "prefix_cache_dir",
    "prefix_cache_max_ram_mb", "default_max_tokens", "max_tokens_cap",
    "default_temperature", "default_top_p", "default_top_k", "prefix_cache_slots",
    "prefix_cache_rungs", "lookup_drafts", "lookup_long_draft", "wide_gemm_min",
    "cpu_split", "small_m", "sdpa_split", "warmup", "kv_bits", "context_window",
}
PROFILE_ALIASES = {
    "max_draft": "max_draft_tokens",
    "confidence": "confidence_threshold",
}


class PoolError(RuntimeError):
    """An API-safe pool failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, status: int = 400,
                 retry_after: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class LoadProfile:
    """The effective, session-local Engine.load configuration for one target."""

    options: dict[str, Any]


@dataclass
class PreparedModel:
    model_id: str
    target_path: str
    target_repo: str
    drafter_path: str | None
    drafter_repo: str | None
    mode: str
    profile: LoadProfile
    estimated_bytes: int
    kv_bytes: int
    warning: str | None = None


@dataclass
class ModelSlot:
    model_id: str
    profile: LoadProfile
    state: str = "loading"
    engine: Any = None
    prepared: PreparedModel | None = None
    keep_loaded: bool = False
    leases: int = 0
    last_idle_at: float = field(default_factory=time.monotonic)
    loading_started_at: float | None = None
    error: str | None = None
    restore_error: str | None = None
    failure_until: float = 0.0
    eviction_reason: str | None = None
    profile_pending: bool = False
    pending_profile: LoadProfile | None = None


def kv_bytes_per_token(cfg: dict, kv_bits: int | None = None) -> int | None:
    """The shared context-KV estimate used by Engine warnings and pool admission."""
    for current in (cfg.get("text_config") or {}, cfg):
        layers = current.get("num_hidden_layers")
        heads = current.get("num_attention_heads")
        if not (isinstance(layers, int) and layers > 0
                and isinstance(heads, int) and heads > 0):
            continue
        kv_heads = current.get("num_key_value_heads") or heads
        head_dim = current.get("head_dim") or (current.get("hidden_size") or 0) // heads
        if not head_dim:
            continue
        attn = layers
        types = current.get("layer_types") or current.get("layers_block_type")
        pattern = current.get("hybrid_override_pattern")
        if isinstance(types, list) and types:
            attn = sum(1 for item in types
                       if isinstance(item, str) and item in ("full_attention", "attention"))
        elif isinstance(pattern, str) and pattern:
            attn = pattern.count("*")
        elif isinstance(current.get("full_attention_interval"), int) \
                and current["full_attention_interval"] > 0:
            attn = layers // current["full_attention_interval"]
        bytes_per_element = 2.0 if not kv_bits else (kv_bits + 0.5) / 8
        return int(int(attn) * int(kv_heads) * int(head_dim) * 2 * bytes_per_element)
    return None


class ModelPool:
    """At most two resident primary models, selected per request through ``model``.

    ``loader`` is injected to keep the state machine model-free in tests.  In production it is
    ``Engine.load`` and receives the shared runtime executor.
    """

    is_model_pool = True

    def __init__(
        self,
        *,
        runtime,
        loader: Callable[..., Any],
        load_defaults: dict[str, Any] | None = None,
        max_resident: int = 2,
        idle_ttl_s: float = 15 * 60,
        failure_backoff_s: float = 30,
        inventory: Callable[[], list[str]] | None = None,
        prepare: Callable[[str, LoadProfile, bool], PreparedModel] | None = None,
        memory_snapshot: Callable[[], tuple[int, int | None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_resident not in (1, 2):
            raise ValueError("max_resident must be 1 or 2")
        self.runtime = runtime
        self._loader = loader
        self._defaults = self._profile_from(dict(load_defaults or {}), base={})
        self.max_resident = max_resident
        self.idle_ttl_s = max(0.0, float(idle_ttl_s))
        self.failure_backoff_s = max(0.0, float(failure_backoff_s))
        self._inventory = inventory or self._installed_model_ids
        self._prepare_override = prepare
        self._memory_snapshot = memory_snapshot or self._live_memory
        self._clock = clock
        self._slots: dict[str, ModelSlot] = {}
        self._profiles: dict[str, LoadProfile] = {}
        self._keep_loaded_preferences: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._request_local = threading.local()
        self._closed = False
        self._reaper_stop = threading.Event()
        self._reaper: threading.Thread | None = None
        if hasattr(runtime, "attach_pool"):
            runtime.attach_pool(self)
        if self.idle_ttl_s > 0:
            self._reaper = threading.Thread(target=self._reap_loop,
                                            name="model-pool-reaper", daemon=True)
            self._reaper.start()

    # ------------------------------------------------------------------ public state

    def has_active_leases(self) -> bool:
        with self._lock:
            return any(slot.leases > 0 or slot.state in ("loading", "restoring")
                       for slot in self._slots.values())

    def telemetry_log(self, model: str | None = None):
        """Return a resident model's round log without holding a request lease."""
        canonical = self._canonical_model(model, require_local=False)
        with self._lock:
            slot = self._slots.get(canonical)
            if slot is None or slot.state != "ready" or slot.engine is None:
                raise PoolError("model_not_loaded", "model is not loaded", status=503)
            return slot.engine.rounds

    @property
    def is_closing(self) -> bool:
        with self._lock:
            return self._closed

    def status(self) -> dict:
        with self._lock:
            models = [self._slot_status(slot) for slot in self._slots.values()]
            ready = [slot for slot in self._slots.values() if slot.state == "ready"]
            loading = any(slot.state in ("loading", "restoring", "evicting")
                          for slot in self._slots.values())
            only = ready[0] if len(ready) == 1 else None
            return {
                "ready": bool(ready),
                "loading": loading,
                "model": only.model_id if only else None,
                "error": next((slot.error for slot in self._slots.values() if slot.error), None),
                "max_resident_models": self.max_resident,
                "idle_ttl_seconds": self.idle_ttl_s,
                "models": models,
            }

    def model_catalog(self) -> list[str]:
        return sorted(set(self._inventory()))

    def models_payload(self) -> dict:
        with self._lock:
            by_id = {slot.model_id: slot for slot in self._slots.values()}
        return {
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "created": int(by_id[model_id].loading_started_at or 0)
                           if model_id in by_id else 0,
                "owned_by": "mlx-dspark",
                "x_mlx_dspark": {
                    "loaded": by_id[model_id].state == "ready" if model_id in by_id else False,
                    "keep_loaded": by_id[model_id].keep_loaded if model_id in by_id else False,
                    "state": by_id[model_id].state if model_id in by_id else "absent",
                },
            } for model_id in self.model_catalog()],
        }

    def register_profile(self, model: str, profile: dict[str, Any]) -> dict:
        canonical = self._canonical_model(model, require_local=False)
        keep_loaded = profile.get("keep_loaded")
        if keep_loaded is not None and not isinstance(keep_loaded, bool):
            raise PoolError("model_profile_conflict", "keep_loaded must be a boolean")
        next_profile = self._profile_from(
            {key: value for key, value in profile.items() if key != "keep_loaded"})
        with self._changed:
            if keep_loaded is not None:
                self._keep_loaded_preferences[canonical] = keep_loaded
            self._profiles[canonical] = next_profile
            slot = self._slots.get(canonical)
            if slot is not None and keep_loaded is not None and slot.state == "ready":
                slot.keep_loaded = keep_loaded
                if not keep_loaded:
                    slot.last_idle_at = self._clock()
            if slot is not None and slot.state == "ready" and slot.profile != next_profile:
                slot.profile_pending = True
                slot.pending_profile = next_profile
            elif slot is not None and slot.profile == next_profile:
                slot.profile_pending = False
                slot.pending_profile = None
            self._changed.notify_all()
            return {"model": canonical, "profile_pending_reload": bool(
                slot and slot.profile_pending), "profile": self._external_profile(next_profile)}

    @contextmanager
    def lease(self, model: str | None, *, local_only: bool = True) -> Iterator[Any]:
        """Acquire a request lease and bind its engine for the existing handler proxy calls."""
        canonical, selected = self._acquire(model, local_only=local_only)
        previous = getattr(self._request_local, "engine", None)
        self._request_local.engine = selected
        try:
            yield selected
        finally:
            self._request_local.engine = previous
            self._release(canonical)

    def admin_load(self, model: str, *, options: dict[str, Any] | None = None,
                   keep_loaded: bool = True, reload: bool = False) -> dict:
        """Manual load/pin entrypoint.  Manual loads may use the established download path."""
        canonical = self._canonical_model(model, require_local=False)
        engine_options = {key: value for key, value in (options or {}).items()
                          if key != "keep_loaded"}
        profile = self._profile_from(engine_options) if options else self._profile_for(canonical)
        with self._changed:
            slot = self._slots.get(canonical)
            if slot is not None and slot.state == "ready":
                if slot.profile != profile:
                    if not reload:
                        raise PoolError("model_profile_conflict",
                                        "profile changed; repeat the load with reload=true",
                                        status=409)
                    if slot.leases:
                        raise PoolError("model_active", "model has active or queued requests",
                                        status=409)
            if options:
                # Validate conflicts before mutating the stored profile or pin preference.
                # The condition is an RLock, so this stays one atomic state transition.
                self.register_profile(canonical, options)
            self._keep_loaded_preferences[canonical] = keep_loaded
            if (slot is not None and slot.state == "ready"
                    and slot.profile == profile and not reload):
                slot.keep_loaded = bool(keep_loaded)
                if not slot.keep_loaded:
                    slot.last_idle_at = self._clock()
                return self.status()
        if reload and slot is not None and slot.state == "ready":
            self._reload(canonical, profile, keep_loaded=keep_loaded)
            return self.status()
        with self.lease(canonical, local_only=False), self._changed:
            resident = self._slots[canonical]
            resident.keep_loaded = bool(keep_loaded)
            if not resident.keep_loaded:
                resident.last_idle_at = self._clock()
        return self.status()

    def unload(self, model: str | None = None, *, all_models: bool = False) -> dict:
        with self._changed:
            ready = [slot for slot in self._slots.values() if slot.state == "ready"]
            reserved = [slot for slot in self._slots.values()
                        if slot.state in ("ready", "loading", "evicting", "restoring")]
            if all_models:
                # ``all: true`` is the one explicit full reset.  Failed entries carry only
                # backoff/status state but must disappear too, otherwise a model-free reset
                # would misleadingly retain a stale failed slot.
                targets = list(self._slots.values())
            elif model is None:
                if len(reserved) > 1:
                    raise PoolError("model_required", "model is required when several models are resident")
                targets = ready
            else:
                canonical = self._canonical_model(model, require_local=False)
                slot = self._slots.get(canonical)
                if slot is not None and slot.state in ("loading", "evicting", "restoring"):
                    raise PoolError("model_active", "model has active or queued requests", status=409)
                targets = [slot] if slot is not None else []
            for slot in targets:
                if slot.leases or slot.state in ("loading", "evicting", "restoring"):
                    raise PoolError("model_active", "model has active or queued requests", status=409)
            for slot in targets:
                slot.state = "evicting"
                slot.eviction_reason = "manual"
        for slot in targets:
            self._close_slot(slot)
            with self._changed:
                self._slots.pop(slot.model_id, None)
                self._changed.notify_all()
        return self.status()

    def shed_caches(self, level: str) -> dict:
        """Called by the process-wide MemoryGuard on the shared MLX worker."""
        with self._lock:
            slots = [slot for slot in self._slots.values()
                     if slot.state == "ready" and slot.engine is not None]
        results = []
        for slot in slots:
            prefix = getattr(slot.engine, "prefix", None)
            if prefix is not None:
                with contextlib.suppress(Exception):
                    results.append({"model": slot.model_id, **prefix.shed(level)})
        if level == "critical":
            self._evict_one(reason="memory")
        return {"action": "pool", "models": results}

    def close(self) -> None:
        with self._changed:
            if self._closed:
                return
            self._closed = True
            self._reaper_stop.set()
            while any(slot.leases > 0 or slot.state in ("loading", "restoring")
                      for slot in self._slots.values()):
                self._changed.wait(timeout=0.2)
            slots = [slot for slot in self._slots.values() if slot.engine is not None]
            for slot in slots:
                slot.state = "evicting"
        if self._reaper is not None:
            self._reaper.join(timeout=2)
        for slot in slots:
            self._close_slot(slot)
        with self._changed:
            self._slots.clear()
            self._changed.notify_all()
        self.runtime.close()

    # ------------------------------------------------------------ request selection

    def __getattr__(self, name: str):
        """Compatibility proxy for the existing handler while a request lease is active."""
        selected = getattr(self._request_local, "engine", None)
        if selected is None:
            raise RuntimeError("a model lease is required for this pool operation")
        return getattr(selected, name)

    def _acquire(self, requested: str | None, *, local_only: bool) -> tuple[str, Any]:
        canonical = self._canonical_model(requested, require_local=local_only)
        while True:
            with self._changed:
                if self._closed:
                    raise PoolError("pool_busy", "server is shutting down", status=503)
                slot = self._slots.get(canonical)
                if slot is not None and slot.state == "ready":
                    slot.leases += 1
                    return canonical, slot.engine
                if slot is not None and slot.state in ("loading", "evicting", "restoring"):
                    self._changed.wait(timeout=0.2)
                    continue
                now = self._clock()
                if slot is not None and slot.state == "failed" and now < slot.failure_until:
                    retry = max(1, int(slot.failure_until - now))
                    raise PoolError("model_load_failed", slot.error or "model load failed",
                                    status=503, retry_after=retry)
                profile = self._profile_for(canonical)
                if slot is None:
                    slot = ModelSlot(model_id=canonical, profile=profile,
                                     keep_loaded=self._keep_loaded_preferences.get(canonical, False))
                    self._slots[canonical] = slot
                else:
                    slot.profile = profile
                    slot.keep_loaded = self._keep_loaded_preferences.get(canonical, False)
                    slot.state = "loading"
                    slot.error = None
                    slot.restore_error = None
                slot.loading_started_at = time.time()
            try:
                prepared = self._prepare(canonical, profile, local_only)
                selected = self._load_new(slot, prepared)
            except PoolError as error:
                self._record_load_failure(slot, error)
                raise
            except Exception as error:
                wrapped = PoolError("model_load_failed", f"could not load {canonical}: {error}",
                                    status=503)
                self._record_load_failure(slot, wrapped)
                raise wrapped from error
            with self._changed:
                slot.engine = selected
                slot.prepared = prepared
                slot.profile = prepared.profile
                slot.state = "ready"
                slot.error = None
                slot.restore_error = None
                slot.failure_until = 0.0
                slot.profile_pending = False
                slot.pending_profile = None
                slot.leases += 1
                self._changed.notify_all()
                return canonical, selected

    def _release(self, canonical: str) -> None:
        with self._changed:
            slot = self._slots.get(canonical)
            if slot is None:
                return
            slot.leases = max(0, slot.leases - 1)
            if slot.leases == 0:
                slot.last_idle_at = self._clock()
                self._changed.notify_all()
        self.runtime.on_pool_idle()

    # -------------------------------------------------------------- loading / eviction

    def _load_new(self, slot: ModelSlot, prepared: PreparedModel):
        victim = None
        with self._changed:
            resident = [item for item in self._slots.values()
                        if item is not slot and item.state in ("ready", "loading", "evicting",
                                                               "restoring")]
            if len(resident) >= self.max_resident:
                victim = self._choose_victim_locked()
                if victim is None:
                    code = "pool_all_pinned" if all(item.keep_loaded for item in resident
                                                     if item.state == "ready") else "pool_busy"
                    raise PoolError(code, "no inactive unpinned model can be evicted", status=503,
                                    retry_after=1 if code == "pool_busy" else None)
                victim.state = "evicting"
                victim.eviction_reason = self._eviction_reason(victim)
                self._changed.notify_all()
        try:
            self._admit(prepared, victim)
        except Exception:
            if victim is not None:
                with self._changed:
                    victim.state = "ready"
                    victim.eviction_reason = None
                    self._changed.notify_all()
            raise
        if victim is not None:
            self._close_slot(victim)
        try:
            loaded = self._load_engine(prepared)
        except Exception as target_error:
            if victim is not None:
                self._restore(victim)
            raise target_error
        if victim is not None:
            with self._changed:
                self._slots.pop(victim.model_id, None)
                self._changed.notify_all()
        return loaded

    def _reload(self, canonical: str, profile: LoadProfile, *, keep_loaded: bool) -> None:
        prepared = self._prepare(canonical, profile, False)
        with self._changed:
            old = self._slots.get(canonical)
            if old is None or old.state != "ready":
                raise PoolError("model_not_found", "model is not resident")
            if old.leases:
                raise PoolError("model_active", "model has active or queued requests", status=409)
            old.state = "evicting"
            old.eviction_reason = "reload"
            replacement = ModelSlot(model_id=canonical, profile=profile,
                                    state="loading", keep_loaded=bool(keep_loaded),
                                    loading_started_at=time.time())
            self._slots[canonical] = replacement
            self._changed.notify_all()
        try:
            self._admit(prepared, old)
            self._close_slot(old)
            loaded = self._load_engine(prepared)
        except Exception as target_error:
            self._restore(old)
            with self._changed:
                if old.state == "ready":
                    self._slots[canonical] = old
                else:
                    replacement.state = "failed"
                    replacement.error = str(target_error)
                    replacement.failure_until = self._clock() + self.failure_backoff_s
                self._changed.notify_all()
            raise
        with self._changed:
            replacement.engine = loaded
            replacement.prepared = prepared
            replacement.state = "ready"
            replacement.profile_pending = False
            replacement.pending_profile = None
            self._slots[canonical] = replacement
            self._changed.notify_all()

    def _restore(self, victim: ModelSlot) -> None:
        if victim.prepared is None:
            return
        with self._changed:
            victim.state = "restoring"
            self._changed.notify_all()
        try:
            restored = self._load_engine(victim.prepared)
        except Exception as error:  # noqa: BLE001 -- restoration is deliberately best effort
            with self._changed:
                victim.engine = None
                victim.state = "failed"
                victim.restore_error = str(error)
                victim.error = f"restore failed: {error}"
                victim.failure_until = self._clock() + self.failure_backoff_s
                self._slots[victim.model_id] = victim
                self._changed.notify_all()
            return
        with self._changed:
            victim.engine = restored
            victim.state = "ready"
            victim.eviction_reason = None
            self._slots[victim.model_id] = victim
            self._changed.notify_all()

    def _record_load_failure(self, slot: ModelSlot, error: PoolError) -> None:
        with self._changed:
            # Capacity and pin conflicts are transient pool states, not a poisoned model profile.
            if error.code in {"pool_all_pinned", "pool_busy", "memory_budget_exceeded"}:
                if self._slots.get(slot.model_id) is slot:
                    self._slots.pop(slot.model_id, None)
            else:
                slot.engine = None
                slot.state = "failed"
                slot.error = str(error)
                slot.failure_until = self._clock() + self.failure_backoff_s
            self._changed.notify_all()

    def _close_slot(self, slot: ModelSlot) -> None:
        engine = slot.engine
        if engine is not None:
            engine.close()
            # Allocator cache is process-wide, so the shared runtime — never the individual
            # Engine — owns this release after an eviction/reload/shutdown.
            self.runtime.clear_allocator_cache()
        slot.engine = None

    def _evict_one(self, *, reason: str) -> bool:
        with self._changed:
            victim = self._choose_victim_locked()
            if victim is None:
                return False
            victim.state = "evicting"
            victim.eviction_reason = reason
        self._close_slot(victim)
        with self._changed:
            self._slots.pop(victim.model_id, None)
            self._changed.notify_all()
        return True

    def _choose_victim_locked(self) -> ModelSlot | None:
        candidates = [slot for slot in self._slots.values()
                      if slot.state == "ready" and not slot.keep_loaded and slot.leases == 0]
        if not candidates:
            return None
        return min(candidates, key=lambda slot: slot.last_idle_at)

    def _eviction_reason(self, slot: ModelSlot) -> str:
        if self.idle_ttl_s and self._clock() - slot.last_idle_at >= self.idle_ttl_s:
            return "ttl"
        return "lru"

    # -------------------------------------------------------------------- preflight

    def _prepare(self, canonical: str, profile: LoadProfile, local_only: bool) -> PreparedModel:
        if self._prepare_override is not None:
            return self._prepare_override(canonical, profile, local_only)
        from .load import resolve_mode

        options = dict(profile.options)
        requested_mode = options.get("mode", "auto")
        explicit_drafter = options.get("drafter")
        try:
            mode, target_repo, drafter_repo = resolve_mode(
                canonical, mode=requested_mode, drafter=explicit_drafter)
        except ValueError as error:
            raise PoolError("model_load_failed", str(error)) from error
        target_path = self._checkpoint_path(target_repo, kind="model",
                                            allow_download=not local_only)
        warning = None
        drafter_path = None
        if drafter_repo is not None:
            try:
                # A drafter is never fetched as an implicit side effect: auto falls back to
                # lookup, and an explicitly requested one gets a clear local-only error.
                drafter_path = self._checkpoint_path(drafter_repo, kind="drafter",
                                                     allow_download=False)
            except PoolError:
                if requested_mode == "auto" and explicit_drafter is None:
                    mode, target_repo, drafter_repo = resolve_mode(canonical, mode="lookup")
                    drafter_path = None
                    warning = "matched drafter is not local; using drafter-free lookup"
                else:
                    raise
        estimate, kv = self._estimate(target_path, drafter_path, options)
        return PreparedModel(model_id=canonical, target_path=target_path,
                             target_repo=target_repo, drafter_path=drafter_path,
                             drafter_repo=drafter_repo, mode=mode, profile=profile,
                             estimated_bytes=estimate, kv_bytes=kv, warning=warning)

    def _load_engine(self, prepared: PreparedModel):
        kwargs = dict(prepared.profile.options)
        kwargs.update({
            "model": prepared.target_path,
            "mode": prepared.mode,
            "drafter": prepared.drafter_path,
            "executor": self.runtime.executor,
            "owns_executor": False,
            "memory_guard": False,
            "wired_limit": False,
        })
        loaded = self._loader(**kwargs)
        # The loader received local paths to guarantee an offline JIT path.  HTTP and status
        # surfaces keep the canonical user-facing ids instead.
        loaded.model_id = prepared.model_id
        loaded.target_repo = prepared.target_repo
        loaded.drafter_repo = prepared.drafter_repo
        if prepared.warning:
            getattr(loaded, "load_notes", []).append(prepared.warning)
        return loaded

    def _local_path(self, repo_or_path: str, *, required: bool) -> str:
        from .load import local_dir

        direct = local_dir(repo_or_path)
        if direct is not None:
            return direct
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_or_path, local_files_only=True)
        except Exception as error:  # local_files_only guarantees no network request
            if required:
                raise PoolError("model_not_local", f"{repo_or_path!r} is not fully local") from error
            return repo_or_path

    def _checkpoint_path(self, repo_or_path: str, *, kind: str, allow_download: bool) -> str:
        path = self._local_path(repo_or_path, required=not allow_download)
        if allow_download and not os.path.isdir(path):
            from .download import ensure_local

            ensure_local(repo_or_path)
            path = self._local_path(repo_or_path, required=True)
        self._validate_checkpoint(path, repo_or_path, kind=kind, required=True)
        return path

    @staticmethod
    def _validate_checkpoint(path: str, repo: str, *, kind: str, required: bool) -> None:
        if not os.path.isdir(path):
            if required:
                raise PoolError("model_not_local", f"{kind} {repo!r} is not fully local")
            return
        config_path = os.path.join(path, "config.json")
        try:
            with open(config_path) as file:
                json.load(file)
        except Exception as error:
            raise PoolError("model_not_local", f"{kind} {repo!r} has no readable config.json") from error
        if not any(name.endswith(".safetensors") for name in os.listdir(path)):
            raise PoolError("model_not_local", f"{kind} {repo!r} has no weights")
        if kind == "model" and not any(os.path.exists(os.path.join(path, name))
                                         for name in ("tokenizer.json", "tokenizer_config.json",
                                                      "tokenizer.model")):
            raise PoolError("model_not_local", f"model {repo!r} has no readable tokenizer")

    @staticmethod
    def _estimate(target_path: str, drafter_path: str | None,
                  options: dict[str, Any]) -> tuple[int, int]:
        def weight_bytes(path: str | None) -> int:
            if not path or not os.path.isdir(path):
                return 0
            return sum(item.stat().st_size for item in Path(path).glob("*.safetensors")
                       if item.is_file())

        cfg = {}
        with contextlib.suppress(Exception), open(os.path.join(target_path, "config.json")) as file:
            cfg = json.load(file)
        kv_per_token = kv_bytes_per_token(cfg, options.get("kv_bits")) or 0
        window = int(options.get("context_window") or 0)
        kv = max(0, kv_per_token * window)
        weights = weight_bytes(target_path) + weight_bytes(drafter_path)
        # Loader/transient buffers are real but model-specific.  The conservative floor avoids
        # treating a tiny tokenizer-only directory as a zero-cost load.
        scratch = max(256 * 1024 ** 2, int(weights * 0.05))
        return weights + kv + scratch, kv

    def _admit(self, prepared: PreparedModel, victim: ModelSlot | None) -> None:
        active, budget = self._memory_snapshot()
        if not budget:
            return                         # tests / non-Metal systems have no trustworthy budget
        victim_bytes = victim.prepared.estimated_bytes if victim and victim.prepared else 0
        reserve = max(2 * GIB, int(budget * 0.10))
        projected = max(0, active - victim_bytes) + prepared.estimated_bytes
        if projected <= budget - reserve:
            return
        # Prefix snapshots and MLX's retained allocator buffers are safely reclaimable, even
        # on pinned models.  Re-check once after shedding rather than keeping two budgets.
        self.runtime.executor.submit(lambda: self.shed_caches("warn")).result()
        active, budget = self._memory_snapshot()
        if budget and max(0, active - victim_bytes) + prepared.estimated_bytes > budget - reserve:
            raise PoolError("memory_budget_exceeded",
                            "loading this profile would exceed the MLX working-set budget",
                            status=503)

    @staticmethod
    def _live_memory() -> tuple[int, int | None]:
        try:
            import mlx.core as mx

            return (int(mx.get_active_memory()) + int(mx.get_cache_memory()),
                    mx.device_info().get("max_recommended_working_set_size"))
        except Exception:  # noqa: BLE001 -- model-free tests and non-Metal hosts
            return 0, None

    # ------------------------------------------------------------------- profiles/catalog

    def _profile_for(self, canonical: str) -> LoadProfile:
        with self._lock:
            return self._profiles.get(canonical, self._defaults)

    def _profile_from(self, options: dict[str, Any], *, base: dict[str, Any] | None = None
                      ) -> LoadProfile:
        merged = dict(self._defaults.options if base is None and hasattr(self, "_defaults") else base or {})
        for raw_key, value in options.items():
            key = PROFILE_ALIASES.get(raw_key, raw_key)
            if key in {"model", "family", "target", "memory_guard", "wired_limit", "batch_widths"}:
                continue
            if key not in PROFILE_KEYS:
                raise PoolError("model_profile_conflict", f"unsupported profile option {raw_key!r}")
            merged[key] = value
        mode = merged.get("mode", "auto")
        if mode not in ("auto", "dspark", "dflash", "lookup", "baseline"):
            raise PoolError("model_profile_conflict", "profile mode is invalid")
        context = merged.get("context_window")
        if context is not None and (isinstance(context, bool) or not isinstance(context, int)
                                    or (context != 0 and context < 1024)):
            raise PoolError("model_profile_conflict", "context_window must be 0 or >= 1024")
        kv_bits = merged.get("kv_bits")
        if kv_bits not in (None, 0, 4, 8):
            raise PoolError("model_profile_conflict", "kv_bits must be 0, 4, or 8")
        max_draft = merged.get("max_draft_tokens")
        if max_draft is not None and max_draft != "auto" and (
                isinstance(max_draft, bool) or not isinstance(max_draft, int)):
            raise PoolError("model_profile_conflict", "max_draft must be an integer or 'auto'")
        confidence = merged.get("confidence_threshold")
        if confidence is not None and (isinstance(confidence, bool)
                                       or not isinstance(confidence, (int, float))
                                       or not 0.0 <= confidence <= 1.0):
            raise PoolError("model_profile_conflict",
                            "confidence_threshold must be a number in [0, 1]")
        for key in ("prefix_cache", "lookup_drafts", "small_m", "sdpa_split", "warmup",
                    "enable_thinking"):
            if key in merged and merged[key] is not None and not isinstance(merged[key], bool):
                raise PoolError("model_profile_conflict", f"{key} must be a boolean")
        for key in ("prefix_cache_slots", "prefix_cache_rungs", "default_max_tokens",
                    "max_tokens_cap", "lookup_long_draft", "drafter_bits"):
            if key in merged and (isinstance(merged[key], bool)
                                  or not isinstance(merged[key], int) or merged[key] < 0):
                raise PoolError("model_profile_conflict", f"{key} must be a non-negative integer")
        for key in ("default_temperature", "default_top_p"):
            if key in merged and merged[key] is not None and (
                    isinstance(merged[key], bool) or not isinstance(merged[key], (int, float))):
                raise PoolError("model_profile_conflict", f"{key} must be a number")
        if "default_top_k" in merged and merged["default_top_k"] is not None and (
                isinstance(merged["default_top_k"], bool)
                or not isinstance(merged["default_top_k"], int)):
            raise PoolError("model_profile_conflict", "default_top_k must be an integer")
        if "drafter" in merged and merged["drafter"] is not None and not isinstance(
                merged["drafter"], str):
            raise PoolError("model_profile_conflict", "drafter must be a string")
        return LoadProfile(merged)

    @staticmethod
    def _external_profile(profile: LoadProfile) -> dict[str, Any]:
        out = dict(profile.options)
        if "max_draft_tokens" in out:
            out["max_draft"] = out.pop("max_draft_tokens")
        return out

    def _canonical_model(self, model: str | None, *, require_local: bool) -> str:
        if model is None or not str(model).strip():
            with self._lock:
                ready = [slot.model_id for slot in self._slots.values() if slot.state == "ready"]
            if len(ready) == 1:
                return ready[0]
            raise PoolError("model_required", "request needs a model when the pool is not singular")
        raw = str(model).strip()
        expanded = os.path.realpath(os.path.expanduser(raw))
        if os.path.isdir(expanded):
            return expanded
        catalog = self.model_catalog()
        exact = [item for item in catalog if item == raw]
        if exact:
            return exact[0]
        known = list(self._profiles)
        if raw in known:
            return raw
        aliases = [item for item in catalog if os.path.basename(item).lower()
                   == os.path.basename(raw).lower()]
        aliases += [item for item in known if os.path.basename(item).lower()
                    == os.path.basename(raw).lower() and item not in aliases]
        if len(aliases) == 1:
            return aliases[0]
        if len(aliases) > 1:
            raise PoolError("model_alias_ambiguous", f"model alias {raw!r} matches {aliases}")
        if require_local:
            raise PoolError("model_not_local", f"no locally installed model matches {raw!r}")
        return raw

    def _installed_model_ids(self) -> list[str]:
        from .diagnostics import installed_models

        complete = []
        for row in installed_models():
            if row.get("kind") != "model":
                continue
            repo = str(row["repo"])
            try:
                path = self._local_path(repo, required=True)
                self._validate_checkpoint(path, repo, kind="model", required=True)
            except PoolError:
                continue
            complete.append(repo)
        return complete

    # ------------------------------------------------------------------------- reaper

    def _reap_loop(self) -> None:
        interval = min(30.0, max(1.0, self.idle_ttl_s / 4))
        while not self._reaper_stop.wait(interval):
            self.reap()

    def reap(self) -> bool:
        if self.idle_ttl_s <= 0:
            return False
        with self._lock:
            candidates = [slot for slot in self._slots.values()
                          if slot.state == "ready" and not slot.keep_loaded and slot.leases == 0
                          and self._clock() - slot.last_idle_at >= self.idle_ttl_s]
        if not candidates:
            return False
        return self._evict_one(reason="ttl")

    def _slot_status(self, slot: ModelSlot) -> dict:
        # Pool health is aggregate by design, so expose the effective knobs on each resident
        # slot. Unloaded slots only have profile values; capability fields stay nil until the
        # engine exists instead of advertising controls that may not work for that target.
        profile = slot.profile.options
        live = slot.engine
        if live is not None:
            live = getattr(live, "engine", live)
        max_draft = (
            "auto" if getattr(live, "cap_controller", None) is not None
            else str(getattr(live, "max_draft_tokens", profile.get("max_draft_tokens"))
                     or "auto")
        )
        kv_bits = getattr(getattr(live, "target", None), "kv_bits", None)
        if kv_bits is None:
            kv_bits = profile.get("kv_bits") or 0
        return {
            "model": slot.model_id,
            "state": slot.state,
            "ready": slot.state == "ready",
            "keep_loaded": slot.keep_loaded,
            "pinned": slot.keep_loaded,
            "leases": slot.leases,
            "last_used": slot.last_idle_at,
            "error": slot.error,
            "restore_error": slot.restore_error,
            "profile_pending_reload": slot.profile_pending,
            "eviction_reason": slot.eviction_reason,
            "warning": slot.prepared.warning if slot.prepared else None,
            "mode": slot.prepared.mode if slot.prepared else slot.profile.options.get("mode"),
            "target": slot.prepared.target_repo if slot.prepared else slot.model_id,
            "drafter": slot.prepared.drafter_repo if slot.prepared else None,
            "max_draft": max_draft,
            "context_window": getattr(live, "context_window", profile.get("context_window")),
            "context_tokens": getattr(live, "_last_context", None),
            "max_output_tokens": getattr(live, "max_tokens_cap", profile.get("max_tokens_cap")),
            "supports_reasoning_effort": (
                bool(getattr(live, "supports_reasoning_effort", False))
                if live is not None else None
            ),
            "confidence_threshold": getattr(
                live, "confidence_threshold", profile.get("confidence_threshold")
            ),
            "race_arm_confidence": True if live is not None else None,
            "lookup_drafts": getattr(live, "lookup_drafts", profile.get("lookup_drafts")),
            "kv_bits": kv_bits,
            "cpu_split": getattr(live, "cpu_split", None) if live is not None else None,
            "thinking_default": (
                "off" if getattr(live, "template_defaults", {}).get("enable_thinking") is False
                else "on"
            ) if live is not None else None,
        }
