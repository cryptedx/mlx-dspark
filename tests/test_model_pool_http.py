"""Pool-specific HTTP contract, using no model weights."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import Future
from http.server import ThreadingHTTPServer

import pytest

from mlx_dspark.model_pool import ModelPool, PreparedModel
from mlx_dspark.server import make_handler


class _Executor:
    def submit(self, fn, /, *args, **kwargs):
        result = Future()
        try:
            result.set_result(fn(*args, **kwargs))
        except BaseException as error:  # noqa: BLE001 -- test executor mirrors Future semantics.
            result.set_exception(error)
        return result


class _Runtime:
    def __init__(self):
        self.executor = _Executor()
        self.memory_guard = None

    def attach_pool(self, _pool):
        return None

    def on_pool_idle(self):
        return None

    def clear_allocator_cache(self):
        return None

    def close(self):
        return None


class _Engine:
    mode = "lookup"
    max_tokens_cap = 32768
    load_notes = []

    def close(self):
        return None

    def metrics(self):
        return {"model": self.model_id, "mode": self.mode, "requests": 0}


def _prepare(model, profile, _local_only):
    return PreparedModel(model, f"/models/{model.rsplit('/', 1)[-1]}", model, None, None,
                         profile.options.get("mode", "auto"), profile, 1, 0)


@pytest.fixture
def pool_server():
    pool = ModelPool(runtime=_Runtime(), loader=lambda **_kwargs: _Engine(),
                     load_defaults={"mode": "auto", "context_window": 65536},
                     inventory=lambda: ["org/One", "org/Two"], prepare=_prepare,
                     memory_snapshot=lambda: (0, None), idle_ttl_s=0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(pool, api_key=None))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield pool, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        pool.close()


def _request(base, path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_profiles_health_catalog_and_targeted_unload(pool_server):
    _pool, base = pool_server
    status, body = _request(base, "/admin/model-profiles", "PUT", {
        "profiles": [{"model": "org/One", "profile": {
            "mode": "lookup", "context_window": 32768, "kv_bits": 0,
        }}],
    })
    assert status == 200 and body["profiles"][0]["model"] == "org/One"

    status, body = _request(base, "/health")
    assert status == 200 and body["status"] == "no_model" and body["pool"]["models"] == []

    status, body = _request(base, "/v1/models")
    assert status == 200 and {item["id"] for item in body["data"]} == {"org/One", "org/Two"}

    assert _request(base, "/admin/load", "POST", {"model": "One", "keep_loaded": True})[0] == 200
    assert _request(base, "/admin/load", "POST", {"model": "org/Two", "keep_loaded": True})[0] == 200

    status, body = _request(base, "/admin/unload", "POST", {})
    assert status == 400 and body["error"]["code"] == "model_required"
    status, body = _request(base, "/admin/unload", "POST", {"model": "org/One"})
    assert status == 200
    assert [slot["model"] for slot in body["models"] if slot["ready"]] == ["org/Two"]


def test_pool_metrics_acquires_model_by_query_parameter(pool_server):
    _pool, base = pool_server
    status, body = _request(base, "/metrics?model=org/One")
    assert status == 200 and body["model"] == "org/One"
    status, health = _request(base, "/health")
    assert status == 200
    slot = health["pool"]["models"][0]
    assert slot["model"] == "org/One" and slot["leases"] == 0
