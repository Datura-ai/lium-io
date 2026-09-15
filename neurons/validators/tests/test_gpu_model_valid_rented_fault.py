"""P125: a RENTED node that lists fewer GPUs than it advertises is delisted after N cycles in a row.

Regression (F-1361, 15 Sep 2026): node ad83de56 advertised 8x RTX 5090 and enumerated 1 from 07:58Z on;
every cycle scored GPU_DETAILS_MISMATCH / 0 and nothing else happened. The node stayed listed with 8
GPUs, the renter deleted his $988 rental himself 18 min later and the close was filed user_initiated.
On origin/main GpuModelValidCheck never sets ``clear_verified_job_info``, so the second rented mismatch
cycle (the first test) returns the plain GPU_DETAILS_MISMATCH and the node is not delisted.
"""

from __future__ import annotations

import json

import pytest

from core.config import settings
from helpers import build_context_config, build_services, build_state, default_executor
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.task.checks.gpu_model_valid import GpuModelValidCheck

EXECUTOR = default_executor()
MODEL = "NVIDIA GeForce RTX 5090"
KEY = f"rented_gpu_fault:{EXECUTOR.uuid}"


class FakeRedis:
    def __init__(self, *, broken: bool = False):
        self.store: dict[str, str] = {}
        self.broken = broken

    async def get(self, key):
        if self.broken:
            raise ConnectionError("redis down")
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def set(self, key, value, ex=None):
        if self.broken:
            raise ConnectionError("redis down")
        self.store[key] = value
        self.ttl = ex

    async def delete(self, key):
        if self.broken:
            raise ConnectionError("redis down")
        self.store.pop(key, None)


def _rented(pods: int) -> RentedExecutorsResponse:
    executors = {}
    if pods:
        executors[EXECUTOR.uuid] = RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address=EXECUTOR.address,
            executor_ip_port=str(EXECUTOR.port),
            pods=[RentedPod(pod_id=f"p{i}", container_name=f"pod_p{i}") for i in range(pods)],
        )
    return RentedExecutorsResponse(executors=executors)


def _ctx(context_factory, *, advertised: int, seen: int, pods: int, redis: FakeRedis | None):
    details = [{"name": MODEL, "uuid": f"GPU-{i}", "capacity": 32768} for i in range(seen)]
    state = build_state(gpu_count=advertised, gpu_details=details, rented_data=_rented(pods))
    services = build_services(redis=redis)
    config = build_context_config(gpu_model_rates={MODEL: 1.0})
    return context_factory(state=state, services=services, config=config, executor=EXECUTOR)


@pytest.fixture(autouse=True)
def _rule(monkeypatch):
    # these are the defaults; pinned so a .env on the box cannot change them. On origin/main the settings do not exist
    # (pydantic refuses the setattr), and the tests below must then fail on their assertions, not error here
    for name, value in (("RENTED_GPU_FAULT_ENABLED", True), ("RENTED_GPU_FAULT_CYCLES", 2)):
        try:
            monkeypatch.setattr(settings, name, value, raising=False)
        except ValueError:
            pass


@pytest.mark.asyncio
async def test_second_rented_mismatch_cycle_clears_the_verified_job(context_factory):
    redis = FakeRedis()
    first = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
    )
    assert first.passed is False
    assert first.event.reason_code == "GPU_DETAILS_MISMATCH"
    assert "clear_verified_job_info" not in first.updates
    assert json.loads(redis.store[KEY])["count"] == 1
    # the streak dies on its own once the node is no longer scraped: "consecutive" means back-to-back cycles
    assert redis.ttl == 4 * 15 * 60
    first_seen = json.loads(redis.store[KEY])["first_seen_at"]

    second = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
    )
    assert second.passed is False
    assert second.event.reason_code == "RENTED_NODE_GPU_FAULT"
    assert second.event.severity == "error"
    assert second.updates["clear_verified_job_info"] is True
    evidence = second.updates["clear_verified_job_evidence"]
    assert evidence["reason_code"] == "RENTED_NODE_GPU_FAULT"
    assert evidence["check_id"] == "gpu.validate.model"
    assert evidence["gpu_count"] == 8
    assert evidence["details_len"] == 1
    assert evidence["consecutive_cycles"] == 2
    assert evidence["pod_ids"] == ["p0"]
    assert evidence["plain_reason_code"] == "GPU_DETAILS_MISMATCH"
    assert (
        second.event.what_we_saw["first_seen_at"] == first_seen
    )  # carried from cycle 1, not reset


@pytest.mark.asyncio
async def test_zero_gpus_on_a_rented_node_counts_too(context_factory):
    redis = FakeRedis()
    await GpuModelValidCheck().run(_ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis))
    result = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=0, seen=0, pods=1, redis=redis)
    )
    # gpu_count 0 has no model to name, so on main this cycle is GPU_MODEL_UNSUPPORTED and nothing more;
    # here it is the second missing-GPU cycle under a rental
    assert result.passed is False
    assert result.event.reason_code == "RENTED_NODE_GPU_FAULT"
    assert (
        result.updates["clear_verified_job_evidence"]["plain_reason_code"]
        == "GPU_MODEL_UNSUPPORTED"
    )
    assert result.updates["clear_verified_job_evidence"]["consecutive_cycles"] == 2


@pytest.mark.asyncio
async def test_idle_node_mismatch_is_the_plain_failure_and_resets_the_streak(context_factory):
    redis = FakeRedis()
    redis.store[KEY] = json.dumps({"count": 5, "first_seen_at": 1.0})
    result = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=1, pods=0, redis=redis)
    )
    assert result.event.reason_code == "GPU_DETAILS_MISMATCH"
    assert "clear_verified_job_info" not in result.updates
    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_clean_scrape_resets_the_streak(context_factory):
    redis = FakeRedis()
    await GpuModelValidCheck().run(_ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis))
    ok = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=8, pods=1, redis=redis)
    )
    assert ok.passed is True
    assert ok.event.reason_code == "GPU_MODEL_OK"
    assert KEY not in redis.store
    again = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
    )
    assert again.event.reason_code == "GPU_DETAILS_MISMATCH"
    assert "clear_verified_job_info" not in again.updates


@pytest.mark.asyncio
async def test_threshold_is_the_setting(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "RENTED_GPU_FAULT_CYCLES", 3, raising=False)
    redis = FakeRedis()
    codes = []
    for _ in range(3):
        result = await GpuModelValidCheck().run(
            _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
        )
        codes.append(result.event.reason_code)
    assert codes == ["GPU_DETAILS_MISMATCH", "GPU_DETAILS_MISMATCH", "RENTED_NODE_GPU_FAULT"]


@pytest.mark.asyncio
async def test_disabled_rule_is_the_old_behaviour(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "RENTED_GPU_FAULT_ENABLED", False, raising=False)
    redis = FakeRedis()
    for _ in range(3):
        result = await GpuModelValidCheck().run(
            _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
        )
        assert result.event.reason_code == "GPU_DETAILS_MISMATCH"
        assert "clear_verified_job_info" not in result.updates
    assert KEY not in redis.store


@pytest.mark.asyncio
async def test_redis_outage_reports_the_plain_failure_and_never_delists(context_factory):
    redis = FakeRedis(broken=True)
    for _ in range(3):
        result = await GpuModelValidCheck().run(
            _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
        )
        assert result.event.reason_code == "GPU_DETAILS_MISMATCH"
        assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_unreadable_streak_key_starts_over(context_factory):
    redis = FakeRedis()
    redis.store[KEY] = "not json"
    result = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=1, pods=1, redis=redis)
    )
    assert result.event.reason_code == "GPU_DETAILS_MISMATCH"
    assert json.loads(redis.store[KEY])["count"] == 1


@pytest.mark.asyncio
async def test_no_redis_service_is_the_plain_failure(context_factory):
    result = await GpuModelValidCheck().run(
        _ctx(context_factory, advertised=8, seen=1, pods=1, redis=None)
    )
    assert result.event.reason_code == "GPU_DETAILS_MISMATCH"
    assert "clear_verified_job_info" not in result.updates
