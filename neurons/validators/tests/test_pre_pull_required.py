"""DAH-2977 — pre-pull requirement on the unrented share, behind PRE_PULL_REQUIRED_CUTOFF.

The multiplier is 1 - PORTION_FOR_PRE_PULL_UNRENTED * missing_past_grace / expected. It is 1.0
with no cutoff (the default), before the cutoff, on a rented node, with nothing measured, and while
every missing image is inside its grace window. The grace window starts per node and per
repo@digest at the first cycle that sees the image missing, and ends when the node holds it.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive import rental_price as rental_price_module
from incentive.config import IncentiveConfig
from incentive.default import get_pre_pull_multiplier
from incentive.rental_price import RentalPriceIncentive
from neurons.validators.src.services.task.checks import pre_pull_cached as check_module
from neurons.validators.src.services.task.checks.pre_pull_cached import PrePullCachedCheck
from protocol.vc_protocol.compute_requests import DefaultDockerImage
from services.redis_service import RedisService
from services.task_service import JobResult
from tests.helpers import build_services, build_state

HOUR = 3600
_CU128 = DefaultDockerImage(
    docker_image="daturaai/pytorch",
    docker_image_tag="2.11.0-py3.12-cuda12.8-devel-ubuntu24.04-dind-lium1",
    docker_image_digest="sha256:" + "a" * 64,
    pre_pull=True,
)
_CUDA = DefaultDockerImage(
    docker_image="nvidia/cuda",
    docker_image_tag="13.0.3-devel-ubuntu22.04",
    docker_image_digest="sha256:" + "b" * 64,
    pre_pull=True,
)
_DEFAULT = DefaultDockerImage(docker_image="daturaai/pytorch", docker_image_tag="default-lium1")


def _cutoff(monkeypatch, *, active: bool | None) -> None:
    if active is None:
        value = None
    else:
        value = datetime.utcnow() - timedelta(days=1) if active else datetime.utcnow() + timedelta(days=1)
    monkeypatch.setattr(settings, "PRE_PULL_REQUIRED_CUTOFF", value)


def _report(expected: int, past_grace: list[str], missing: list[str] | None = None) -> dict:
    missing = past_grace if missing is None else missing
    return {"expected": expected, "cached": expected - len(missing), "missing": missing, "missing_past_grace": past_grace}


# --- the multiplier -----------------------------------------------------------------------


def test_defaults():
    assert settings.PRE_PULL_REQUIRED_CUTOFF is None
    assert settings.PRE_PULL_REQUIRED_GRACE_SECONDS == 6 * HOUR
    assert settings.PORTION_FOR_PRE_PULL_UNRENTED == 0.1


def test_no_cutoff_never_scales(monkeypatch):
    _cutoff(monkeypatch, active=None)
    assert get_pre_pull_multiplier(_report(2, ["a", "b"])) == 1.0


def test_before_cutoff_never_scales(monkeypatch):
    _cutoff(monkeypatch, active=False)
    assert get_pre_pull_multiplier(_report(2, ["a", "b"])) == 1.0


@pytest.mark.parametrize(
    "report,expected",
    [
        (_report(2, []), 1.0),
        (_report(2, [], missing=["a"]), 1.0),  # missing, still inside the grace window
        (_report(2, ["a"]), 0.95),
        (_report(2, ["a", "b"]), 0.9),
        (_report(0, []), 1.0),
        (None, 1.0),
        ({}, 1.0),
        ({"expected": "x", "missing_past_grace": ["a"]}, 1.0),
    ],
    ids=["all-cached", "in-grace", "one-of-two", "two-of-two", "none-served", "not-measured", "empty", "junk"],
)
def test_after_cutoff(monkeypatch, report, expected):
    _cutoff(monkeypatch, active=True)
    assert get_pre_pull_multiplier(report) == pytest.approx(expected)


def test_rented_is_exempt(monkeypatch):
    _cutoff(monkeypatch, active=True)
    assert get_pre_pull_multiplier(_report(2, ["a", "b"]), is_rented=True) == 1.0


def test_portion_is_the_ceiling(monkeypatch):
    _cutoff(monkeypatch, active=True)
    monkeypatch.setattr(settings, "PORTION_FOR_PRE_PULL_UNRENTED", 1.0)
    assert get_pre_pull_multiplier(_report(2, ["a", "b"])) == 0.0
    # more refs past grace than served cannot push it below 1 - portion
    monkeypatch.setattr(settings, "PORTION_FOR_PRE_PULL_UNRENTED", 0.1)
    assert get_pre_pull_multiplier(_report(1, ["a", "b"])) == pytest.approx(0.9)


# --- through the unrented share -----------------------------------------------------------


def _job(pre_pull_images: dict | None, is_rented: bool = False) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="executor-pre-pull-test",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="pre-pull-test-batch",
        log_status="success",
        log_text="ok",
        gpu_model="H100",
        gpu_count=1,
        is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=True,
        pre_pull_images=pre_pull_images,
    )


def _incentive(job_results: dict[str, list[JobResult]]) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    incentive = RentalPriceIncentive(
        IncentiveConfig(
            algorithm="rental_price",
            rental_incentive_gpu_types=["H100"],
            max_unrented_gpus={"H100": {1: 1}},
            rental_prices_per_hour={"H100": 4.0},
            gpu_count_custom_prices={"H100": {"1": 4.0}},
        ),
        redis,
        job_results,
        total_gpu_model_count_map={"H100": 1},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 1.0
    price_provider.get_alpha_rate.return_value = 1.0
    incentive.price_provider = price_provider
    return incentive


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "active,report,rate",
    [
        (None, _report(2, ["a", "b"]), 4.0),
        (False, _report(2, ["a", "b"]), 4.0),
        (True, _report(2, []), 4.0),
        (True, _report(2, ["a"]), 3.8),
        (True, _report(2, ["a", "b"]), 3.6),
        (True, None, 4.0),
    ],
    ids=["no-cutoff", "before-cutoff", "all-cached", "one-missing", "both-missing", "not-measured"],
)
async def test_effective_rate(monkeypatch, active, report, rate):
    _cutoff(monkeypatch, active=active)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", {"H100": "H100"})
    job = _job(report)
    incentive = _incentive({"miner": [job]})

    await incentive.calculate_mining_scores()

    assert job.effective_rate == pytest.approx(rate)
    assert job.pre_pull_multiplier == pytest.approx(rate / 4.0)
    assert job.incentive > 0


# --- the grace clock ----------------------------------------------------------------------


class FakeRedis:
    """RedisService's two pre-pull calls over a dict, with a settable clock."""

    def __init__(self):
        self.first: dict[tuple[str, str], float] = {}

    async def pre_pull_missing_since(self, executor_id, pinned_ref, now, ttl_seconds):
        return self.first.setdefault((executor_id, pinned_ref), now)

    async def clear_pre_pull_missing(self, executor_id, pinned_ref):
        self.first.pop((executor_id, pinned_ref), None)


def _run(context_factory, redis, stdout: str) -> dict:
    backend = Mock()
    backend.get_default_docker_image = AsyncMock(return_value=[_DEFAULT, _CU128, _CUDA])
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=Mock(exit_status=0, stdout=stdout))
    ctx = context_factory(
        services=build_services(backend=backend, redis=redis),
        state=build_state(gpu_model="NVIDIA H100 80GB HBM3", specs={"gpu": {"driver": "580.178.04"}}),
        ssh=ssh,
    )
    return asyncio.run(PrePullCachedCheck().run(ctx)).updates["state"].pre_pull_images


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1_800_000_000.0}
    monkeypatch.setattr(check_module.time, "time", lambda: now["t"])
    return now


def test_grace_starts_at_the_first_miss_and_ends_with_the_image(context_factory, clock):
    redis = FakeRedis()

    assert _run(context_factory, redis, "1\n0\n")["missing_past_grace"] == []
    clock["t"] += 6 * HOUR - 60
    assert _run(context_factory, redis, "1\n0\n")["missing_past_grace"] == []
    clock["t"] += 120
    assert _run(context_factory, redis, "1\n0\n")["missing_past_grace"] == [_CU128.image_ref]

    # the node pulls it: the record goes, and a later miss gets a full window again
    assert _run(context_factory, redis, "0\n0\n")["missing_past_grace"] == []
    assert redis.first == {}
    assert _run(context_factory, redis, "1\n0\n")["missing_past_grace"] == []


def test_a_new_digest_gets_its_own_window(context_factory, clock):
    redis = FakeRedis()
    _run(context_factory, redis, "1\n0\n")
    clock["t"] += 7 * HOUR
    assert _run(context_factory, redis, "1\n0\n")["missing_past_grace"] == [_CU128.image_ref]

    rebuilt = _CU128.model_copy(update={"docker_image_digest": "sha256:" + "e" * 64})
    backend = Mock()
    backend.get_default_docker_image = AsyncMock(return_value=[_DEFAULT, rebuilt, _CUDA])
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=Mock(exit_status=0, stdout="1\n0\n"))
    ctx = context_factory(
        services=build_services(backend=backend, redis=redis),
        state=build_state(gpu_model="NVIDIA H100 80GB HBM3", specs={"gpu": {"driver": "580.178.04"}}),
        ssh=ssh,
    )
    result = asyncio.run(PrePullCachedCheck().run(ctx))
    assert result.updates["state"].pre_pull_images["missing_past_grace"] == []


def test_redis_error_fails_open(context_factory, clock):
    redis = FakeRedis()
    redis.pre_pull_missing_since = AsyncMock(side_effect=RuntimeError("redis down"))
    report = _run(context_factory, redis, "1\n1\n")
    assert report["missing"] == [_CU128.image_ref, _CUDA.image_ref]
    assert report["missing_past_grace"] == []


# --- RedisService ---------------------------------------------------------------------------


class _AioRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return None
        self.data[key] = value
        self.ttl[key] = ex
        return True

    async def expire(self, key, seconds):
        self.ttl[key] = seconds

    async def get(self, key):
        value = self.data.get(key)
        return value.encode() if value is not None else None

    async def delete(self, key):
        self.data.pop(key, None)


def _service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = _AioRedis()
    service.lock = asyncio.Lock()
    return service


def test_redis_keeps_the_first_sighting_and_renews_the_ttl():
    service = _service()

    async def go():
        first = await service.pre_pull_missing_since("ex-1", "repo@sha256:a", 100.0, ttl_seconds=10)
        again = await service.pre_pull_missing_since("ex-1", "repo@sha256:a", 500.0, ttl_seconds=20)
        other = await service.pre_pull_missing_since("ex-2", "repo@sha256:a", 700.0, ttl_seconds=20)
        await service.clear_pre_pull_missing("ex-1", "repo@sha256:a")
        fresh = await service.pre_pull_missing_since("ex-1", "repo@sha256:a", 900.0, ttl_seconds=20)
        return first, again, other, fresh

    first, again, other, fresh = asyncio.run(go())
    assert (first, again, other, fresh) == (100.0, 100.0, 700.0, 900.0)
    assert service.redis.ttl["pre_pull_missing:ex-1:repo@sha256:a"] == 20
