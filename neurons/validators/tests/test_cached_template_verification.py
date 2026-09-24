"""DAH-2265 — CachedTemplateVerificationCheck.

Verifies the executor has the recommended default image pre-pulled, publishing the
result to executor.specs (via pipeline state) and a structured event. Before
``settings.CACHED_TEMPLATE_CUTOFF`` it is advisory (never fails / changes score); on/after
the cutoff the two bad signals (not cached, stale digest) fail verification. It must fail
open on every uncertainty regardless of the cutoff.

An autouse fixture pins the cutoff to the future by default so the advisory-mode assertions
are wall-clock-independent; cutoff-enforcement tests opt in with ``_set_cutoff(active=True)``.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from core.config import Settings, settings
from neurons.validators.src.services.task.checks.cached_template_verification import (
    CachedTemplateVerificationCheck,
    _remediation,
)
from services.redis_service import RedisService
from neurons.validators.src.services.task.messages import CachedTemplateMessages as Msg

from protocol.vc_protocol.compute_requests import DefaultDockerImage
from tests.helpers import build_context_config, build_services, build_state

_IMAGE = DefaultDockerImage(
    docker_image="daturaai/torch", docker_image_tag="2.4.0", docker_image_size=12_000_000_000
)
_IMAGE_REF = "daturaai/torch:2.4.0"
_IMAGE_DIGEST = "sha256:aaa"
_LOCAL_MATCH = '["daturaai/torch@sha256:aaa"]'
_LOCAL_MISMATCH = '["daturaai/torch@sha256:bbb"]'
_GPU = "NVIDIA H200"
_DRIVER = "580.95.05"
_SPECS = {"gpu": {"driver": _DRIVER}}


def _backend(images=None, raises=False):
    backend = Mock()
    if raises:
        backend.get_default_docker_image = AsyncMock(side_effect=RuntimeError("boom"))
    else:
        backend.get_default_docker_image = AsyncMock(return_value=images)
    return backend


def _ssh(exit_status=0, stdout="", raises=False):
    # stdout is always a real str (M5): the check does `(inspect.stdout or "").strip()`, so a
    # bare Mock attribute would corrupt the parse rather than fail open deterministically.
    ssh = AsyncMock()
    if raises:
        ssh.run = AsyncMock(side_effect=RuntimeError("ssh down"))
    else:
        ssh.run = AsyncMock(return_value=Mock(exit_status=exit_status, stdout=stdout))
    return ssh


def _result(exit_status=0, stdout=""):
    return Mock(exit_status=exit_status, stdout=stdout)


def _ssh_seq(*results):
    """SSH whose successive ``run`` calls return the given results in order.

    DAH-2470 makes a second call on the failure path (the prefetch-state read), so these
    tests need to answer the two calls differently. An Exception instance is raised.
    """
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=list(results))
    return ssh


def _services(backend_images):
    return build_services(backend=_backend(images=backend_images))


def _config_with_digests(digest_map: dict[str, str]):
    # DAH-2380: the validator's Docker Hub digest snapshot now lives on ctx.config
    # (a per-cycle dict), not on ctx.services.
    return build_context_config(default_docker_image_digests=digest_map)


@pytest.fixture(autouse=True)
def _advisory_by_default(monkeypatch):
    # Pin every test to BEFORE the cutoff (advisory mode) so assertions are independent of the
    # wall clock. Cutoff-enforcement tests opt in with _set_cutoff(active=True).
    monkeypatch.setattr(
        settings, "CACHED_TEMPLATE_CUTOFF", datetime.utcnow() + timedelta(days=365)
    )


def _set_cutoff(monkeypatch, *, active: bool) -> None:
    cutoff = datetime.utcnow() - timedelta(days=1) if active else datetime.utcnow() + timedelta(days=1)
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_CUTOFF", cutoff)


def test_check_is_fatal():
    # The gate is live: a passed=False result on/after the cutoff halts the pipeline (score 0).
    assert CachedTemplateVerificationCheck.fatal is True


@pytest.mark.asyncio
async def test_image_cached_publishes_true(context_factory):
    backend = _backend(images=[_IMAGE])
    ssh = _ssh(exit_status=0)
    ctx = context_factory(
        services=build_services(backend=backend),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=ssh,
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CACHED.reason
    assert result.event.what_we_saw["cached"] is True
    assert result.event.what_we_saw["recommended_image"] == "daturaai/torch:2.4.0"
    assert result.updates["state"].recommended_image_cached is True
    # The probe used a read-only, fail-open inspect.
    cmd = ssh.run.await_args.args[0]
    assert "docker image inspect" in cmd
    assert ssh.run.await_args.kwargs["check"] is False
    backend.get_default_docker_image.assert_awaited_once_with(_GPU, _DRIVER)


@pytest.mark.asyncio
async def test_image_not_cached_publishes_false(context_factory):
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=1),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    assert result.event.what_we_saw["cached"] is False
    assert result.updates["state"].recommended_image_cached is False


@pytest.mark.asyncio
async def test_skips_when_gpu_model_missing(context_factory):
    backend = _backend(images=[_IMAGE])
    ctx = context_factory(
        services=build_services(backend=backend),
        state=build_state(gpu_model=None, specs=_SPECS),
        ssh=_ssh(),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert "state" not in result.updates
    backend.get_default_docker_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_driver_missing(context_factory):
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs={"gpu": {}}),
        ssh=_ssh(),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert "state" not in result.updates


@pytest.mark.asyncio
async def test_skips_when_backend_returns_nothing(context_factory):
    ssh = _ssh()
    ctx = context_factory(
        services=build_services(backend=_backend(images=None)),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=ssh,
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert "state" not in result.updates
    # No recommended image → never probe the executor.
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_fails_open_when_backend_raises(context_factory):
    ctx = context_factory(
        services=build_services(backend=_backend(raises=True)),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert "state" not in result.updates


@pytest.mark.asyncio
async def test_fails_open_when_inspect_raises(context_factory):
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(raises=True),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert "state" not in result.updates


# --- DAH-2265 digest verification (advisory, strict fail-open) ----------------------------


@pytest.mark.asyncio
async def test_digest_match_publishes_true(context_factory):
    # Branch 4: cached + local RepoDigest == validator digest cache → match True.
    ssh = _ssh(exit_status=0, stdout=_LOCAL_MATCH)
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=ssh,
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_MATCH.reason
    assert result.updates["state"].recommended_image_cached is True
    assert result.updates["state"].recommended_image_digest_match is True
    # The probe now reads RepoDigests (not just exit status) and stays read-only/fail-open.
    cmd = ssh.run.await_args.args[0]
    assert "--format" in cmd
    assert "RepoDigests" in cmd
    assert ssh.run.await_args.kwargs["check"] is False


@pytest.mark.asyncio
async def test_digest_mismatch_publishes_false(context_factory):
    # Branch 5: cached + local RepoDigest != validator digest cache → match False (stale content).
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MISMATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.updates["state"].recommended_image_cached is True
    assert result.updates["state"].recommended_image_digest_match is False


@pytest.mark.asyncio
async def test_digest_match_uses_validator_digest_cache(context_factory):
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.event.reason_code == Msg.DIGEST_MATCH.reason
    assert result.updates["state"].recommended_image_digest_match is True


@pytest.mark.asyncio
async def test_digest_none_when_validator_digest_missing(context_factory):
    # Branch 2: cached, but the validator digest cache has no entry → match None, cached stays True.
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.event.reason_code == Msg.CACHED.reason
    assert result.updates["state"].recommended_image_cached is True
    assert result.updates["state"].recommended_image_digest_match is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout",
    [
        "[]",  # no RepoDigests (locally built / docker-loaded image)
        '["other/repo@sha256:zzz"]',  # different repo, different sha
        '["other/repo@sha256:aaa"]',  # M5 danger: different repo, SAME sha → must NOT match
    ],
)
async def test_digest_none_when_no_repo_match(context_factory, stdout):
    # Branch 3: cached + backend digest set, but no RepoDigest for THIS repo → match None.
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=stdout),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_SKIPPED.reason
    assert result.updates["state"].recommended_image_cached is True
    assert result.updates["state"].recommended_image_digest_match is None


@pytest.mark.asyncio
async def test_digest_fails_open_on_unparseable_stdout(context_factory):
    # Branch 3 variant: cached + backend digest set, RepoDigests JSON is garbage → match None,
    # never raises.
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout="not json at all"),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_SKIPPED.reason
    assert result.updates["state"].recommended_image_digest_match is None


# --- DAH-2265 cutoff enforcement (critical on/after the cutoff) --------------------------


@pytest.mark.asyncio
async def test_not_cached_fails_after_cutoff(context_factory, monkeypatch):
    # Image not pre-pulled + after cutoff → fatal fail. The reason rides the event (and thus
    # log_text → provider), the cached signal is still published to specs.
    _set_cutoff(monkeypatch, active=True)
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=1),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    assert result.event.severity == "error"
    assert result.updates["state"].recommended_image_cached is False


@pytest.mark.asyncio
async def test_digest_mismatch_fails_after_cutoff(context_factory, monkeypatch):
    # Stale content under the same tag + after cutoff → fatal fail.
    _set_cutoff(monkeypatch, active=True)
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MISMATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.event.severity == "error"
    assert result.updates["state"].recommended_image_cached is True
    assert result.updates["state"].recommended_image_digest_match is False


@pytest.mark.asyncio
async def test_cached_passes_after_cutoff(context_factory, monkeypatch):
    # Pre-pulled, no backend digest to compare → CACHED, never a fatal fail.
    _set_cutoff(monkeypatch, active=True)
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.CACHED.reason


@pytest.mark.asyncio
async def test_digest_match_passes_after_cutoff(context_factory, monkeypatch):
    _set_cutoff(monkeypatch, active=True)
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_MATCH.reason


@pytest.mark.asyncio
async def test_uncertainty_fails_open_after_cutoff(context_factory, monkeypatch):
    # No recommended image from backend → SKIPPED, stays advisory even after the cutoff.
    _set_cutoff(monkeypatch, active=True)
    ssh = _ssh()
    ctx = context_factory(
        services=build_services(backend=_backend(images=None)),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=ssh,
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_digest_skipped_passes_after_cutoff(context_factory, monkeypatch):
    # Cached but RepoDigest unreadable for this repo → DIGEST_SKIPPED, never a fatal fail.
    _set_cutoff(monkeypatch, active=True)
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout="[]"),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_SKIPPED.reason


# --- DAH-2470 prefetch-state passthrough --------------------------------------------------
#
# When this check zeroes a node it also carries the executor's own record of what its
# cache-prefetch loop was doing, so a reader can tell a provider-side cause from ours. The
# read happens on the failure path only, and can never change the verdict.

_PREFETCH_STATE = json.dumps(
    {
        "schema_version": 1,
        "executor_version": "4.0.3",
        "sweep_count": 208,
        "images": {
            _IMAGE_REF: {
                "last_outcome": "remote_digest_unreadable",
                "last_error": "429 Client Error: Too Many Requests",
                "outcome_counts": {"remote_digest_unreadable": 208},
            }
        },
    }
)


def _failing_ctx(context_factory, monkeypatch, ssh):
    """Context that lands on DIGEST_MISMATCH after the cutoff — i.e. the node is zeroed."""
    _set_cutoff(monkeypatch, active=True)
    return context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=ssh,
    )


@pytest.mark.asyncio
async def test_prefetch_state_attached_on_failure(context_factory, monkeypatch):
    ssh = _ssh_seq(_result(stdout=_LOCAL_MISMATCH), _result(stdout=_PREFETCH_STATE))
    ctx = _failing_ctx(context_factory, monkeypatch, ssh)

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    state = result.event.what_we_saw["prefetch_state"]
    assert state["sweep_count"] == 208
    assert state["images"][_IMAGE_REF]["last_outcome"] == "remote_digest_unreadable"
    # Read from the executor container, over the connection the check already holds.
    assert "/var/lib/lium/cache_prefetch_state.json" in ssh.run.await_args.args[0]
    assert ssh.run.await_args.kwargs["check"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state_result", "expected"),
    [
        (_result(exit_status=1), "missing"),
        (_result(stdout="   "), "empty"),
        (_result(stdout="{not json"), "unparseable"),
        (_result(stdout='"a string, not an object"'), "unparseable"),
        (_result(stdout="x" * 9000), "oversized"),
        (RuntimeError("connection closed"), "ssh_read_failed"),
    ],
)
async def test_prefetch_state_absence_is_itself_a_finding(
    context_factory, monkeypatch, state_result, expected
):
    # Each way the read can fall short must be distinguishable: an executor too old to
    # write the file reads differently from one whose loop crashed.
    ssh = _ssh_seq(_result(stdout=_LOCAL_MISMATCH), state_result)
    ctx = _failing_ctx(context_factory, monkeypatch, ssh)

    result = await CachedTemplateVerificationCheck().run(ctx)

    # The verdict is decided before the read and is never touched by it.
    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.event.what_we_saw["prefetch_state"]["unavailable"] == expected


@pytest.mark.asyncio
async def test_healthy_node_issues_no_extra_ssh_call(context_factory, monkeypatch):
    # Roughly three quarters of checked nodes pass. None of them may pay for this.
    _set_cutoff(monkeypatch, active=True)
    ssh = _ssh_seq(_result(stdout=_LOCAL_MATCH))
    ctx = context_factory(
        services=_services([_IMAGE]),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=ssh,
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert ssh.run.await_count == 1
    assert "prefetch_state" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_not_cached_stays_advisory_before_cutoff(context_factory, monkeypatch):
    # Same bad signal as the fatal test, but before the cutoff → advisory pass, info severity.
    _set_cutoff(monkeypatch, active=False)
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=1),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    assert result.event.severity != "error"
    assert result.updates["state"].recommended_image_cached is False


# --- fresh-node grace and the remediation quote ------------------------------------------
#
# A new node was verified 5 s after it was added, while its executor's first pinned pull of a
# multi-GB image had just failed and the loop was backing off; it was zeroed on NOT_CACHED and
# passed a batch later. A node this validator has just found without the image is held as
# PENDING until the executor's first sweep completes or the grace ends; after that it fails as
# before, and the remediation quotes the executor's own pull error.

_PULL_ERROR = "Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout"


def _prefetch_doc(*, first_sweep_ok_at=None, pull_error=_PULL_ERROR, last_outcome="loop_error"):
    return json.dumps(
        {
            "schema_version": 1,
            "sweep_count": 1,
            "last_outcome": last_outcome,
            "outcome_counts": {last_outcome: 1},
            "first_sweep_ok_at": first_sweep_ok_at,
            "images": {
                _IMAGE_REF: {
                    "last_outcome": "pull_failed" if pull_error else "up_to_date",
                    "last_pull_error": pull_error,
                }
            },
        }
    )


def _fake_redis_service():
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    return service


def _uncached_ctx(
    context_factory, monkeypatch, redis, prefetch_doc, *, digests=None, grace_enabled=True
):
    _set_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED", grace_enabled)
    return context_factory(
        services=build_services(backend=_backend(images=[_IMAGE]), redis=redis),
        config=_config_with_digests(digests or {}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh_seq(_result(exit_status=1), _result(stdout=prefetch_doc)),
    )


@pytest.mark.asyncio
async def test_fresh_node_is_pending_inside_the_grace(context_factory, monkeypatch):
    redis = _fake_redis_service()
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, _prefetch_doc())

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PENDING.reason
    assert result.event.severity == "info"
    grace = result.event.what_we_saw["fresh_node_grace"]
    assert grace["pending"] is True
    assert grace["first_sweep_completed"] is False
    assert grace["seconds_since_first_uncached"] < 5
    # Still published as not cached: the grace changes the verdict, not the observation.
    assert result.updates["state"].recommended_image_cached is False


@pytest.mark.asyncio
async def test_uncached_node_fails_once_the_grace_has_passed(context_factory, monkeypatch):
    redis = _fake_redis_service()
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, _prefetch_doc())
    first_seen = time.time() - settings.CACHED_TEMPLATE_FRESH_NODE_GRACE_SECONDS - 60
    await redis.redis.set(f"cached_template_first_uncached:{ctx.executor.uuid}", repr(first_seen))

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    assert result.event.severity == "error"
    assert result.event.what_we_saw["fresh_node_grace"]["pending"] is False


@pytest.mark.asyncio
async def test_grace_window_is_kept_from_the_first_sighting(context_factory, monkeypatch):
    # A second cycle inside the window reads the first sighting, not its own clock: an executor
    # cannot move the window, and a node that stays uncached is failed once it closes.
    redis = _fake_redis_service()
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, _prefetch_doc())
    first = await redis.first_uncached_at(ctx.executor.uuid, 1_000.0, 3600)
    again = await redis.first_uncached_at(ctx.executor.uuid, 2_000.0, 3600)

    assert first == again == 1_000.0


@pytest.mark.asyncio
async def test_fresh_node_fails_once_its_first_sweep_completed(context_factory, monkeypatch):
    # The executor finished a sweep and the image is still missing: nothing is in flight, so the
    # grace has nothing to wait for.
    redis = _fake_redis_service()
    doc = _prefetch_doc(first_sweep_ok_at="2026-09-20T10:00:00Z", last_outcome="sweep_ok")
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, doc)

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    grace = result.event.what_we_saw["fresh_node_grace"]
    assert grace == {**grace, "pending": False, "first_sweep_completed": True}


@pytest.mark.asyncio
async def test_older_executor_counts_a_sweep_ok_as_the_first_sweep(context_factory, monkeypatch):
    redis = _fake_redis_service()
    doc = json.loads(_prefetch_doc(last_outcome="sweep_ok"))
    del doc["first_sweep_ok_at"]
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, json.dumps(doc))

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["fresh_node_grace"]["first_sweep_completed"] is True


@pytest.mark.asyncio
async def test_redis_error_means_no_grace(context_factory, monkeypatch):
    redis = Mock()
    redis.first_uncached_at = AsyncMock(side_effect=RuntimeError("redis down"))
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, _prefetch_doc())

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    assert "redis down" in result.event.what_we_saw["fresh_node_grace"]["error"]


@pytest.mark.asyncio
async def test_grace_disabled_fails_a_fresh_node(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_SECONDS", 0)
    ctx = _uncached_ctx(context_factory, monkeypatch, _fake_redis_service(), _prefetch_doc())

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert "fresh_node_grace" not in result.event.what_we_saw


def test_fresh_node_grace_flag_is_off_by_default():
    assert Settings.model_fields["CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED"].default is False


@pytest.mark.asyncio
async def test_grace_flag_off_fails_a_fresh_node_as_before_and_logs_the_hold(
    context_factory, monkeypatch, caplog
):
    ctx = _uncached_ctx(
        context_factory, monkeypatch, _fake_redis_service(), _prefetch_doc(), grace_enabled=False
    )

    with caplog.at_level(logging.INFO):
        result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.NOT_CACHED.reason
    assert result.event.severity == "error"
    assert set(result.event.what_we_saw) == {
        "recommended_image",
        "cached",
        "digest_match",
        "backend_digest",
        "local_digest",
        "gpu_model",
        "driver_version",
        "after_cutoff",
        "prefetch_state",
    }
    assert any("would have been held as pending" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_grace_flag_off_logs_nothing_for_a_node_past_its_grace(
    context_factory, monkeypatch, caplog
):
    redis = _fake_redis_service()
    ctx = _uncached_ctx(context_factory, monkeypatch, redis, _prefetch_doc(), grace_enabled=False)
    first_seen = time.time() - settings.CACHED_TEMPLATE_FRESH_NODE_GRACE_SECONDS - 60
    await redis.redis.set(f"cached_template_first_uncached:{ctx.executor.uuid}", repr(first_seen))

    with caplog.at_level(logging.INFO):
        result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert not any("would have been held" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_pending_without_a_prefetch_document_says_only_the_time_bound_applies(
    context_factory, monkeypatch
):
    ctx = _uncached_ctx(context_factory, monkeypatch, _fake_redis_service(), "")

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.event.reason_code == Msg.PENDING.reason
    assert result.event.what_we_saw["fresh_node_grace"]["first_sweep_completed"] is None
    assert "may still be fetching" in result.event.remediation
    assert "does not report its pre-pull state" in result.event.remediation


@pytest.mark.asyncio
async def test_not_cached_with_no_named_cause_names_the_image_to_pull(context_factory, monkeypatch):
    doc = _prefetch_doc(
        first_sweep_ok_at="2026-09-20T10:00:00Z", pull_error=None, last_outcome="sweep_ok"
    )
    ctx = _uncached_ctx(context_factory, monkeypatch, _fake_redis_service(), doc)

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert f"docker pull {_IMAGE_REF}`" in result.event.remediation
    assert "<image>" not in result.event.remediation


@pytest.mark.asyncio
async def test_not_cached_with_no_named_cause_names_the_pinned_image_to_pull(
    context_factory, monkeypatch
):
    doc = _prefetch_doc(
        first_sweep_ok_at="2026-09-20T10:00:00Z", pull_error=None, last_outcome="sweep_ok"
    )
    ctx = _uncached_ctx(
        context_factory,
        monkeypatch,
        _fake_redis_service(),
        doc,
        digests={_IMAGE_REF: _IMAGE_DIGEST},
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert "docker pull daturaai/torch@sha256:aaa`" in result.event.remediation


def _mismatch_ctx(context_factory, monkeypatch, prefetch_read):
    _set_cutoff(monkeypatch, active=True)
    return context_factory(
        services=build_services(backend=_backend(images=[_IMAGE]), redis=_fake_redis_service()),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh_seq(_result(stdout=_LOCAL_MISMATCH), prefetch_read),
    )


@pytest.mark.asyncio
async def test_digest_mismatch_with_no_named_cause_keeps_its_own_remediation(
    context_factory, monkeypatch
):
    doc = _prefetch_doc(
        first_sweep_ok_at="2026-09-20T10:00:00Z", pull_error=None, last_outcome="sweep_ok"
    )
    ctx = _mismatch_ctx(context_factory, monkeypatch, _result(stdout=doc))

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.event.remediation == Msg.DIGEST_MISMATCH.remediation


@pytest.mark.asyncio
async def test_digest_mismatch_without_a_prefetch_document_calls_the_image_stale(
    context_factory, monkeypatch
):
    ctx = _mismatch_ctx(context_factory, monkeypatch, _result(exit_status=1))

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert "to see why the image on the host is stale." in result.event.remediation
    assert "why the image is missing" not in result.event.remediation


def test_cached_template_messages_carry_no_placeholder():
    for template in (Msg.NOT_CACHED, Msg.PENDING, Msg.DIGEST_MISMATCH):
        assert "<" not in (template.remediation or ""), template.reason


@pytest.mark.asyncio
async def test_digest_mismatch_gets_no_fresh_node_grace(context_factory, monkeypatch):
    redis = _fake_redis_service()
    _set_cutoff(monkeypatch, active=True)
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED", True)
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE]), redis=redis),
        config=_config_with_digests({_IMAGE_REF: _IMAGE_DIGEST}),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh_seq(_result(stdout=_LOCAL_MISMATCH), _result(stdout=_prefetch_doc())),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert "fresh_node_grace" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_remediation_quotes_the_executors_pull_error(context_factory, monkeypatch):
    redis = _fake_redis_service()
    doc = _prefetch_doc(first_sweep_ok_at="2026-09-20T10:00:00Z")
    ctx = _uncached_ctx(
        context_factory, monkeypatch, redis, doc, digests={_IMAGE_REF: _IMAGE_DIGEST}
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    remediation = result.event.remediation
    assert _PULL_ERROR in remediation
    assert "registry-1.docker.io and production.cloudflare.docker.com" in remediation
    assert "cache_template_service" not in remediation


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            '404 Client Error: Not Found ("No such image: daturaai/torch@sha256:aaa")',
            "update the executor",
        ),
        ("toomanyrequests: You have reached your pull rate limit", "docker login"),
        ("write /var/lib/docker/tmp/x: no space left on device", "free disk"),
        ("manifest for daturaai/torch@sha256:aaa not found: manifest unknown", "registry mirror"),
        ("unauthorized: authentication required", "Docker login"),
        ("something new", "docker pull daturaai/torch@sha256:aaa"),
    ],
)
def test_remediation_names_the_next_step_for_the_error(error, expected):
    state = json.loads(_prefetch_doc(pull_error=error))

    text = _remediation(state, _IMAGE_REF, "daturaai/torch@sha256:aaa", cached=False)

    assert error[:40] in text
    assert expected in text


def test_remediation_does_not_quote_an_error_a_later_sweep_moved_past():
    state = {"last_outcome": "sweep_ok", "images": {_IMAGE_REF: {
        "last_outcome": "up_to_date", "last_pull_error": _PULL_ERROR,
    }}}

    assert _remediation(state, _IMAGE_REF, _IMAGE_REF, cached=False) is None


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"unavailable": "missing"}, "published no pre-pull state (missing)"),
        (
            {"last_outcome": "prefetch_disabled_no_backend_url", "images": {}},
            "COMPUTE_REST_API_URL",
        ),
        (
            {"last_outcome": "sweep_ok", "images": {_IMAGE_REF: {
                "last_outcome": "insufficient_disk",
                "last_disk_required_bytes": 90 * 1024**3,
                "last_disk_available_bytes": 20 * 1024**3,
            }}},
            "needs 90 GiB free and has 20 GiB",
        ),
        (
            {"last_outcome": "sweep_ok", "images": {_IMAGE_REF: {
                "last_outcome": "pull_ok", "last_pull_ok_at": "2026-09-20T10:00:00Z",
            }}},
            "no longer on the host",
        ),
    ],
)
def test_remediation_reads_the_loop_and_disk_outcomes(state, expected):
    assert expected in _remediation(state, _IMAGE_REF, _IMAGE_REF, cached=False)


def test_remediation_without_a_prefetch_document_says_why_the_image_fails():
    missing = _remediation({"unavailable": "empty"}, _IMAGE_REF, _IMAGE_REF, cached=False)
    stale = _remediation({"unavailable": "empty"}, _IMAGE_REF, _IMAGE_REF, cached=True)

    assert missing.endswith("to see why the image is missing.")
    assert stale.endswith("to see why the image on the host is stale.")
