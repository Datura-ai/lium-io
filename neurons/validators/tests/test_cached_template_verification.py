"""DAH-2265 — CachedTemplateVerificationCheck.

Verifies the executor has the recommended default image pre-pulled, publishing the
result to executor.specs (via pipeline state) and a structured event. Before
``settings.CACHED_TEMPLATE_CUTOFF`` it is advisory (never fails / changes score); on/after
the cutoff the two bad signals (not cached, stale digest) fail verification. It must fail
open on every uncertainty regardless of the cutoff.

An autouse fixture pins the cutoff to the future by default so the advisory-mode assertions
are wall-clock-independent; cutoff-enforcement tests opt in with ``_set_cutoff(active=True)``.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from core.config import settings
from neurons.validators.src.services.task.checks.cached_template_verification import (
    CachedTemplateVerificationCheck,
)
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
    # DAH-2265: the check is fatal-capable; run() only returns passed=False after the cutoff.
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
