"""DAH-2265 Plan 2 — CachedTemplateVerificationCheck (advisory, non-fatal).

Verifies the executor has the recommended default image pre-pulled, publishing the
result to executor.specs (via pipeline state) and a structured event. It must never
fail the pipeline or change score, and must fail open on every uncertainty.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from neurons.validators.src.services.task.checks.cached_template_verification import (
    CachedTemplateVerificationCheck,
)
from neurons.validators.src.services.task.messages import CachedTemplateMessages as Msg

from protocol.vc_protocol.compute_requests import DefaultDockerImage
from tests.helpers import build_services, build_state

_IMAGE = DefaultDockerImage(
    docker_image="daturaai/torch", docker_image_tag="2.4.0", docker_image_size=12_000_000_000
)
_IMAGE_WITH_DIGEST = DefaultDockerImage(
    docker_image="daturaai/torch",
    docker_image_tag="2.4.0",
    docker_image_size=12_000_000_000,
    docker_image_digest="sha256:aaa",
)
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


def test_check_is_non_fatal():
    assert CachedTemplateVerificationCheck.fatal is False


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
    # Branch 4: cached + local RepoDigest == backend digest → match True.
    ssh = _ssh(exit_status=0, stdout=_LOCAL_MATCH)
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE_WITH_DIGEST])),
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
    # Branch 5: cached + local RepoDigest != backend digest → match False (stale content).
    ctx = context_factory(
        services=build_services(backend=_backend(images=[_IMAGE_WITH_DIGEST])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MISMATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_MISMATCH.reason
    assert result.updates["state"].recommended_image_cached is True
    assert result.updates["state"].recommended_image_digest_match is False


@pytest.mark.asyncio
async def test_digest_match_normalizes_repo_at_sha_backend_value(context_factory):
    # M3: a backend digest pasted as "repo@sha256:…" must compare only on the bare sha.
    image = DefaultDockerImage(
        docker_image="daturaai/torch",
        docker_image_tag="2.4.0",
        docker_image_size=12_000_000_000,
        docker_image_digest="daturaai/torch@sha256:aaa",
    )
    ctx = context_factory(
        services=build_services(backend=_backend(images=[image])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout=_LOCAL_MATCH),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.event.reason_code == Msg.DIGEST_MATCH.reason
    assert result.updates["state"].recommended_image_digest_match is True


@pytest.mark.asyncio
async def test_digest_none_when_backend_digest_missing(context_factory):
    # Branch 2: cached, but backend published no digest → match None, cached stays True.
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
        services=build_services(backend=_backend(images=[_IMAGE_WITH_DIGEST])),
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
        services=build_services(backend=_backend(images=[_IMAGE_WITH_DIGEST])),
        state=build_state(gpu_model=_GPU, specs=_SPECS),
        ssh=_ssh(exit_status=0, stdout="not json at all"),
    )

    result = await CachedTemplateVerificationCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DIGEST_SKIPPED.reason
    assert result.updates["state"].recommended_image_digest_match is None
