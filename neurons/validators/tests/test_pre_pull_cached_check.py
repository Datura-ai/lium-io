"""DAH-2977 — PrePullCachedCheck.

Advisory: probes every pre-pull entry the backend serves the node (digest-pinned) with
`docker image inspect repo@digest`, publishes {"expected", "cached", "missing"} into
executor.specs["pre_pull_images"] through pipeline state, and never fails or changes score.
Fails open (skip event, nothing published) on every uncertainty.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlparse

import pytest

from core.config import settings
from neurons.validators.src.services.task.checks.pre_pull_cached import PrePullCachedCheck
from neurons.validators.src.services.task.messages import PrePullCachedMessages as Msg
from neurons.validators.src.services.task.result_handler import ResultHandler

from protocol.vc_protocol.compute_requests import DefaultDockerImage, DefaultDockerImagesResponse
from tests.helpers import build_services, build_state

_GPU = "NVIDIA H100 80GB HBM3"
_DRIVER = "580.178.04"
_SPECS = {"gpu": {"driver": _DRIVER}}

_DEFAULT = DefaultDockerImage(
    docker_image="daturaai/pytorch",
    docker_image_tag="2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind-lium1",
    docker_image_size=9_133_782_473,
)
_CU128 = DefaultDockerImage(
    docker_image="daturaai/pytorch",
    docker_image_tag="2.11.0-py3.12-cuda12.8-devel-ubuntu24.04-dind-lium1",
    docker_image_size=6_491_891_139,
    docker_image_digest="sha256:" + "a" * 64,
    pre_pull=True,
)
_CUDA = DefaultDockerImage(
    docker_image="nvidia/cuda",
    docker_image_tag="13.0.3-devel-ubuntu22.04",
    docker_image_size=3_961_262_240,
    docker_image_digest="sha256:" + "b" * 64,
    pre_pull=True,
)


def _backend(images=None, raises=False):
    backend = Mock()
    if raises:
        backend.get_default_docker_image = AsyncMock(side_effect=RuntimeError("boom"))
    else:
        backend.get_default_docker_image = AsyncMock(return_value=images)
    return backend


def _ssh(stdout="", raises=False):
    ssh = AsyncMock()
    if raises:
        ssh.run = AsyncMock(side_effect=RuntimeError("ssh down"))
    else:
        ssh.run = AsyncMock(return_value=Mock(exit_status=0, stdout=stdout))
    return ssh


def _ctx(context_factory, *, images, ssh, specs=_SPECS, gpu_model=_GPU):
    backend = images if isinstance(images, Mock) else _backend(images=images)
    return context_factory(
        services=build_services(backend=backend),
        state=build_state(gpu_model=gpu_model, specs=specs),
        ssh=ssh,
    )


def test_check_is_never_fatal():
    assert PrePullCachedCheck.fatal is False


def test_default_on():
    assert settings.PRE_PULL_CACHED_CHECK_ENABLED is True


@pytest.mark.asyncio
async def test_all_cached(context_factory):
    backend = _backend(images=[_DEFAULT, _CU128, _CUDA])
    ssh = _ssh(stdout="0\n0\n")
    ctx = _ctx(context_factory, images=backend, ssh=ssh)

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.ALL_CACHED.reason
    assert result.updates["state"].pre_pull_images == {
        "expected": 2,
        "cached": 2,
        "missing": [],
        "missing_past_grace": [],
    }
    backend.get_default_docker_image.assert_awaited_once_with(_GPU, _DRIVER, include_pre_pull=True)
    # One round trip; each probe is pinned to the digest the executor pulls, never the tag,
    # and the default image (no pre_pull) is not probed here.
    cmd = ssh.run.await_args.args[0]
    assert ssh.run.await_count == 1
    assert ssh.run.await_args.kwargs["check"] is False
    assert f"daturaai/pytorch@sha256:{'a' * 64}" in cmd
    assert f"nvidia/cuda@sha256:{'b' * 64}" in cmd
    assert _DEFAULT.docker_image_tag not in cmd


@pytest.mark.asyncio
async def test_one_missing_is_advisory(context_factory):
    ctx = _ctx(context_factory, images=[_DEFAULT, _CU128, _CUDA], ssh=_ssh(stdout="1\n0\n"))

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert result.event.severity == "info"
    assert result.updates["state"].pre_pull_images == {
        "expected": 2,
        "cached": 1,
        "missing": [_CU128.image_ref],
        # no Redis in these tests: the grace clock fails open, nothing counts against the node
        "missing_past_grace": [],
    }
    assert result.event.what_we_saw["cached_refs"] == [_CUDA.image_ref]


@pytest.mark.asyncio
async def test_entry_without_digest_is_not_probed(context_factory):
    undigested = _CUDA.model_copy(update={"docker_image_digest": None})
    ssh = _ssh(stdout="0\n")
    ctx = _ctx(context_factory, images=[_DEFAULT, _CU128, undigested], ssh=ssh)

    result = await PrePullCachedCheck().run(ctx)

    assert result.updates["state"].pre_pull_images["expected"] == 1
    assert "nvidia/cuda" not in ssh.run.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "images",
    [None, [], [_DEFAULT]],
    ids=["backend-none", "empty", "default-only"],
)
async def test_no_pre_pull_entries_skips(context_factory, images):
    ssh = _ssh()
    ctx = _ctx(context_factory, images=images, ssh=ssh)

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_backend_raises_skips(context_factory):
    ctx = _ctx(context_factory, images=_backend(raises=True), ssh=_ssh())

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_ssh_error_skips(context_factory):
    ctx = _ctx(context_factory, images=[_DEFAULT, _CU128], ssh=_ssh(raises=True))

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout", ["", "0\n", "0\n0\n0\n", "x\n0\n"], ids=["empty", "short", "long", "junk"])
async def test_unparseable_output_skips(context_factory, stdout):
    ctx = _ctx(context_factory, images=[_DEFAULT, _CU128, _CUDA], ssh=_ssh(stdout=stdout))

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gpu_model,specs",
    [(None, _SPECS), (_GPU, {"gpu": {}}), (_GPU, None)],
    ids=["no-gpu-model", "no-driver", "no-specs"],
)
async def test_missing_gpu_or_driver_skips(context_factory, gpu_model, specs):
    backend = _backend(images=[_DEFAULT, _CU128])
    ctx = _ctx(context_factory, images=backend, ssh=_ssh(), gpu_model=gpu_model, specs=specs)

    result = await PrePullCachedCheck().run(ctx)

    assert result.event.reason_code == Msg.SKIPPED.reason
    backend.get_default_docker_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_off_skips_without_io(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "PRE_PULL_CACHED_CHECK_ENABLED", False)
    backend = _backend(images=[_DEFAULT, _CU128])
    ssh = _ssh(stdout="0\n")
    ctx = _ctx(context_factory, images=backend, ssh=ssh)

    result = await PrePullCachedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    backend.get_default_docker_image.assert_not_awaited()
    ssh.run.assert_not_awaited()


def test_backend_entries_parse_pre_pull_flag():
    parsed = DefaultDockerImagesResponse.model_validate(
        [
            {"docker_image": "daturaai/pytorch", "docker_image_tag": "a", "docker_image_size": 1},
            {
                "docker_image": "nvidia/cuda",
                "docker_image_tag": "b",
                "docker_image_size": 2,
                "docker_image_digest": "sha256:" + "c" * 64,
                "pre_pull": True,
            },
        ]
    ).root
    assert [image.pre_pull for image in parsed] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_pre_pull,expected", [(False, None), (True, ["true"])])
async def test_backend_client_query(include_pre_pull, expected):
    from clients.backend_client import BackendClient

    client = BackendClient.__new__(BackendClient)
    client.get = AsyncMock(return_value=None)

    await client.get_default_docker_image(_GPU, _DRIVER, include_pre_pull=include_pre_pull)

    query = parse_qs(urlparse(client.get.await_args.args[0]).query)
    assert query["gpu_model"] == [_GPU]
    assert query["driver_version"] == [_DRIVER]
    assert query.get("include_pre_pull") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("report", [None, {"expected": 2, "cached": 1, "missing": ["nvidia/cuda:x"]}])
async def test_result_handler_publishes_only_when_measured(context_factory, report):
    state = build_state(specs={"gpu": {"count": 1}}, pre_pull_images=report)
    ctx = context_factory(
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        ssh_pub_keys=[],
        rented=False,
    )

    result = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    if report is None:
        assert "pre_pull_images" not in result.spec
    else:
        assert result.spec["pre_pull_images"] == report
