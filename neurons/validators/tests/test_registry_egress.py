"""DAH-2835 — RegistryEgressCheck.

A host that cannot reach Docker Hub still passes every check we have, because none of them
touches the registry: the sysbox probe bundles its image, `docker run` carries no `--pull`,
and the cached-template check only inspects what is already local. So the host stays
verified and rentable, and every rental of a non-cached image dies at `docker_pull`.

This check asks the one question nothing else asks. It never changes score — the backend
hides the host from the listing instead.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from neurons.validators.src.services.task.checks.registry_egress import RegistryEgressCheck
from neurons.validators.src.services.task.messages import RegistryEgressMessages as Msg

from tests.helpers import build_context_config, build_state

_DIGESTS = {"daturaai/torch:2.4.0": "sha256:aaa"}


def _ssh(stdout="401", exit_status=0, raises=False):
    ssh = AsyncMock()
    if raises:
        ssh.run = AsyncMock(side_effect=RuntimeError("ssh down"))
    else:
        ssh.run = AsyncMock(return_value=Mock(exit_status=exit_status, stdout=stdout))
    return ssh


def _ctx(context_factory, ssh, digests=None):
    return context_factory(
        config=build_context_config(
            default_docker_image_digests=_DIGESTS if digests is None else digests
        ),
        state=build_state(),
        ssh=ssh,
    )


def test_check_is_not_fatal():
    # Listing-only enforcement: a broken registry must never zero a provider's score.
    assert RegistryEgressCheck.fatal is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["401", "200"])
async def test_registry_answers_publishes_reachable(context_factory, status):
    # /v2/ answers 401 without credentials and 200 with them. Both prove egress works.
    ssh = _ssh(stdout=status)
    ctx = _ctx(context_factory, ssh)

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.REACHABLE.reason
    assert result.updates["state"].registry_reachable is True
    cmd = ssh.run.await_args.args[0]
    assert "registry-1.docker.io/v2/" in cmd
    assert ssh.run.await_args.kwargs["check"] is False


@pytest.mark.asyncio
async def test_timeout_publishes_unreachable(context_factory):
    # curl writes 000 when it never got a response.
    ctx = _ctx(context_factory, _ssh(stdout="000", exit_status=28))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNREACHABLE.reason
    assert result.event.what_we_saw["http_status"] == "000"
    assert result.updates["state"].registry_reachable is False


@pytest.mark.asyncio
async def test_server_error_publishes_unreachable(context_factory):
    # A 5xx from the registry is not egress the host can rent on either.
    ctx = _ctx(context_factory, _ssh(stdout="503"))

    result = await RegistryEgressCheck().run(ctx)

    assert result.updates["state"].registry_reachable is False


@pytest.mark.asyncio
async def test_validator_lost_docker_hub_skips(context_factory):
    # The guard that stops a Docker Hub outage from hiding the whole fleet: an empty digest
    # snapshot means the VALIDATOR could not reach the registry this cycle, so a host that
    # cannot reach it either proves nothing. Publish nothing, leave the last verdict standing.
    ssh = _ssh(stdout="000")
    ctx = _ctx(context_factory, ssh, digests={})

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_ssh_failure_fails_open(context_factory):
    # An SSH error says nothing about the registry. Publish nothing rather than a false verdict.
    ctx = _ctx(context_factory, _ssh(raises=True))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_unreadable_output_fails_open(context_factory):
    # curl printed something that is not a status code: unknown, not unreachable.
    ctx = _ctx(context_factory, _ssh(stdout="curl: command not found", exit_status=127))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}
