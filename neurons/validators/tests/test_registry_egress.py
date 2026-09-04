"""DAH-2835 — RegistryEgressCheck.

No check touches the registry, so a host whose egress to Docker Hub died stays verified,
scored and rentable, and fails every rental of an image it has not cached at `docker_pull`.
This check asks the one question nothing else asks.

A failed probe is an AVAILABILITY error (DAH-2748), the shared class for "we could not reach
something": one is enough to zero the score and take the node off the market. So the tests below
guard two things — that a real failure carries that category, and that everything which leaves
the probe unable to answer accuses nobody.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from neurons.validators.src.services.task.checks.registry_egress import RegistryEgressCheck
from neurons.validators.src.services.task.availability import (
    AVAILABILITY_CATEGORY,
    AvailabilityErrorCode,
)
from neurons.validators.src.services.task.messages import RegistryEgressMessages as Msg

from tests.helpers import build_context_config, build_state

_DIGESTS = {"daturaai/torch:2.4.0": "sha256:aaa"}


def _ssh(stdout: str = "401", exit_status: int = 0, raises: bool = False) -> AsyncMock:
    ssh = AsyncMock()
    if raises:
        ssh.run = AsyncMock(side_effect=RuntimeError("ssh down"))
    else:
        ssh.run = AsyncMock(return_value=Mock(exit_status=exit_status, stdout=stdout))
    return ssh


def _ctx(context_factory, ssh: AsyncMock, digests: dict[str, str] | None = None):
    return context_factory(
        config=build_context_config(
            default_docker_image_digests=_DIGESTS if digests is None else digests
        ),
        state=build_state(),
        ssh=ssh,
    )


def test_check_is_fatal():
    # A host that cannot pull cannot serve a rental, so it scores 0 like every other fatal check.
    assert RegistryEgressCheck.fatal is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["401", "200"])
async def test_registry_answers_passes(context_factory, status):
    # /v2/ answers 401 without credentials and 200 with them. Both prove egress works.
    ssh = _ssh(stdout=status)
    ctx = _ctx(context_factory, ssh)

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.REACHABLE.reason
    cmd = ssh.run.await_args.args[0]
    assert "registry-1.docker.io/v2/" in cmd
    assert "--retry 1" in cmd
    assert ssh.run.await_args.kwargs["check"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["000", "503"])
async def test_no_answer_fails_the_host(context_factory, status):
    # 000 is what curl writes when nothing answered; a 5xx is not egress a pull can run on.
    ctx = _ctx(context_factory, _ssh(stdout=status, exit_status=28))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is False
    # The category is what carries the node off the market: services/task/service.py reads it
    # off the last event and the backend hides any node whose cycle reported one.
    assert result.event.category == AVAILABILITY_CATEGORY
    assert result.event.reason_code == AvailabilityErrorCode.DOCKER_HUB_UNREACHABLE
    assert result.event.what_we_saw["http_status"] == status
    # Who could not reach what — the field a future container-to-Hugging-Face check fills in too.
    assert result.event.what_we_saw["reacher"] == "container"
    assert result.event.what_we_saw["reached"] == "docker_hub"
    assert result.event.remediation


@pytest.mark.asyncio
async def test_validator_lost_docker_hub_accuses_nobody(context_factory):
    # The guard that stops a Docker Hub outage from zeroing the whole fleet: an empty digest
    # snapshot means the VALIDATOR could not reach the registry, so a host that cannot reach
    # it either proves nothing. No SSH call is made at all.
    ssh = _ssh(stdout="000")
    ctx = _ctx(context_factory, ssh, digests={})

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.category != AVAILABILITY_CATEGORY
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_ssh_failure_accuses_nobody(context_factory):
    # An SSH error says nothing about the registry, and our own inability to measure must
    # never zero a provider.
    ctx = _ctx(context_factory, _ssh(raises=True))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason


@pytest.mark.asyncio
async def test_unreadable_output_accuses_nobody(context_factory):
    # curl printed something that is not a status code: unknown, not unreachable.
    ctx = _ctx(context_factory, _ssh(stdout="curl: command not found", exit_status=127))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
