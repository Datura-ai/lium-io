"""A pod killed between `docker run` and the end of the bootstrap reports why.

19 Sep, one node, one hour, 3 failed rents: `Docker container is not ready for exec:
status='removing' exit_code=137`, then `exec start: Conflict ("container is not running")` and
`inspect: No such container` — the container was killed (137: OOM, or a `docker rm -f` / `docker
kill` on the node) while the SSH bootstrap ran, and each rent failed as a generic `ssh_bootstrap` /
`set_environment` exec error. Now the readiness inspect that first sees the container gone
(`removing` / `exited` / `dead`, or already "No such container") ends the bootstrap there — no exec
is attempted against it — carrying the State read at that moment; the create names the failure
`killed_during_bootstrap` with `oom_killed` and the exit code, or `cancelled_by_delete` when the
validator's own delete for that pod is in flight (DAH-2728).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from docker.errors import APIError, NotFound
from payload_models.payloads import ContainerCreated, CustomOptions, FailedContainerRequest
from services.docker_service import (
    KILLED_DURING_BOOTSTRAP_EVENT,
    KILLED_DURING_BOOTSTRAP_STEP,
    ContainerKilledDuringBootstrap,
    DockerService,
    inflight_creates,
)
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerGoneBeforeExec,
    ContainerStateSnapshot,
    RentalDockerOperationError,
    RentalDockerSdkClient,
)
from test_deploy_optimizations import (
    _executor_info,
    _FakeRentalDockerClient,
    _FakeRentalDockerFactory,
    _patch_happy,
    _payload,
    _ssh_client,
)
from test_rental_docker_sdk import FakeApiClient, _container_state

from services import rental_docker_sdk


def _oom_killed_state() -> dict:
    # `docker inspect` of the 19 Sep containers at the moment the bootstrap saw them
    state = _container_state(status="removing", running=False, exit_code=137)
    state["State"]["OOMKilled"] = True
    return state


def _sigkilled_state() -> dict:
    return _container_state(status="removing", running=False, exit_code=137)


def _not_running_conflict() -> APIError:
    return APIError(
        '409 Client Error for http+docker://ssh/v1.52/exec/exec-id/start: '
        'Conflict ("container is not running")'
    )


def _exec_spec() -> ContainerExecSpec:
    return ContainerExecSpec(container_name="pod_exec", argv=("sh", "-c", "true"))


# ------------------------------------------------------------------
# The bootstrap wait (Docker SDK client)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_removing_container_gets_no_exec_and_its_state_is_read_at_that_inspect():
    api = FakeApiClient()
    api.container_states = [_oom_killed_state()]

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert api.exec_created == [] and api.events == ["inspect_container"]
    gone = info.value
    assert gone.container_name == "pod_exec"
    assert gone.state is not None
    assert (gone.state.status, gone.state.exit_code, gone.state.oom_killed) == ("removing", 137, True)
    assert gone.state.killed_by_host is True
    assert "status='removing'" in str(gone) and "exit_code=137" in str(gone)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["exited", "dead"])
async def test_exited_and_dead_are_gone_too(status):
    api = FakeApiClient()
    api.container_states = [_container_state(status=status, running=False, dead=status == "dead", exit_code=1)]

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert api.exec_created == []
    assert info.value.state is not None and info.value.state.exit_code == 1


@pytest.mark.asyncio
async def test_a_container_already_gone_at_the_first_inspect_has_no_state_to_read():
    api = FakeApiClient()
    api.inspect_container = Mock(side_effect=NotFound("No such container: pod_exec"))

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert api.exec_created == []
    assert info.value.state is None
    assert "No such container" in str(info.value)


@pytest.mark.asyncio
async def test_an_exec_refused_because_the_container_stopped_reads_its_state_then():
    # inspect said running, the exec lost the race: Docker answers 409 "container is not running"
    api = FakeApiClient()
    api.container_states = [_container_state(), _sigkilled_state()]
    api.exec_start = Mock(side_effect=_not_running_conflict())

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert len(api.exec_created) == 1  # the one that was refused, none after it
    assert api.events == ["inspect_container", "exec_create", "inspect_container"]
    assert info.value.state is not None
    assert (info.value.state.exit_code, info.value.state.oom_killed) == (137, False)
    assert "container is not running" in str(info.value)


@pytest.mark.asyncio
async def test_an_exec_refused_on_a_container_that_is_back_is_the_plain_error():
    api = FakeApiClient()
    api.container_states = [_container_state()]
    api.exec_start = Mock(side_effect=_not_running_conflict())

    with pytest.raises(RentalDockerOperationError) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert not isinstance(info.value, ContainerGoneBeforeExec)
    assert api.events == ["inspect_container", "exec_create", "inspect_container"]


@pytest.mark.asyncio
async def test_a_paused_container_is_the_plain_terminal_error_not_gone():
    api = FakeApiClient()
    api.container_states = [_container_state(status="paused", paused=True)]

    with pytest.raises(RentalDockerOperationError) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert not isinstance(info.value, ContainerGoneBeforeExec)
    assert api.exec_created == []


@pytest.mark.asyncio
async def test_a_running_container_is_execed_as_before():
    api = FakeApiClient()
    api.container_states = [_container_state()]

    result = await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert result.exit_status == 0
    assert api.events == ["inspect_container", "exec_create"]


# ------------------------------------------------------------------
# The create (DockerService.create_container)
# ------------------------------------------------------------------


class _SdkExecClient(_FakeRentalDockerClient):
    """The deploy-flow fake, with `exec_in_container` going through the real SDK client (readiness
    inspect, exec, the 409 re-inspect) over a FakeApiClient scripted per test."""

    def __init__(self, api: FakeApiClient):
        super().__init__(image_exists_result=True)
        self.api = api
        self.sdk = RentalDockerSdkClient(api)

    async def exec_in_container(self, spec):
        self.exec_specs.append(spec)
        return await self.sdk.exec_in_container(spec)

    async def inspect_container_state(self, *, container_name: str) -> ContainerStateSnapshot:
        return await self.sdk.inspect_container_state(container_name=container_name)


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _bootstrapping_create(svc, monkeypatch, api: FakeApiClient) -> _SdkExecClient:
    """The happy deploy flow with the real key injection, sshd bootstrap and environment steps."""
    ssh = _ssh_client()
    _patch_happy(svc, monkeypatch, ssh)
    client = _SdkExecClient(api)
    svc.rental_docker_client_factory = _FakeRentalDockerFactory(client)
    # the key and script execs stream stdin over an exec socket; the FakeApiClient has none, and
    # the transport is not what is under test here — inspect, exec_create and exec_start are
    monkeypatch.setattr(rental_docker_sdk, "_write_stdin_and_read_exec_output", lambda _socket, _stdin: (b"", b""))
    monkeypatch.setattr(
        svc,
        "install_open_ssh_server_and_start_ssh_service_with_rental_docker",
        DockerService.install_open_ssh_server_and_start_ssh_service_with_rental_docker.__get__(svc),
    )
    return client


async def _create(svc, payload):
    return await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _events(caplog) -> list[dict]:
    return [
        record.msg.extra
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event") == KILLED_DURING_BOOTSTRAP_EVENT
    ]


def _failure_extra(caplog) -> dict:
    return next(record.msg.extra for record in caplog.records if str(record.msg) == "Failed create_container")


@pytest.mark.asyncio
async def test_oom_killed_during_the_ssh_bootstrap_is_killed_during_bootstrap_oom(svc, monkeypatch, caplog):
    api = FakeApiClient()
    # running for the key injection; OOM-killed and being removed when the bootstrap looks
    api.container_states = [_container_state(), _oom_killed_state()]
    client = _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)
    payload = _payload()

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP == "killed_during_bootstrap"
    # the keys went in; the bootstrap asked for one exec and got none against the dying container
    assert len(client.exec_specs) == 2
    assert api.events == ["inspect_container", "exec_create", "inspect_container"]
    detail = result.detail
    assert "killed_during_bootstrap" in detail
    assert "the container was stopped by the node before it was ready: it ran out of memory" in detail
    assert "during ssh_bootstrap" in detail
    assert "cause=oom oom_killed=true exit_code=137 status='removing'" in detail
    assert "Failed to set environment variables" not in detail
    assert _failure_extra(caplog)["failure_step"] == "killed_during_bootstrap"

    (event,) = _events(caplog)
    assert event["container_name"] == f"pod_{payload.pod_id}"
    assert event["bootstrap_step"] == "ssh_bootstrap"
    assert (event["cause"], event["oom_killed"], event["exit_code"], event["status"]) == ("oom", True, 137, "removing")
    svc.redis_service.add_rented_pod.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_sigkill_without_oom_is_killed_not_oom(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_container_state(), _sigkilled_state()]
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "killed_during_bootstrap"
    assert "it was killed" in result.detail and "oom_killed=false exit_code=137" in result.detail
    (event,) = _events(caplog)
    assert (event["cause"], event["oom_killed"]) == ("killed", False)


@pytest.mark.asyncio
async def test_a_delete_in_flight_makes_it_cancelled_by_delete(svc, monkeypatch, caplog):
    api = FakeApiClient()
    payload = _payload()
    real_inspect = api.inspect_container
    seen: list[str] = []

    def inspect(container_name):
        # the second look is the bootstrap's: by then the pod's delete has landed on the node
        # (the DAH-2728 flag is up) and taken the container with it
        if len(seen) == 1:
            inflight_creates.cancel(payload.pod_id)
            api.container_states = [_sigkilled_state()]
        seen.append(container_name)
        return real_inspect(container_name)

    api.container_states = [_container_state()]
    api.inspect_container = inspect
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    with inflight_creates.track(payload.pod_id):  # as miner_service / compute_client do around a create
        result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "cancelled_by_delete"
    assert "delete for pod" in result.detail and "killed_during_bootstrap" not in result.detail
    assert _events(caplog) == []
    assert len(api.exec_created) == 1  # the keys; nothing against the removed container


@pytest.mark.asyncio
async def test_a_kill_seen_at_the_key_injection_is_the_kill_not_an_exiting_image(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_oom_killed_state()]  # gone before the very first exec
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "killed_during_bootstrap"
    assert "has no long-running command" not in result.detail  # DAH-3678 is for an image's own exit
    assert "it ran out of memory" in result.detail and "during add_public_keys" in result.detail
    assert api.exec_created == []


@pytest.mark.asyncio
async def test_an_image_whose_command_exits_at_the_key_injection_keeps_its_own_explanation(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_container_state(status="exited", running=False, exit_code=0)]
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "add_public_keys"
    assert "has no long-running command" in result.detail
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_the_start_path_keeps_its_soft_failure_for_a_gone_container(svc, monkeypatch, caplog):
    # `start_existing_container` / the edit undo read the bool and warn; only the create asks to raise
    api = FakeApiClient()
    api.container_states = [_container_state(status="exited", running=False, exit_code=137)]
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    caplog.set_level(logging.WARNING)

    ok = await svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
        docker_client=_SdkExecClient(api), container_name="pod_exec", log_tag="t", log_extra={}
    )

    assert ok is False
    assert api.exec_created == []
    with pytest.raises(ContainerGoneBeforeExec):
        await svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
            docker_client=_SdkExecClient(api),
            container_name="pod_exec",
            log_tag="t",
            log_extra={},
            raise_if_container_gone=True,
        )


@pytest.mark.asyncio
async def test_a_healthy_container_bootstraps_as_before(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_container_state()]
    client = _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload(custom_options=CustomOptions(environment={"APP_MODE": "prod"})))

    assert isinstance(result, ContainerCreated), getattr(result, "detail", result)
    assert _events(caplog) == []
    # keys, bootstrap script written, bootstrap script run, environment: one inspect, one exec each
    assert api.events == ["inspect_container", "exec_create"] * 4
    assert len(client.exec_specs) == 4
    svc.redis_service.add_rented_pod.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_kill_seen_at_set_environment_names_that_step(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_container_state()]
    _bootstrapping_create(svc, monkeypatch, api)
    monkeypatch.setattr(
        svc, "install_open_ssh_server_and_start_ssh_service_with_rental_docker", AsyncMock(return_value=True)
    )
    real_inspect = api.inspect_container
    looks: list[str] = []

    def inspect(container_name):
        looks.append(container_name)
        if len(looks) == 2:
            raise NotFound("No such container: " + container_name)
        return real_inspect(container_name)

    api.inspect_container = inspect
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload(custom_options=CustomOptions(environment={"APP_MODE": "prod"})))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "killed_during_bootstrap"
    assert "it was removed" in result.detail and "during set_environment" in result.detail
    (event,) = _events(caplog)
    assert (event["bootstrap_step"], event["cause"], event["exit_code"]) == ("set_environment", "removed", None)


# ------------------------------------------------------------------
# The classification
# ------------------------------------------------------------------


def _snapshot(status: str, exit_code: int | None, oom_killed: bool) -> ContainerStateSnapshot:
    return ContainerStateSnapshot(
        status=status, running=False, restarting=False, exit_code=exit_code,
        restart_count=0, error=None, oom_killed=oom_killed,
    )


@pytest.mark.parametrize(
    ("state", "cause", "sentence"),
    [
        (_snapshot("removing", 137, True), "oom", "it ran out of memory"),
        (_snapshot("exited", 137, True), "oom", "it ran out of memory"),
        (_snapshot("removing", 137, False), "killed", "it was killed"),
        (_snapshot("dead", 137, False), "killed", "it was killed"),
        (None, "removed", "it was removed"),
        (_snapshot("removing", 0, False), "removed", "it was removed"),
        (_snapshot("exited", 1, False), "exited", "its command exited"),
    ],
)
def test_the_cause_and_the_renter_sentence_follow_the_state(state, cause, sentence):
    killed = ContainerKilledDuringBootstrap(
        container_name="pod_x", bootstrap_step="ssh_bootstrap", state=state, detail="Docker container is not ready for exec"
    )

    assert killed.cause == cause
    text = str(killed)
    assert text.startswith("killed_during_bootstrap: the container ")
    assert sentence in text and "during ssh_bootstrap" in text
    assert f"cause={cause} oom_killed={str(bool(state and state.oom_killed)).lower()}" in text
    assert killed.exit_code == (state.exit_code if state else None)
    assert text.endswith("Docker container is not ready for exec")
