"""A pod killed between `docker run` and the end of the bootstrap reports why.

The readiness inspect that first sees the container gone (`removing` / `exited` / `dead`, or already
"No such container") ends the bootstrap there, carrying the State read at that moment; the create
names the failure `killed_during_bootstrap` with `oom_killed` and the exit code, or
`cancelled_by_delete` when the validator's own delete for that pod is in flight.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from docker.errors import APIError, NotFound
from payload_models.payloads import ContainerCreated, CustomOptions, FailedContainerRequest
from services.docker_service import (
    CONTAINER_GONE_KILL_CAUSES,
    KILLED_DURING_BOOTSTRAP_EVENT,
    KILLED_DURING_BOOTSTRAP_STEP,
    ContainerKilledDuringBootstrap,
    DockerService,
    container_gone_cause,
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
    state = _container_state(status="removing", running=False, exit_code=137)
    state["State"]["OOMKilled"] = True
    return state


def _sigkilled_state() -> dict:
    return _container_state(status="removing", running=False, exit_code=137)


def _stopped_state() -> dict:
    # `docker stop` on the node: SIGTERM, a CMD that handles it (sshd, python, tini) exits 143
    return _container_state(status="exited", running=False, exit_code=143)


def _not_running_conflict(*, with_response: bool = False) -> APIError:
    """docker-py's 409 for `exec start` on a stopped container; ``with_response`` attaches the HTTP
    response so the status-code branch decides."""
    message = (
        '409 Client Error for http+docker://ssh/v1.52/exec/exec-id/start: '
        'Conflict ("container is not running")'
    )
    if not with_response:
        return APIError(message)
    response = requests.Response()
    response.status_code = 409
    response._content = b'{"message":"container is not running"}'
    return APIError(message, response=response, explanation="container is not running")


def _exec_spec() -> ContainerExecSpec:
    return ContainerExecSpec(container_name="pod_exec", argv=("sh", "-c", "true"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (_oom_killed_state(), ("removing", 137, True)),
        (_container_state(status="exited", running=False, exit_code=1), ("exited", 1, False)),
        (_container_state(status="dead", running=False, dead=True, exit_code=1), ("dead", 1, False)),
    ],
)
async def test_a_gone_container_gets_no_exec_and_its_state_is_read_at_that_inspect(state, expected):
    api = FakeApiClient()
    api.container_states = [state]

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert api.exec_created == [] and api.events == ["inspect_container"]
    gone = info.value
    assert gone.container_name == "pod_exec" and gone.state is not None
    assert (gone.state.status, gone.state.exit_code, gone.state.oom_killed) == expected
    assert f"status='{expected[0]}'" in str(gone)


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
    api.exec_start = Mock(side_effect=_not_running_conflict(with_response=True))

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert api.events == ["inspect_container", "exec_create", "inspect_container"]
    assert info.value.state is not None
    assert (info.value.state.exit_code, info.value.state.oom_killed) == (137, False)
    assert "container is not running" in str(info.value)


@pytest.mark.asyncio
async def test_an_exec_create_404_after_a_running_inspect_is_gone_not_a_plain_exec_error():
    # inspect said running, then the node removed the container before exec_create reached it
    api = FakeApiClient()
    api.container_states = [_container_state()]

    def exec_create(**_kwargs):
        api.events.append("exec_create")
        api.inspect_container_error = NotFound("No such container: pod_exec")
        raise NotFound("404 Client Error: Not Found (\"No such container: pod_exec\")")

    api.exec_create = exec_create

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert api.events == ["inspect_container", "exec_create", "inspect_container"]
    assert info.value.state is None
    assert container_gone_cause(info.value.state) == "removed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "exec_error"),
    [(_container_state(), _not_running_conflict()), (_container_state(status="paused", paused=True), None)],
    ids=["refused-but-back", "paused"],
)
async def test_a_container_that_is_back_or_paused_is_the_plain_error(state, exec_error):
    api = FakeApiClient()
    api.container_states = [state]
    if exec_error is not None:
        api.exec_start = Mock(side_effect=exec_error)

    with pytest.raises(RentalDockerOperationError) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    assert not isinstance(info.value, ContainerGoneBeforeExec)


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


def _bootstrapping_create(svc, monkeypatch, api: FakeApiClient, *, skip_ssh_bootstrap: bool = False) -> _SdkExecClient:
    """The happy deploy flow with the real key injection, sshd bootstrap and environment steps."""
    ssh = _ssh_client()
    _patch_happy(svc, monkeypatch, ssh)
    client = _SdkExecClient(api)
    svc.rental_docker_client_factory = _FakeRentalDockerFactory(client)
    # the key and script execs stream stdin over an exec socket; the FakeApiClient has none, and
    # the transport is not what is under test here — inspect, exec_create and exec_start are
    monkeypatch.setattr(rental_docker_sdk, "_write_stdin_and_read_exec_output", lambda _socket, _stdin: (b"", b""))
    bootstrap = (
        AsyncMock(return_value=True)
        if skip_ssh_bootstrap
        else DockerService.install_open_ssh_server_and_start_ssh_service_with_rental_docker.__get__(svc)
    )
    monkeypatch.setattr(svc, "install_open_ssh_server_and_start_ssh_service_with_rental_docker", bootstrap)
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


def _env_payload():
    return _payload(custom_options=CustomOptions(environment={"APP_MODE": "prod"}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gone_state", "sentence", "event_fields"),
    [
        (_oom_killed_state(), "it ran out of memory", ("oom", True, 137, "SIGKILL", "removing")),
        (_sigkilled_state(), "it was killed (SIGKILL)", ("killed", False, 137, "SIGKILL", "removing")),
        (_stopped_state(), "it was stopped (SIGTERM)", ("killed", False, 143, "SIGTERM", "exited")),
    ],
    ids=["oom", "sigkill", "docker-stop-143"],
)
async def test_a_kill_during_the_ssh_bootstrap_is_killed_during_bootstrap(
    svc, monkeypatch, caplog, gone_state, sentence, event_fields
):
    api = FakeApiClient()
    # running for the key injection; killed when the bootstrap looks
    api.container_states = [_container_state(), gone_state]
    client = _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)
    payload = _payload()

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP == "killed_during_bootstrap"
    # the keys went in; the bootstrap asked for one exec and got none against the dying container
    assert len(client.exec_specs) == 2
    assert api.events == ["inspect_container", "exec_create", "inspect_container"]
    cause, oom, exit_code, signal, status = event_fields
    assert f"the container was stopped by the node before it was ready: {sentence}" in result.detail
    assert "during ssh_bootstrap" in result.detail
    assert (
        f"cause={cause} oom_killed={str(oom).lower()} exit_code={exit_code} signal='{signal}' status='{status}'"
        in result.detail
    )
    assert "has no long-running command" not in result.detail
    assert _failure_extra(caplog)["failure_step"] == "killed_during_bootstrap"
    (event,) = _events(caplog)
    assert event["container_name"] == f"pod_{payload.pod_id}" and event["bootstrap_step"] == "ssh_bootstrap"
    fields = (event["cause"], event["oom_killed"], event["exit_code"], event["signal"], event["status"])
    assert fields == event_fields
    svc.redis_service.add_rented_pod.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_delete_in_flight_makes_it_cancelled_by_delete(svc, monkeypatch, caplog):
    api = FakeApiClient()
    payload = _payload()
    real_inspect = api.inspect_container
    seen: list[str] = []

    def inspect(container_name):
        # the second look is the bootstrap's: by then the pod's delete has landed on the node
        # (the cancel-on-delete flag is up) and taken the container with it
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
@pytest.mark.parametrize(
    ("gone_state", "cause", "sentence"),
    [
        (_oom_killed_state(), "oom", "it ran out of memory"),
        # a SIGTERM-handling CMD exits 0 on a host stop, then the node removes the container
        (_container_state(status="removing", running=False, exit_code=0), "removed", "it was removed"),
    ],
    ids=["oom", "removing-exit-0"],
)
async def test_a_kill_seen_at_the_key_injection_is_the_kill_not_an_exiting_image(
    svc, monkeypatch, caplog, gone_state, cause, sentence
):
    api = FakeApiClient()
    api.container_states = [gone_state]  # gone before the very first exec
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "killed_during_bootstrap"
    assert "has no long-running command" not in result.detail
    assert sentence in result.detail and "during add_public_keys" in result.detail
    assert api.exec_created == []
    (event,) = _events(caplog)
    assert (event["bootstrap_step"], event["cause"]) == ("add_public_keys", cause)


@pytest.mark.asyncio
async def test_an_image_whose_command_exits_at_the_key_injection_keeps_its_own_explanation(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_container_state(status="exited", running=False, exit_code=0)]
    _bootstrapping_create(svc, monkeypatch, api)
    real_inspect = api.inspect_container
    looks: list[str] = []

    def inspect(container_name):
        # the readiness inspect sees `exited`; Docker's autoremove takes the container right after,
        # so a second inspect finds nothing
        looks.append(container_name)
        if len(looks) > 1:
            raise NotFound("No such container: " + container_name)
        return real_inspect(container_name)

    api.inspect_container = inspect
    caplog.set_level(logging.WARNING)
    payload = _payload()

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "add_public_keys"
    assert "has no long-running command" in result.detail and "exit_code=0" in result.detail
    assert looks == ["pod_" + payload.pod_id]  # the State the gone error carried was reused
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_the_start_path_keeps_its_soft_failure_for_a_gone_container(svc, monkeypatch, caplog):
    # `start_existing_container` / the edit undo read the bool and warn; only the create asks to raise
    api = FakeApiClient()
    api.container_states = [_container_state(status="exited", running=False, exit_code=137)]
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    caplog.set_level(logging.WARNING)
    kwargs = dict(container_name="pod_exec", log_tag="t", log_extra={})

    ok = await svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
        docker_client=_SdkExecClient(api), **kwargs
    )

    assert ok is False
    assert api.exec_created == []
    with pytest.raises(ContainerGoneBeforeExec):
        await svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
            docker_client=_SdkExecClient(api), raise_if_container_gone=True, **kwargs
        )


@pytest.mark.asyncio
async def test_a_healthy_container_bootstraps_as_before(svc, monkeypatch, caplog):
    api = FakeApiClient()
    api.container_states = [_container_state()]
    client = _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _env_payload())

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
    _bootstrapping_create(svc, monkeypatch, api, skip_ssh_bootstrap=True)
    real_inspect = api.inspect_container
    looks: list[str] = []

    def inspect(container_name):
        looks.append(container_name)
        if len(looks) == 2:
            raise NotFound("No such container: " + container_name)
        return real_inspect(container_name)

    api.inspect_container = inspect
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _env_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "killed_during_bootstrap"
    assert "it was removed" in result.detail and "during set_environment" in result.detail
    (event,) = _events(caplog)
    assert (event["bootstrap_step"], event["cause"], event["exit_code"]) == ("set_environment", "removed", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(("step", "exit_code"), [("set_environment", 1), ("ssh_bootstrap", 0)])
async def test_an_image_whose_command_exits_after_the_key_step_is_not_a_kill(svc, monkeypatch, caplog, step, exit_code):
    api = FakeApiClient()
    api.container_states = [_container_state(), _container_state(status="exited", running=False, exit_code=exit_code)]
    _bootstrapping_create(svc, monkeypatch, api, skip_ssh_bootstrap=step == "set_environment")
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _env_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == step  # the step keeps its name
    assert "killed_during_bootstrap" not in result.detail
    assert _events(caplog) == []  # no KILLED_DURING_BOOTSTRAP event: nothing on the node killed it


def _snapshot(status: str, exit_code: int | None, oom_killed: bool) -> ContainerStateSnapshot:
    return ContainerStateSnapshot(status, False, False, exit_code, 0, None, oom_killed)


@pytest.mark.parametrize(
    ("state", "cause", "sentence"),
    [
        (_snapshot("removing", 137, True), "oom", "it ran out of memory"),
        (_snapshot("exited", 137, True), "oom", "it ran out of memory"),
        (_snapshot("removing", 137, False), "killed", "it was killed (SIGKILL)"),
        (_snapshot("dead", 137, False), "killed", "it was killed (SIGKILL)"),
        (_snapshot("exited", 143, False), "killed", "it was stopped (SIGTERM)"),
        (_snapshot("removing", 143, False), "killed", "it was stopped (SIGTERM)"),
        (_snapshot("exited", 130, False), "killed", "it was stopped (SIGINT)"),
        (None, "removed", "it was removed"),
        (_snapshot("removing", 0, False), "removed", "it was removed"),
    ],
)
def test_the_cause_and_the_renter_sentence_follow_the_state(state, cause, sentence):
    assert container_gone_cause(state) == cause in CONTAINER_GONE_KILL_CAUSES
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
