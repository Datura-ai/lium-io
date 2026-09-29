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
    KILLED_DURING_BOOTSTRAP_EVENT,
    KILLED_DURING_BOOTSTRAP_STEP,
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


_SIGKILLED = _container_state(status="removing", running=False, exit_code=137)
# `docker stop` on the node: SIGTERM, a CMD that handles it (sshd, python, tini) exits 143
_STOPPED = _container_state(status="exited", running=False, exit_code=143)


def _not_running_conflict(*, with_response: bool = False) -> APIError:
    """docker-py's 409 for `exec start` on a stopped container; ``with_response`` attaches the HTTP
    response so the status-code branch decides."""
    message = (
        "409 Client Error for http+docker://ssh/v1.52/exec/exec-id/start: "
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


def _gone_at_exec_create(api: FakeApiClient) -> None:
    # inspect said running, then the node removed the container before exec_create reached it
    def exec_create(**_kwargs):
        api.events.append("exec_create")
        api.inspect_container_error = NotFound("No such container: pod_exec")
        raise NotFound('404 Client Error: Not Found ("No such container: pod_exec")')

    api.exec_create = exec_create


_RACE = ["inspect_container", "exec_create", "inspect_container"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("states", "setup", "events", "expected", "text"),
    [
        ([_oom_killed_state()], None, ["inspect_container"], ("removing", 137, True), "status='removing'"),
        ([_container_state(status="exited", running=False, exit_code=1)], None, ["inspect_container"],
         ("exited", 1, False), "status='exited'"),
        ([_container_state(status="dead", running=False, dead=True, exit_code=1)], None, ["inspect_container"],
         ("dead", 1, False), "status='dead'"),
        # already "No such container" at the first inspect: no state to read
        ([], lambda api: setattr(api, "inspect_container", Mock(side_effect=NotFound("No such container: pod_exec"))),
         None, None, "No such container"),
        # inspect said running, the exec lost the race: Docker answers 409 "container is not running"
        ([_container_state(), _SIGKILLED],
         lambda api: setattr(api, "exec_start", Mock(side_effect=_not_running_conflict(with_response=True))),
         _RACE, ("removing", 137, False), "container is not running"),
        ([_container_state()], _gone_at_exec_create, _RACE, None, "No such container"),
    ],
    ids=["oom", "exited", "dead", "gone-at-first-inspect", "409-after-running", "exec-create-404"],
)  # fmt: skip
async def test_a_gone_container_gets_no_exec_and_its_state_is_read_then(
    states, setup, events, expected, text
):
    api = FakeApiClient()
    api.container_states = states
    if setup:
        setup(api)

    with pytest.raises(ContainerGoneBeforeExec) as info:
        await RentalDockerSdkClient(api).exec_in_container(_exec_spec())

    gone = info.value
    assert gone.container_name == "pod_exec" and text in str(gone)
    assert api.exec_started == [] and (events is None or api.events == events)
    if expected is None:
        assert gone.state is None and container_gone_cause(gone.state) == "removed"
    else:
        assert (gone.state.status, gone.state.exit_code, gone.state.oom_killed) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "exec_error"),
    [
        (_container_state(), _not_running_conflict()),
        (_container_state(status="paused", paused=True), None),
    ],
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


def _bootstrapping_create(
    svc, monkeypatch, api: FakeApiClient, *, skip_ssh_bootstrap: bool = False
) -> _SdkExecClient:
    """The happy deploy flow with the real key injection, sshd bootstrap and environment steps."""
    ssh = _ssh_client()
    _patch_happy(svc, monkeypatch, ssh)
    client = _SdkExecClient(api)
    svc.rental_docker_client_factory = _FakeRentalDockerFactory(client)
    # the key and script execs stream stdin over an exec socket; the FakeApiClient has none, and
    # the transport is not what is under test here — inspect, exec_create and exec_start are
    monkeypatch.setattr(
        rental_docker_sdk, "_write_stdin_and_read_exec_output", lambda _socket, _stdin: (b"", b"")
    )
    real = DockerService.install_open_ssh_server_and_start_ssh_service_with_rental_docker
    bootstrap = AsyncMock(return_value=True) if skip_ssh_bootstrap else real.__get__(svc)
    monkeypatch.setattr(
        svc, "install_open_ssh_server_and_start_ssh_service_with_rental_docker", bootstrap
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
    extras = [getattr(getattr(r, "msg", None), "extra", {}) for r in caplog.records]
    return [e for e in extras if e.get("event") == KILLED_DURING_BOOTSTRAP_EVENT]


def _failure_extra(caplog) -> dict:
    return next(r.msg.extra for r in caplog.records if str(r.msg) == "Failed create_container")


def _env_payload():
    return _payload(custom_options=CustomOptions(environment={"APP_MODE": "prod"}))


def _gone_at_inspect(api: FakeApiClient, n: int) -> list[str]:
    """Inspect call `n` (1-based) and every later one find no container: Docker's autoremove took it."""
    real_inspect, looks = api.inspect_container, []

    def inspect(container_name):
        looks.append(container_name)
        if len(looks) >= n:
            raise NotFound("No such container: " + container_name)
        return real_inspect(container_name)

    api.inspect_container = inspect
    return looks


def _exec_exits(*codes: int):
    # the kill lands while the exec runs: the exec ends with its status, not a refused exec
    exits = [{"ExitCode": c} for c in codes]
    return lambda api: setattr(api, "exec_inspect", Mock(side_effect=exits))


def _exec_exits_then_gone(n: int, *codes: int):
    # the node finishes removing the container before the inspect that follows the failed exec
    return lambda api: (_exec_exits(*codes)(api), _gone_at_inspect(api, n))


_RUNNING = _container_state()
_REMOVING_0 = _container_state(status="removing", running=False, exit_code=0)
_DEAD_137 = _container_state(status="dead", running=False, dead=True, exit_code=137)
_SIGINT = _container_state(status="exited", running=False, exit_code=130)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "states", "setup", "sentence", "event"),
    [
        ("ssh_bootstrap", [_RUNNING, _oom_killed_state()], None, "it ran out of memory",
         {"cause": "oom", "oom_killed": True, "exit_code": 137, "signal": "SIGKILL", "status": "removing"}),
        ("ssh_bootstrap", [_RUNNING, _SIGKILLED], None, "it was killed (SIGKILL)",
         {"cause": "killed", "oom_killed": False, "exit_code": 137, "signal": "SIGKILL", "status": "removing"}),
        ("ssh_bootstrap", [_RUNNING, _STOPPED], None, "it was stopped (SIGTERM)",
         {"cause": "killed", "oom_killed": False, "exit_code": 143, "signal": "SIGTERM", "status": "exited"}),
        ("ssh_bootstrap", [_RUNNING, _RUNNING, _RUNNING, _DEAD_137], _exec_exits(0, 0, 137),
         "it was killed (SIGKILL)", {"cause": "killed", "exit_code": 137, "status": "dead"}),
        ("ssh_bootstrap", [_RUNNING] * 3, _exec_exits_then_gone(4, 0, 0, 137), "it was removed",
         {"cause": "removed", "exit_code": None, "status": None}),
        ("add_public_keys", [_oom_killed_state()], None, "it ran out of memory", {"cause": "oom"}),
        # a SIGTERM-handling CMD exits 0 on a host stop, then the node removes the container
        ("add_public_keys", [_REMOVING_0], None, "it was removed", {"cause": "removed"}),
        ("add_public_keys", [_RUNNING, _SIGINT], _exec_exits(130), "it was stopped (SIGINT)",
         {"cause": "killed", "exit_code": 130, "signal": "SIGINT", "status": "exited"}),
        ("add_public_keys", [_RUNNING], _exec_exits_then_gone(2, 137), "it was removed",
         {"cause": "removed", "exit_code": None, "status": None}),
        ("set_environment", [_RUNNING], lambda api: _gone_at_inspect(api, 2), "it was removed",
         {"cause": "removed", "exit_code": None}),
    ],
    ids=["ssh-oom", "ssh-sigkill", "ssh-docker-stop-143", "ssh-exec-137-dead", "ssh-exec-137-404",
         "keys-oom", "keys-removing-exit-0", "keys-exec-130", "keys-exec-137-404", "env-removed"],
)  # fmt: skip
async def test_a_kill_during_a_bootstrap_step_is_killed_during_bootstrap(
    svc, monkeypatch, caplog, step, states, setup, sentence, event
):
    api = FakeApiClient()
    api.container_states = states
    client = _bootstrapping_create(
        svc, monkeypatch, api, skip_ssh_bootstrap=step == "set_environment"
    )
    if setup:
        setup(api)
    caplog.set_level(logging.WARNING)
    payload = _env_payload() if step == "set_environment" else _payload()

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP == "killed_during_bootstrap"
    assert f"the container was stopped by the node before it was ready: {sentence}" in result.detail
    assert f"during {step}" in result.detail and "has no long-running command" not in result.detail
    assert f"(cause={event['cause']} oom_killed=" in result.detail
    assert _failure_extra(caplog)["failure_step"] == "killed_during_bootstrap"
    (logged,) = _events(caplog)
    assert logged["container_name"] == f"pod_{payload.pod_id}" and logged["bootstrap_step"] == step
    assert {k: logged[k] for k in event} == event
    svc.redis_service.add_rented_pod.assert_not_awaited()
    if setup is not None:  # the exec counts below hold only when the failing exec was refused
        return
    if step == "add_public_keys":
        assert api.exec_created == []
    elif step == "ssh_bootstrap":
        # the keys went in; the bootstrap's one exec got none against the dying container
        assert len(client.exec_specs) == 2 and api.events == _RACE


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
            api.container_states = [_SIGKILLED]
        seen.append(container_name)
        return real_inspect(container_name)

    api.container_states = [_container_state()]
    api.inspect_container = inspect
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)

    # as miner_service / compute_client do around a create
    with inflight_creates.track(payload.pod_id):
        result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "cancelled_by_delete"
    assert "delete for pod" in result.detail and "killed_during_bootstrap" not in result.detail
    assert _events(caplog) == []
    assert len(api.exec_created) == 1  # the keys; nothing against the removed container


@pytest.mark.asyncio
async def test_an_image_whose_command_exits_at_the_key_injection_keeps_its_own_explanation(
    svc, monkeypatch, caplog
):
    api = FakeApiClient()
    api.container_states = [_container_state(status="exited", running=False, exit_code=0)]
    _bootstrapping_create(svc, monkeypatch, api)
    # the readiness inspect sees `exited`, then autoremove takes it
    looks = _gone_at_inspect(api, 2)
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
@pytest.mark.parametrize(("step", "exit_code"), [("set_environment", 1), ("ssh_bootstrap", 0)])
async def test_an_image_whose_command_exits_after_the_key_step_is_not_a_kill(
    svc, monkeypatch, caplog, step, exit_code
):
    api = FakeApiClient()
    exited = _container_state(status="exited", running=False, exit_code=exit_code)
    api.container_states = [_RUNNING, exited]
    _bootstrapping_create(svc, monkeypatch, api, skip_ssh_bootstrap=step == "set_environment")
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _env_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == step  # the step keeps its name
    assert "killed_during_bootstrap" not in result.detail
    # the exec error is kept as it was, with the container's own exit
    assert "Docker container is not ready for exec" in result.detail
    assert f"exit_code={exit_code}" in result.detail
    assert _events(caplog) == []  # no KILLED_DURING_BOOTSTRAP event: nothing on the node killed it
