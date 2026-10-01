"""A pod killed between `docker run` and the end of the bootstrap reports why.

The readiness inspect that first sees the container gone (`removing` / `exited` / `dead`, or already
"No such container") ends the bootstrap there, carrying the State read at that moment; the create
names the failure `killed_during_bootstrap` with `oom_killed` and the exit code, or
`cancelled_by_delete` when the validator's own delete for that pod is in flight.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from unittest.mock import AsyncMock, Mock

import asyncssh
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
    own_sweep_removals,
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
from services.prerun_host_probe import parse_container_listing


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
    assert "is not running" not in gone.kill_detail
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


def _container_id(name: str, generation: int = 0) -> str:
    return hashlib.sha256(f"{name}/{generation}".encode()).hexdigest()


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


@pytest.fixture(autouse=True)
def _no_sweeps_from_other_tests():
    own_sweep_removals.clear()


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _bootstrapping_create(
    svc, monkeypatch, api: FakeApiClient, *, skip_ssh_bootstrap: bool = False
) -> _SdkExecClient:
    """The happy deploy flow with the real key injection, sshd bootstrap and environment steps."""
    ssh = _ssh_client()
    _patch_happy(svc, monkeypatch, ssh)
    svc._run_rental_docker_create_with_port_retry.side_effect = (
        lambda **kwargs: _container_id(kwargs["container_name"])
    )
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


def _inspect_fails_at(api: FakeApiClient, n: int) -> None:
    """Inspect call `n` (1-based) fails once with a transient daemon error; the calls around it answer."""
    real_inspect, looks = api.inspect_container, []

    def inspect(container_name):
        looks.append(container_name)
        if len(looks) == n:
            raise APIError("500 Server Error: Internal Server Error (\"context deadline exceeded\")")
        return real_inspect(container_name)

    api.inspect_container = inspect


def _refused_then_reinspect_fails(api: FakeApiClient, *, gone_after: bool = False) -> None:
    # Docker refuses the key exec with 409, the SDK's re-inspect of the State fails, and the next inspect reads it
    api.exec_start = Mock(side_effect=_not_running_conflict(with_response=True))
    _inspect_fails_at(api, 2)
    if gone_after:
        _gone_at_inspect(api, 3)


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
_DEAD_1 = _container_state(status="dead", running=False, dead=True, exit_code=1)
_SIGINT = _container_state(status="exited", running=False, exit_code=130)
_EXITED_137 = _container_state(status="exited", running=False, exit_code=137)
_BY_NODE = "the container was stopped by the node before it was ready: "
_ENDED_ON = "the container stopped before it was ready: its command ended on "


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "states", "setup", "sentence", "event"),
    [
        ("ssh_bootstrap", [_RUNNING, _oom_killed_state()], None, _BY_NODE + "it ran out of memory",
         {"cause": "oom", "oom_killed": True, "exit_code": 137, "signal": "SIGKILL", "status": "removing"}),
        ("ssh_bootstrap", [_RUNNING, _SIGKILLED], None, _BY_NODE + "it was killed (SIGKILL)",
         {"cause": "killed", "oom_killed": False, "exit_code": 137, "signal": "SIGKILL", "status": "removing"}),
        ("ssh_bootstrap", [_RUNNING, _STOPPED], None, _ENDED_ON + "SIGTERM (exit 143)",
         {"cause": "signaled", "oom_killed": False, "exit_code": 143, "signal": "SIGTERM", "status": "exited"}),
        # an image whose CMD exits 137 itself reads the same as a `docker kill`: neutral, not the node
        ("ssh_bootstrap", [_RUNNING, _EXITED_137], None, _ENDED_ON + "SIGKILL (exit 137)",
         {"cause": "signaled", "oom_killed": False, "exit_code": 137, "signal": "SIGKILL", "status": "exited"}),
        ("ssh_bootstrap", [_RUNNING, _RUNNING, _RUNNING, _DEAD_137], _exec_exits(0, 0, 137),
         _BY_NODE + "it was killed (SIGKILL)", {"cause": "killed", "exit_code": 137, "status": "dead"}),
        ("ssh_bootstrap", [_RUNNING] * 3, _exec_exits_then_gone(4, 0, 0, 137), _BY_NODE + "it was removed",
         {"cause": "removed", "exit_code": None, "status": None}),
        ("add_public_keys", [_oom_killed_state()], None, _BY_NODE + "it ran out of memory", {"cause": "oom"}),
        # a SIGTERM-handling CMD exits 0 on a host stop, then the node removes the container
        ("add_public_keys", [_REMOVING_0], None, _BY_NODE + "it was removed", {"cause": "removed"}),
        # `dead`: a removal the daemon could not finish, whatever the exit code; not the image's exit
        ("add_public_keys", [_DEAD_1], None, _BY_NODE + "it was removed",
         {"cause": "removed", "exit_code": 1, "status": "dead"}),
        ("add_public_keys", [_RUNNING, _SIGINT], _exec_exits(130), _ENDED_ON + "SIGINT (exit 130)",
         {"cause": "signaled", "exit_code": 130, "signal": "SIGINT", "status": "exited"}),
        ("add_public_keys", [_RUNNING], _exec_exits_then_gone(2, 137), _BY_NODE + "it was removed",
         {"cause": "removed", "exit_code": None, "status": None}),
        ("set_environment", [_RUNNING], lambda api: _gone_at_inspect(api, 2), _BY_NODE + "it was removed",
         {"cause": "removed", "exit_code": None}),
        ("set_environment", [_RUNNING, _RUNNING, _SIGKILLED], _exec_exits(0, 137), _BY_NODE + "it was killed (SIGKILL)",
         {"cause": "killed"}),
        # inspect said running, Docker refused the exec with 409 "is not running", the re-inspect reads the kill
        ("add_public_keys", [_RUNNING, _SIGKILLED],
         lambda api: setattr(api, "exec_start", Mock(side_effect=_not_running_conflict(with_response=True))),
         _BY_NODE + "it was killed (SIGKILL)", {"cause": "killed", "exit_code": 137, "status": "removing"}),
        ("ssh_bootstrap", [_RUNNING, _RUNNING, _SIGKILLED],
         lambda api: setattr(api, "exec_start", Mock(side_effect=[None, _not_running_conflict(with_response=True)])),
         _BY_NODE + "it was killed (SIGKILL)", {"cause": "killed", "exit_code": 137, "status": "removing"}),
        # review of e41ca1d: the re-inspect after the 409 fails once, so the key step's own inspect reads the kill
        ("add_public_keys", [_RUNNING, _SIGKILLED], _refused_then_reinspect_fails,
         _BY_NODE + "it was killed (SIGKILL)", {"cause": "killed", "exit_code": 137, "status": "removing"}),
        ("add_public_keys", [_RUNNING], lambda api: _refused_then_reinspect_fails(api, gone_after=True),
         _BY_NODE + "it was removed", {"cause": "removed", "exit_code": None, "status": None}),
    ],
    ids=["ssh-oom", "ssh-sigkill", "ssh-exited-143", "ssh-exited-137", "ssh-exec-137-dead", "ssh-exec-137-404",
         "keys-oom", "keys-removing-exit-0", "keys-dead-exit-1", "keys-exec-130", "keys-exec-137-404", "env-removed",
         "env-exec-137", "keys-exec-409", "ssh-exec-409", "keys-409-reinspect-error", "keys-409-reinspect-error-404"],
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
    assert sentence in result.detail
    # the backend builds the renter's error from msg: the cause sentence, without the diagnosis
    assert sentence in result.msg and "cause=" not in result.msg and "Failed create_container" not in result.msg
    assert f"during {step}" in result.detail and "has no long-running command" not in result.detail
    assert f"(cause={event['cause']} oom_killed=" in result.detail
    # the backend reads Docker's "is not running" as the renter's image exiting, not a kill on the node
    assert "is not running" not in result.detail
    assert _failure_extra(caplog)["failure_step"] == "killed_during_bootstrap"
    failure = next(r for r in caplog.records if str(r.msg) == "Failed create_container")
    assert failure.levelno == logging.ERROR and failure.exc_info is None
    assert failure.msg.extra["reason"] == "killed_during_bootstrap"
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


def _swept_at_inspect(api: FakeApiClient, n: int, container_id: str) -> None:
    """Another create's sweep sends `rm` for ``container_id`` just before inspect call `n` (1-based)."""
    real_inspect, looks = api.inspect_container, []
    loop = asyncio.get_running_loop()

    def inspect(container_name):
        # docker-py runs in a worker thread; the sweep lands on the loop before this inspect returns
        looks.append(container_name)
        if len(looks) == n:
            loop.call_soon_threadsafe(own_sweep_removals.mark, [container_id])
        return real_inspect(container_name)

    api.inspect_container = inspect


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "swept",
    [
        "during-bootstrap",
        "older-same-name-container-before-docker-run",
        "older-same-name-container-during-bootstrap",
        "during-bootstrap-oom",
    ],
)
async def test_a_container_another_create_swept_is_not_a_node_kill(svc, monkeypatch, caplog, swept):
    """A customer's create removes every filler on the node, one still bootstrapping included: that
    create finds its container gone, but the validator sent the `rm`, so no kill is filed. The sweep is
    matched by container ID: a sweep of an older container with the same name (a retry of the pod), or
    an OOM (a sweep's `rm -f` never causes one), still files the kill."""
    api = FakeApiClient()
    api.container_states = [_RUNNING, _oom_killed_state() if swept.endswith("-oom") else _SIGKILLED]
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)
    payload = _payload()
    name = f"pod_{payload.pod_id}"
    this, older = _container_id(name), _container_id(name, generation=1)
    if swept == "older-same-name-container-before-docker-run":
        own_sweep_removals.mark([older])
    elif swept == "older-same-name-container-during-bootstrap":
        _swept_at_inspect(api, 2, older)
    else:
        _swept_at_inspect(api, 2, this)

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    if swept.startswith("older-") or swept.endswith("-oom"):
        assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP
        (logged,) = _events(caplog)
        assert logged["container_name"] == name
        assert logged["cause"] == ("oom" if swept.endswith("-oom") else "killed")
        return
    assert result.failure_step == "ssh_bootstrap"
    assert "stopped by the node" not in result.detail
    assert _events(caplog) == []
    assert any(getattr(r.msg, "extra", {}).get("reason") == "removed_by_own_sweep" for r in caplog.records)


def _sweep_host(listing: str, rm_prints=None, rm_exit: int = 0) -> Mock:
    """`docker ps -a` answers ``listing``; `docker rm` prints ``rm_prints(targets)`` and exits ``rm_exit``."""
    ssh = Mock()
    rms: list[list[str]] = []

    async def run(command, **_kwargs):
        if command.startswith("/usr/bin/docker rm -fv "):
            targets = command.removeprefix("/usr/bin/docker rm -fv ").split()
            rms.append(targets)
            printed = targets if rm_prints is None else rm_prints(targets)
            return Mock(stdout="".join(f"{t}\n" for t in printed), stderr="", exit_status=rm_exit)
        return Mock(stdout=listing, stderr="", exit_status=0)

    ssh.run = AsyncMock(side_effect=run)
    ssh.rms = rms
    return ssh


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("from_probe", "rm_error", "survivor"),
    [(False, None, None), (True, None, None), (False, RuntimeError("rm never sent"), None),
     (False, None, "same"), (False, None, "replacement")],
    ids=["listed", "from-probe", "rm-failed", "filler-survived-the-rm", "replacement-after-the-rm"],
)  # fmt: skip
async def test_the_stale_sweep_records_the_container_ids_it_sends_rm_for(svc, from_probe, rm_error, survivor):
    swept_id = _container_id("filler_swept-1")
    replacement_id = _container_id("filler_swept-1", generation=1)
    # a filler retry took the name with a new ID after the old one's `rm`
    survivors = {"same": {"filler_swept-1": swept_id}, "replacement": {"filler_swept-1": replacement_id}}.get(
        survivor, {}
    )
    ssh = _sweep_host(f"filler_swept-1 {swept_id}\npod_keep {_container_id('pod_keep')}\n")
    probe = (
        Mock(container_names=("filler_swept-1", "pod_keep"), container_ids={"filler_swept-1": swept_id})
        if from_probe else None
    )  # fmt: skip

    async def remove(_ssh, _extra, _pod, _names, _targets, _every, own_ids=()):
        assert list(own_ids) == [swept_id]
        if rm_error is not None:
            raise rm_error
        own_sweep_removals.mark(own_ids)  # what the real `rm` does once it is sent
        return survivors

    svc._remove_stale_containers = remove
    sweep = svc.clean_existing_containers(
        ssh_client=ssh, default_extra={}, pod_name="pod_new", clear_volume=False,
        active_container_names=["pod_keep", "filler_swept-1"], remove_every_filler=True,
        host_probe=probe,
    )

    if rm_error is not None:
        with pytest.raises(RuntimeError, match="rm never sent"):
            await sweep
    else:
        assert await sweep == ["filler_swept-1"]
    if rm_error is not None:  # the stub raised before any SSH rm
        assert ssh.rms == []
        assert not own_sweep_removals.sent_rm_for(swept_id)
        return
    assert own_sweep_removals.sent_rm_for(swept_id)
    # the customer's create removes the replacement by its own ID and records it as well
    assert own_sweep_removals.sent_rm_for(replacement_id) == (survivor == "replacement")
    assert ssh.rms == ([[replacement_id]] if survivor == "replacement" else [])
    assert not own_sweep_removals.sent_rm_for(None)
    # the IDs come with the listing: no `docker inspect` round trip, from the probe or not
    assert all("docker inspect" not in c.args[0] for c in ssh.run.await_args_list)
    # the replacement: its rm and the confirmation that follows it
    assert ssh.run.await_count == (0 if from_probe else 1) + (2 if survivor == "replacement" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attempts", "ours"),
    [(["ok"], True), (["answer-lost"], True), (["no-channel"], False),
     (["answer-lost", "no-channel"], True), (["no-channel", "ok"], True)],
    ids=["answered", "answer-lost", "never-sent", "lost-then-no-channel", "no-channel-then-answered"],
)  # fmt: skip
async def test_a_sweep_id_is_ours_once_its_rm_is_handed_to_ssh(monkeypatch, attempts, ours):
    monkeypatch.setattr("core.utils.wait_fixed", lambda _s: __import__("tenacity").wait_none())
    swept_id = _container_id("filler_swept-1")
    outcomes = iter(attempts)
    ssh = Mock()

    async def run(command, **_kwargs):
        assert command == f"/usr/bin/docker rm -fv {swept_id}"
        assert own_sweep_removals.sent_rm_for(swept_id)
        outcome = next(outcomes)
        if outcome == "no-channel":
            raise asyncssh.ChannelOpenError(asyncssh.OPEN_CONNECT_FAILED, "SSH connection closed")
        if outcome == "answer-lost":
            raise ConnectionResetError("SSH dropped after the rm was sent")
        return Mock(stdout="", stderr="", exit_status=0)

    ssh.run = AsyncMock(side_effect=run)

    rm = DockerService._rm_containers(ssh, [swept_id], max_attempts=len(attempts), own_ids=[swept_id])
    if attempts[-1] == "ok":
        await rm
    else:
        with pytest.raises((asyncssh.ChannelOpenError, ConnectionResetError)):
            await rm

    assert ssh.run.await_count == len(attempts)
    assert own_sweep_removals.sent_rm_for(swept_id) == ours


@pytest.mark.asyncio
@pytest.mark.parametrize("other_order", ["before", "during"])
async def test_an_unsent_rm_keeps_an_id_another_sweep_marked(other_order):
    """Another sweep's sent `rm` of the same ID, made before this one's channel fails or while it waits for
    one, keeps the ID ours."""
    swept_id = _container_id("filler_swept-1")
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail_open(*_args, **_kwargs):
        entered.set()
        await release.wait()
        raise asyncssh.ChannelOpenError(asyncssh.OPEN_CONNECT_FAILED, "SSH connection closed")

    first = Mock(run=AsyncMock(side_effect=fail_open))
    second = Mock(run=AsyncMock(return_value=Mock(stdout="", stderr="", exit_status=0)))
    if other_order == "before":
        await DockerService._rm_containers(second, [swept_id], max_attempts=1, own_ids=[swept_id])
    pending = asyncio.create_task(
        DockerService._rm_containers(first, [swept_id], max_attempts=1, own_ids=[swept_id])
    )
    await entered.wait()
    if other_order == "during":
        await DockerService._rm_containers(second, [swept_id], max_attempts=1, own_ids=[swept_id])
    release.set()

    with pytest.raises(asyncssh.ChannelOpenError):
        await pending
    assert own_sweep_removals.sent_rm_for(swept_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "after",
    ["gone", "still-listed", "listing-failed", "rm-answer-lost-listing-failed", "rm-answer-lost-still-listed",
     "rm-never-sent"],
)  # fmt: skip
async def test_a_replacement_filler_is_ours_once_its_rm_is_handed_to_ssh(svc, monkeypatch, after):
    """The sweep's rule: the ID is ours once its `rm` is sent, whatever the `rm` answers or the confirmation
    lists; only a channel that never opened records none."""
    monkeypatch.setattr("core.utils.wait_fixed", lambda _s: __import__("tenacity").wait_none())
    replacement_id = _container_id("filler_swept-1", generation=1)
    listing = f"filler_swept-1 {replacement_id}\n" if after.endswith("still-listed") else ""
    ssh = _sweep_host(listing)
    rm_ok = ssh.run.side_effect

    async def run(command, **kwargs):
        if command.startswith("/usr/bin/docker rm -fv "):
            assert own_sweep_removals.sent_rm_for(replacement_id)
            if after == "rm-never-sent":
                raise asyncssh.ChannelOpenError(asyncssh.OPEN_CONNECT_FAILED, "SSH connection closed")
            if after.startswith("rm-answer-lost"):
                raise ConnectionResetError("SSH dropped after the rm was sent")
            return await rm_ok(command, **kwargs)
        if after.endswith("listing-failed"):
            return Mock(stdout="", stderr="daemon unreachable", exit_status=1)
        return await rm_ok(command, **kwargs)

    ssh.run = AsyncMock(side_effect=run)

    await svc._remove_replacement_fillers(ssh, {}, "pod_new", {"filler_swept-1": replacement_id})

    assert own_sweep_removals.sent_rm_for(replacement_id) == (after != "rm-never-sent")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["acknowledged", "node-removed-it-first", "replaced-by-a-new-same-name-container", "no-id-listed",
     "acknowledged-then-a-later-target-failed", "rm-sent-answer-lost", "rm-sent-answer-lost-confirmation-failed"],
)  # fmt: skip
async def test_every_listed_id_the_sweep_sends_rm_for_is_ours(svc, monkeypatch, case):
    """The sweep removes by the listed ID and records each ID before its `rm` is sent, whatever the `rm`
    answers: a same-name container created since the listing is not removed under the old ID. A customer's
    create then finds that replacement filler and removes it by its own ID. A name listed without an ID is
    removed by name and recorded as nothing."""
    monkeypatch.setattr("core.utils.wait_fixed", lambda _s: __import__("tenacity").wait_none())
    old, new, other = _container_id("filler_a"), _container_id("filler_a", generation=1), _container_id("filler_b")
    listing = f"filler_a {old}\nfiller_b {other}\n"
    if case == "no-id-listed":
        listing = "filler_a\n"
    after = {  # what `docker ps -a` lists once the rm has run
        "node-removed-it-first": "",
        "replaced-by-a-new-same-name-container": f"filler_a {new}\n",
        "acknowledged-then-a-later-target-failed": f"filler_b {other}\n",
    }.get(case, "")
    listings = iter([listing, after, after, after])
    rms: list[list[str]] = []

    async def run(command, **_kwargs):
        if command.startswith("/usr/bin/docker rm -fv "):
            targets = command.removeprefix("/usr/bin/docker rm -fv ").split()
            rms.append(targets)
            if targets == [new]:
                return Mock(stdout=f"{new}\n", stderr="", exit_status=0)
            if case.startswith("rm-sent-answer-lost"):
                raise ConnectionResetError("SSH dropped after the rm was sent")
            if case in ("node-removed-it-first", "replaced-by-a-new-same-name-container"):
                printed, code = [t for t in targets if t != old], 1  # "No such container: <old>"
            elif case == "acknowledged-then-a-later-target-failed":
                printed, code = [t for t in targets if t != other], 1
            else:
                printed, code = targets, 0
            return Mock(stdout="".join(f"{t}\n" for t in printed), stderr="err", exit_status=code)
        text = next(listings)
        if case == "rm-sent-answer-lost-confirmation-failed" and text is not listing:
            return Mock(stdout="", stderr="daemon unreachable", exit_status=1)
        if [new] in rms:  # the replacement is gone once its own rm ran
            text = "".join(line for line in text.splitlines(keepends=True) if new not in line)
        if "--no-trunc" not in command:  # the names-only confirmation listing
            text = "".join(f"{line.split()[0]}\n" for line in text.splitlines() if line.strip())
        return Mock(stdout=text, stderr="", exit_status=0)

    ssh = Mock()
    ssh.run = AsyncMock(side_effect=run)
    sweep = svc.clean_existing_containers(
        ssh_client=ssh, default_extra={}, pod_name="pod_new", clear_volume=False,
        active_container_names=[], remove_every_filler=True,
    )
    if case == "acknowledged-then-a-later-target-failed":
        with pytest.raises(Exception, match="exit_code 1"):
            await sweep
    elif case == "rm-sent-answer-lost-confirmation-failed":
        # nothing shows the rm took effect: the sweep fails, and its finalizer still records the IDs
        with pytest.raises(ConnectionResetError):
            await sweep
    else:
        await sweep

    replaced = case == "replaced-by-a-new-same-name-container"
    assert all(new not in targets and "filler_a" not in targets for targets in rms[1 : len(rms) - replaced])
    if replaced:
        assert rms[-1] == [new]
    if case == "no-id-listed":
        assert rms == [["filler_a"]]
        assert not own_sweep_removals.sent_rm_for(old)
        return
    assert rms[0] == [old, other]
    assert own_sweep_removals.sent_rm_for(old) and own_sweep_removals.sent_rm_for(other)
    assert own_sweep_removals.sent_rm_for(new) == replaced


def test_the_listing_maps_each_name_to_its_full_container_id():
    good = _container_id("pod_a")
    names, ids = parse_container_listing([f"pod_a {good}", "", "pod_b", f"pod_c {good[:12]}"])
    assert names == ("pod_a", "pod_b", "pod_c") and ids == {"pod_a": good}


def test_the_sweep_registry_keeps_the_newest_ids_up_to_its_cap(monkeypatch):
    registry = type(own_sweep_removals)()
    monkeypatch.setattr(registry, "MAX_IDS", 2)
    first, second, third = (_container_id(n) for n in ("pod_a", "pod_b", "pod_c"))
    registry.mark([first, second])
    registry.mark([first])  # marked again: now the newest
    registry.mark([third])

    assert registry.sent_rm_for(first) and registry.sent_rm_for(third)
    assert not registry.sent_rm_for(second) and not registry.sent_rm_for(None) and not registry.sent_rm_for("")


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
    # keys, bootstrap script written, bootstrap script run, environment: one inspect, one exec each;
    # then the State read before the pod is cached
    assert api.events == ["inspect_container", "exec_create"] * 4 + ["inspect_container"]
    assert len(client.exec_specs) == 4
    svc.redis_service.add_rented_pod.assert_awaited_once()


@pytest.mark.parametrize("exit_code", [137, 143, 1])
def test_a_restarting_container_is_the_images_own_exit(exit_code):
    state = rental_docker_sdk.ContainerStateSnapshot(
        status="restarting", running=False, restarting=True, exit_code=exit_code, restart_count=1,
        error=None, oom_killed=False,
    )  # fmt: skip
    assert container_gone_cause(state) == "exited"
    state.oom_killed = True
    assert container_gone_cause(state) == "oom"


@pytest.mark.asyncio
async def test_a_bootstrap_exec_killed_while_docker_restarts_the_image_is_not_a_node_kill(svc, monkeypatch, caplog):
    api = FakeApiClient()
    restarting = _container_state(status="restarting", running=False, restarting=True, exit_code=137)
    api.container_states = [_RUNNING, _RUNNING, _RUNNING, restarting]
    _bootstrapping_create(svc, monkeypatch, api)
    _exec_exits(0, 0, 137)(api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    # a failed bootstrap exec on a container that was not killed stays the soft failure it was
    assert getattr(result, "failure_step", None) != KILLED_DURING_BOOTSTRAP_STEP
    assert "stopped by the node" not in getattr(result, "detail", "")
    assert _events(caplog) == []
    assert api.exec_created and len(api.exec_created) >= 3


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("after", "failure_step", "cause"),
    [("sigkill", KILLED_DURING_BOOTSTRAP_STEP, "killed"), ("gone", KILLED_DURING_BOOTSTRAP_STEP, "removed"),
     ("image-exited", "jupyter_setup", None), ("still-running", "jupyter_setup", None)],
)  # fmt: skip
async def test_a_kill_during_jupyter_setup_is_killed_during_bootstrap(svc, monkeypatch, caplog, after, failure_step, cause):
    """run_jupyter's shell `docker exec` fails with a plain error when the container is gone: the State
    read before cleanup names a kill; an image's own exit or a live container keeps the step."""
    api = FakeApiClient()
    later = {"sigkill": _SIGKILLED, "image-exited": _container_state(status="exited", running=False, exit_code=0)}
    api.container_states = [_RUNNING, later.get(after, _RUNNING)]
    _bootstrapping_create(svc, monkeypatch, api, skip_ssh_bootstrap=True)
    if after == "gone":
        _gone_at_inspect(api, 2)
    # a Jupyter port off the image's 8888, so the validator runs run_jupyter
    monkeypatch.setattr(
        svc, "generate_portMappings", AsyncMock(return_value=([(22, 20001, 20001), (8889, 20002, 20002)], (8889, 20002)))
    )
    monkeypatch.setattr(svc, "run_jupyter", AsyncMock(side_effect=Exception("Error response from daemon: container is not running")))
    caplog.set_level(logging.WARNING)
    payload = _payload(enable_jupyter=True)

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == failure_step
    svc.run_jupyter.assert_awaited_once()
    if cause is None:
        assert _events(caplog) == []
        return
    (logged,) = _events(caplog)
    assert logged["bootstrap_step"] == "jupyter_setup" and logged["cause"] == cause
    assert "during jupyter_setup" in result.detail
    # the backend reads Docker's "is not running" as the renter's image exiting, not a kill on the node
    assert "is not running" not in result.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(("after", "cause"), [("sigkill", "killed"), ("gone", "removed")])
async def test_a_silent_kill_during_jupyter_setup_is_killed_during_bootstrap(svc, monkeypatch, caplog, after, cause):
    """run_jupyter's last exec does not check its exit status, so a kill mid-exec returns normally: the
    State read after it still names the kill instead of a ContainerCreated for a dead pod."""
    api = FakeApiClient()
    api.container_states = [_RUNNING, _SIGKILLED if after == "sigkill" else _RUNNING]
    _bootstrapping_create(svc, monkeypatch, api, skip_ssh_bootstrap=True)
    if after == "gone":
        _gone_at_inspect(api, 2)
    monkeypatch.setattr(
        svc, "generate_portMappings", AsyncMock(return_value=([(22, 20001, 20001), (8889, 20002, 20002)], (8889, 20002)))
    )
    monkeypatch.setattr(svc, "run_jupyter", AsyncMock(return_value=None))
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload(enable_jupyter=True))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP
    svc.run_jupyter.assert_awaited_once()
    (logged,) = _events(caplog)
    assert logged["bootstrap_step"] == "jupyter_setup" and logged["cause"] == cause


def _inspect_always_fails_from(api: FakeApiClient, n: int) -> None:
    """Inspect call `n` (1-based) and every later one fail with a transient daemon error."""
    real_inspect, looks = api.inspect_container, []

    def inspect(container_name):
        looks.append(container_name)
        if len(looks) >= n:
            raise APIError("500 Server Error: Internal Server Error (\"context deadline exceeded\")")
        return real_inspect(container_name)

    api.inspect_container = inspect


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("after", "outcome"),
    [("sigkill", "killed"), ("oom", "oom"), ("gone", "removed"), ("still-running", None),
     ("image-exited-0", "exited"), ("image-exited-1", "exited"), ("500-then-sigkill", "killed"),
     ("500-then-running", None), ("500-every-time", "unread")],
)  # fmt: skip
async def test_a_kill_after_the_last_bootstrap_exec_is_not_a_created_container(svc, monkeypatch, caplog, after, outcome):
    """No Jupyter, no environment and a skipped SSH bootstrap (ships_sshd) run no exec after the key step:
    the State read before the pod is cached still names a kill, or the image's own exit, instead of a
    ContainerCreated; a transient inspect error is retried, and a State never read fails the create."""
    monkeypatch.setattr("services.docker_service.FINAL_STATE_INSPECT_RETRY_DELAY_S", 0)
    api = FakeApiClient()
    later = {
        "sigkill": _SIGKILLED,
        "500-then-sigkill": _SIGKILLED,
        "oom": _oom_killed_state(),
        "image-exited-0": _container_state(status="exited", running=False, exit_code=0),
        "image-exited-1": _container_state(status="exited", running=False, exit_code=1),
    }
    api.container_states = [_RUNNING, later.get(after, _RUNNING)]
    _bootstrapping_create(svc, monkeypatch, api, skip_ssh_bootstrap=True)
    if after == "gone":
        _gone_at_inspect(api, 2)
    elif after.startswith("500-then"):
        _inspect_fails_at(api, 2)
    elif after == "500-every-time":
        _inspect_always_fails_from(api, 2)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    if outcome is None:
        assert isinstance(result, ContainerCreated)
        assert _events(caplog) == []
        return
    assert isinstance(result, FailedContainerRequest)
    svc.redis_service.add_rented_pod.assert_not_awaited()
    if outcome in ("exited", "unread"):
        # the image's own exit, or a State never read: the step's own failure, not a kill on the node
        assert result.failure_step == "final_state_check"
        assert _events(caplog) == []
        if outcome == "unread":
            assert "could not read the container State before caching the pod (3 attempts)" in result.detail
        else:
            assert "status='exited'" in result.detail
        return
    assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP
    (logged,) = _events(caplog)
    assert logged["bootstrap_step"] == "final_state_check" and logged["cause"] == outcome
    assert "is not running" not in result.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("image_exit", [0, 1])
async def test_a_failed_ssh_bootstrap_on_a_stopped_container_is_not_a_created_container(
    svc, monkeypatch, caplog, image_exit
):
    """The bootstrap's exec ends nonzero while the image's own command exits: the bootstrap returns its
    soft False, and with no environment exec after it the final State check fails the create under
    ssh_bootstrap, with no node-kill event."""
    api = FakeApiClient()
    exited = _container_state(status="exited", running=False, exit_code=image_exit)
    # keys, bootstrap script written, bootstrap script run: running; the State read after the run's exit 1
    api.container_states = [_RUNNING, _RUNNING, _RUNNING, exited]
    _bootstrapping_create(svc, monkeypatch, api)
    _exec_exits(0, 0, 1)(api)
    caplog.set_level(logging.WARNING)

    result = await _create(svc, _payload())

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == "ssh_bootstrap"
    assert "killed_during_bootstrap" not in result.detail and f"exit_code={image_exit}" in result.detail
    assert _events(caplog) == []
    svc.redis_service.add_rented_pod.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_ssh_bootstrap_on_a_running_container_still_creates(svc, monkeypatch, caplog):
    # the inherited soft failure: a bootstrap that ends nonzero on a live container goes on as before
    api = FakeApiClient()
    api.container_states = [_RUNNING]
    _bootstrapping_create(svc, monkeypatch, api)
    _exec_exits(0, 0, 1)(api)

    result = await _create(svc, _payload())

    assert isinstance(result, ContainerCreated), getattr(result, "detail", result)
