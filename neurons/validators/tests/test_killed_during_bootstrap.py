"""A pod killed between `docker run` and the end of the bootstrap reports why.

The readiness inspect that first sees the container gone (`removing` / `exited` / `dead`, or already
"No such container") ends the bootstrap there, carrying the State read at that moment; the create
names the failure `killed_during_bootstrap` with `oom_killed` and the exit code, or
`cancelled_by_delete` when the validator's own delete for that pod is in flight.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
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
    ],
    ids=["ssh-oom", "ssh-sigkill", "ssh-exited-143", "ssh-exited-137", "ssh-exec-137-dead", "ssh-exec-137-404",
         "keys-oom", "keys-removing-exit-0", "keys-dead-exit-1", "keys-exec-130", "keys-exec-137-404", "env-removed",
         "env-exec-137"],
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
    assert f"during {step}" in result.detail and "has no long-running command" not in result.detail
    assert f"(cause={event['cause']} oom_killed=" in result.detail
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


def _swept_at_inspect(api: FakeApiClient, n: int, container_id: str, *, rm: str = "done") -> None:
    """Another create's sweep takes ``container_id`` just before inspect call `n` (1-based). ``rm``:
    "done" (its `rm` has returned), or "ok" / "failed" (still in flight; it ends that way a moment
    later, while this create is classifying), "unanswered" (its SSH call got no answer), "stuck"
    (still in flight when the wait runs out), or "overlap" (a second create sweeps the same ID and
    finds it already gone at once, while the first sweep's acknowledgement comes a moment later)."""
    real_inspect, looks = api.inspect_container, []
    loop = asyncio.get_running_loop()

    def sweep() -> None:
        in_flight = own_sweep_removals.begin([container_id])
        if rm == "stuck":
            return
        if rm == "overlap":
            second = own_sweep_removals.begin([container_id])
            own_sweep_removals.end([container_id], second, removed=[])
        end = functools.partial(
            own_sweep_removals.end, [container_id], in_flight,
            removed=[] if rm in ("failed", "unanswered") else [container_id],
            unanswered=[container_id] if rm == "unanswered" else [],
        )  # fmt: skip
        if rm == "done":
            end()
        else:
            loop.call_later(0.05, end)

    def inspect(container_name):
        # docker-py runs in a worker thread; the sweep lands on the loop before this inspect returns
        looks.append(container_name)
        if len(looks) == n:
            loop.call_soon_threadsafe(sweep)
        return real_inspect(container_name)

    api.inspect_container = inspect


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "swept",
    [
        "during-bootstrap",
        "listed-before-docker-run-returned",
        "rm-in-flight-then-ok",
        "older-same-name-container-before-docker-run",
        "older-same-name-container-during-bootstrap",
        "rm-in-flight-then-failed",
        "rm-sent-answer-lost",
        "rm-still-in-flight-after-the-wait",
        "second-sweep-found-it-gone-first-acknowledged-later",
        "during-bootstrap-oom",
    ],
)
async def test_a_container_another_create_swept_is_not_a_node_kill(svc, monkeypatch, caplog, swept):
    """A customer's create removes every filler on the node, one still bootstrapping included: that
    create finds its container gone, but the validator removed it, so no kill is filed. The sweep is
    matched by container ID, so it counts whenever it listed the containers (even before this
    create's `docker run` returned, as long as it removed this very container) and waits for an
    `rm` still in flight. A sweep of an older container with the same name (a retry of the pod), an
    `rm` that failed, or an OOM (a sweep's `rm -f` never causes one) still files the kill."""
    api = FakeApiClient()
    api.container_states = [_RUNNING, _oom_killed_state() if swept.endswith("-oom") else _SIGKILLED]
    _bootstrapping_create(svc, monkeypatch, api)
    caplog.set_level(logging.WARNING)
    payload = _payload()
    name = f"pod_{payload.pod_id}"
    this, older = _container_id(name), _container_id(name, generation=1)
    if swept == "older-same-name-container-before-docker-run":
        own_sweep_removals.end([older], own_sweep_removals.begin([older]), removed=[older])
    elif swept == "older-same-name-container-during-bootstrap":
        _swept_at_inspect(api, 2, older)
    elif swept == "rm-in-flight-then-ok":
        _swept_at_inspect(api, 2, this, rm="ok")
    elif swept == "rm-in-flight-then-failed":
        _swept_at_inspect(api, 2, this, rm="failed")
    elif swept == "rm-sent-answer-lost":
        # `rm -fv` reached Docker (SIGKILL, removing/137) but SSH dropped before its stdout came back
        _swept_at_inspect(api, 2, this, rm="unanswered")
    elif swept == "rm-still-in-flight-after-the-wait":
        monkeypatch.setattr(own_sweep_removals, "IN_FLIGHT_WAIT_SECONDS", 0.01)
        _swept_at_inspect(api, 2, this, rm="stuck")
    elif swept == "second-sweep-found-it-gone-first-acknowledged-later":
        _swept_at_inspect(api, 2, this, rm="overlap")
    else:
        # "listed-before-docker-run-returned": the ID match makes the listing time irrelevant
        _swept_at_inspect(api, 2, this)

    result = await _create(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    if swept.startswith("older-") or swept.endswith(("-failed", "-oom")):
        assert result.failure_step == KILLED_DURING_BOOTSTRAP_STEP
        (logged,) = _events(caplog)
        assert logged["container_name"] == name
        assert logged["cause"] == ("oom" if swept.endswith("-oom") else "killed")
        return
    assert result.failure_step == "ssh_bootstrap"
    assert "stopped by the node" not in result.detail
    assert _events(caplog) == []
    maybe = swept in ("rm-sent-answer-lost", "rm-still-in-flight-after-the-wait")
    reason = "maybe_removed_by_own_sweep" if maybe else "removed_by_own_sweep"
    assert any(getattr(r.msg, "extra", {}).get("reason") == reason for r in caplog.records)


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
    ids=["listed", "from-probe", "rm-failed", "filler-survived-the-rm", "replacement-after-ack"],
)  # fmt: skip
async def test_the_stale_sweep_records_the_container_ids_it_removed(svc, from_probe, rm_error, survivor):
    swept_id = _container_id("filler_swept-1")
    replacement_id = _container_id("filler_swept-1", generation=1)
    # a filler retry took the name with a new ID after `docker rm` acknowledged the old one
    survivors = {"same": {"filler_swept-1": swept_id}, "replacement": {"filler_swept-1": replacement_id}}.get(
        survivor, {}
    )
    ssh = _sweep_host(f"filler_swept-1 {swept_id}\npod_keep {_container_id('pod_keep')}\n")
    probe = (
        Mock(container_names=("filler_swept-1", "pod_keep"), container_ids={"filler_swept-1": swept_id})
        if from_probe else None
    )  # fmt: skip

    async def remove(_ssh, _extra, _pod, _names, targets, _every, acknowledged, _unanswered):
        if rm_error is not None:
            raise rm_error
        acknowledged.update(targets)
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
        assert not await own_sweep_removals.removed_by_us(swept_id)
        return
    assert await sweep == ["filler_swept-1"]
    # the same ID still on the host after the rm (_confirm_fillers_removed saw it) was not removed by us;
    # a new ID under the name does not undo the acknowledged removal of the old one
    assert await own_sweep_removals.removed_by_us(swept_id) == (survivor != "same")
    # the customer's create removes the replacement by its own ID and records it as well
    assert await own_sweep_removals.removed_by_us(replacement_id) == (survivor == "replacement")
    assert ssh.rms == ([[replacement_id]] if survivor == "replacement" else [])
    assert not await own_sweep_removals.removed_by_us(None)
    # the IDs come with the listing: no `docker inspect` round trip, from the probe or not
    assert all("docker inspect" not in c.args[0] for c in ssh.run.await_args_list)
    # the replacement: its rm and the confirmation that follows it
    assert ssh.run.await_count == (0 if from_probe else 1) + (2 if survivor == "replacement" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("after", ["gone", "still-listed", "listing-failed"])
async def test_a_replacement_filler_counts_as_ours_only_once_confirmed_gone(svc, after):
    """`docker rm` acknowledging the replacement's ID is not enough: the confirmation must no longer list it."""
    replacement_id = _container_id("filler_swept-1", generation=1)
    listing = f"filler_swept-1 {replacement_id}\n" if after == "still-listed" else ""
    ssh = _sweep_host(listing)
    if after == "listing-failed":
        rm_only = ssh.run.side_effect

        async def run(command, **kwargs):
            if command.startswith("/usr/bin/docker rm -fv "):
                return await rm_only(command, **kwargs)
            return Mock(stdout="", stderr="daemon unreachable", exit_status=1)

        ssh.run = AsyncMock(side_effect=run)

    await svc._remove_replacement_fillers(ssh, {}, "pod_new", {"filler_swept-1": replacement_id})

    assert await own_sweep_removals.removed_by_us(replacement_id) == (after == "gone")
    assert not own_sweep_removals.maybe_removed_by_us(replacement_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["acknowledged", "node-removed-it-first", "replaced-by-a-new-same-name-container", "no-id-listed",
     "acknowledged-then-a-later-target-failed", "rm-sent-answer-lost"],
)  # fmt: skip
async def test_only_an_id_docker_rm_printed_back_is_recorded_as_ours(svc, monkeypatch, case):
    """The sweep removes by the listed ID and records an ID only when `docker rm` printed it back:
    an empty listing after a failed rm is not proof, and a same-name container created since the
    listing is not removed under the old ID. A customer's create then finds that replacement filler and
    removes it by its own ID. An rm whose SSH call got no answer leaves its IDs neither ours nor the node's."""
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
            if case == "rm-sent-answer-lost":
                raise ConnectionResetError("SSH dropped after the rm was sent")
            if case in ("node-removed-it-first", "replaced-by-a-new-same-name-container"):
                printed, code = [t for t in targets if t != old], 1  # "No such container: <old>"
            elif case == "acknowledged-then-a-later-target-failed":
                printed, code = [t for t in targets if t != other], 1
            else:
                printed, code = targets, 0
            return Mock(stdout="".join(f"{t}\n" for t in printed), stderr="err", exit_status=code)
        text = next(listings)
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
    else:
        await sweep

    replaced = case == "replaced-by-a-new-same-name-container"
    assert all(new not in targets and "filler_a" not in targets for targets in rms[1 : len(rms) - replaced])
    if replaced:
        assert rms[-1] == [new]
    if case == "no-id-listed":
        assert rms == [["filler_a"]]
        assert not await own_sweep_removals.removed_by_us(old)
        return
    assert rms[0] == [old, other]
    assert await own_sweep_removals.removed_by_us(old) == (case in ("acknowledged", "acknowledged-then-a-later-target-failed"))
    assert await own_sweep_removals.removed_by_us(new) == replaced
    assert await own_sweep_removals.removed_by_us(other) == (
        case not in ("acknowledged-then-a-later-target-failed", "rm-sent-answer-lost")
    )
    lost = case == "rm-sent-answer-lost"
    assert own_sweep_removals.maybe_removed_by_us(old) == lost and own_sweep_removals.maybe_removed_by_us(other) == lost


def test_the_listing_maps_each_name_to_its_full_container_id():
    good = _container_id("pod_a")
    names, ids = parse_container_listing([f"pod_a {good}", "", "pod_b", f"pod_c {good[:12]}"])
    assert names == ("pod_a", "pod_b", "pod_c") and ids == {"pod_a": good}


@pytest.mark.asyncio
async def test_an_rm_still_in_flight_is_waited_for_and_a_stuck_one_is_only_maybe_ours(monkeypatch):
    registry = type(own_sweep_removals)()
    ok, failed, stuck = (_container_id(n) for n in ("pod_ok", "pod_failed", "pod_stuck"))
    for container_id, removed in ((ok, True), (failed, False)):
        sweep = registry.begin([container_id])
        end = functools.partial(registry.end, [container_id], sweep, removed=[container_id] if removed else [])
        asyncio.get_running_loop().call_later(0.01, end)
    registry.begin([stuck])
    monkeypatch.setattr(registry, "IN_FLIGHT_WAIT_SECONDS", 0.05)

    assert await registry.removed_by_us(ok)
    assert not await registry.removed_by_us(failed) and not registry.maybe_removed_by_us(failed)
    assert not await registry.removed_by_us(stuck) and registry.maybe_removed_by_us(stuck)


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["acknowledged", "failed", "stuck"])
async def test_every_open_sweep_of_an_id_is_waited_for(monkeypatch, first):
    """Two creates sweep the same filler; the second finds it already gone while the first `rm` runs."""
    registry = type(own_sweep_removals)()
    monkeypatch.setattr(registry, "IN_FLIGHT_WAIT_SECONDS", 0.05)
    filler = _container_id("filler_a")
    first_sweep = registry.begin([filler])
    registry.end([filler], registry.begin([filler]), removed=[])
    if first != "stuck":
        removed = [filler] if first == "acknowledged" else []
        asyncio.get_running_loop().call_later(0.01, lambda: registry.end([filler], first_sweep, removed=removed))

    assert await registry.removed_by_us(filler) is (first == "acknowledged")
    assert registry.maybe_removed_by_us(filler) is (first == "stuck")


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
