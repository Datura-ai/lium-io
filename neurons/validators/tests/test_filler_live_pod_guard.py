"""E-187 (DAH-3706 family): a filler never starts on GPUs a RUNNING customer pod holds — the validator's half.

The platform judges the launch against its rows; this is the host-truth check right before the
filler's `docker run`: `docker inspect` every RUNNING `pod_*` container for the GPUs it was given
(HostConfig.DeviceRequests) and refuse the filler when the sets intersect — BEFORE the create's
container sweep, so the customer's container is never the thing removed to make room.

Two outcomes, two failure steps (the platform keys on them — lium-platform#608):
- `filler_live_pod_guard`            a confirmed overlap: event FILLER_START_REFUSED_LIVE_POD, the
                                     platform closes the run STOPPED (a lost race, no backoff);
- `filler_live_pod_guard_unreadable` the host could not be read (docker ps / inspect failed, hung,
                                     or printed a line the parser cannot trust): NO event, the
                                     platform keeps today's FAILED + backoff path.

Failing-first: the create tests read only pre-existing names — on main the filler is CREATED beside
the pod. The parser / overlap unit tests import the fix's module inside the test body.

Round 2 (fresh-reader verdict on a2f3a11): `docker inspect --format` does NOT translate a literal
`\\t` (only `docker ps --format` does), so the r1 template printed a backslash and a t and the
parser saw no separator. The fixtures below are REAL lines captured from `docker inspect --format
'{{.Name}}{{"\\t"}}{{json .HostConfig.DeviceRequests}}'` on docker 29.1.3 (see REAL_INSPECT_*).
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated, FailedContainerRequest, WorkloadKind
from services.docker_service import DockerService

from tests.test_deploy_optimizations import (
    _created_run_spec,
    _patch_happy,
    _payload,
    _run,
    _ssh_client,
    _ssh_run_cmds,
)


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


EVENT = "FILLER_START_REFUSED_LIVE_POD"
GUARD_STEP = "filler_live_pod_guard"
UNREADABLE_STEP = "filler_live_pod_guard_unreadable"
HOST_READ_TIMEOUT_SECONDS = 30  # the same bound every neighbouring host read on the create path uses

# Captured verbatim on docker 29.1.3 (Server 29.1.3, this lane's VM, 20 Sep 2026):
#   docker create --name pod_e187tab --gpus '"device=GPU-aaaa…,GPU-bbbb…"' alpine:3.19 sleep 60
#   docker create --name pod_e187all --gpus all alpine:3.19 sleep 60
#   docker create --name pod_e187nogpu alpine:3.19 sleep 60
#   docker inspect --format '{{.Name}}{{"\t"}}{{json .HostConfig.DeviceRequests}}' <the three>
# `cat -A` shows ^I (a real tab) between the name and the JSON. The r1 template
# '{{.Name}}\t{{json …}}' printed the two characters '\' 't' instead (REAL_INSPECT_R1_LITERAL).
REAL_INSPECT_PINNED = (
    '/pod_e187tab\t[{"Driver":"","Count":0,"DeviceIDs":["GPU-aaaa1111-0000-0000-0000-000000000001",'
    '"GPU-bbbb2222-0000-0000-0000-000000000002"],"Capabilities":[["gpu"]],"Options":{}}]'
)
REAL_INSPECT_WHOLE_HOST = '/pod_e187all\t[{"Driver":"","Count":-1,"DeviceIDs":null,"Capabilities":[["gpu"]],"Options":{}}]'
REAL_INSPECT_NO_GPU = "/pod_e187nogpu\tnull"
REAL_INSPECT_R1_LITERAL = (
    '/pod_e187tab\\t[{"Driver":"","Count":0,"DeviceIDs":["GPU-aaaa1111-0000-0000-0000-000000000001",'
    '"GPU-bbbb2222-0000-0000-0000-000000000002"],"Capabilities":[["gpu"]],"Options":{}}]'
)

# Shorter hand-written lines in the same (real) shape for the create tests.
POD_ON_0_1 = '/pod_cust1\t[{"Driver":"nvidia","Count":0,"DeviceIDs":["GPU-0","GPU-1"],"Capabilities":[["gpu"]],"Options":{}}]'
POD_WHOLE_HOST = '/pod_whole\t[{"Driver":"nvidia","Count":-1,"DeviceIDs":null,"Capabilities":[["gpu"]],"Options":{}}]'


def _is_ps(cmd: str) -> bool:
    return "docker ps" in cmd and "name=pod_" in cmd and "DeviceRequests" not in cmd


def _is_inspect(cmd: str) -> bool:
    return "docker inspect" in cmd and "DeviceRequests" in cmd


def _result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    result = AsyncMock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


def _host_with_running_pods(*inspect_lines: str, ps_exit: int = 0, inspect_exit: int = 0, hang: str | None = None):
    """An ssh client whose live-pod reads answer with these inspect lines.

    `docker ps -q` returns one fake id per line; `docker inspect <ids>` returns the lines. `ps_exit` /
    `inspect_exit` make either command fail the way dockerd does (non-zero, error on stderr, empty
    stdout); `hang="ps"|"inspect"` makes that command raise asyncio.TimeoutError.
    """
    client = _ssh_client()
    ids = [f"{index:012x}" for index in range(1, len(inspect_lines) + 1)]

    def _side(cmd, *args, **kwargs):
        if _is_ps(cmd):
            if hang == "ps":
                raise asyncio.TimeoutError()
            if ps_exit:
                return _result(ps_exit, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock")
            return _result(0, "\n".join(ids) + ("\n" if ids else ""))
        if _is_inspect(cmd):
            if hang == "inspect":
                raise asyncio.TimeoutError()
            if inspect_exit:
                return _result(inspect_exit, "", "error: no such object: deadbeef")
            return _result(0, "\n".join(inspect_lines) + "\n" if inspect_lines else "")
        return _result(0, "")

    client.run = AsyncMock(side_effect=_side)
    return client


def _filler_payload(gpu_uuids: list[str]):
    return _payload(workload_kind=WorkloadKind.FILLER, gpu_uuids=gpu_uuids, active_container_names=[])


def _events(caplog) -> list[dict]:
    return [
        record.msg.extra
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event") == EVENT
    ]


def _live_pod_read_cmds(ssh_client) -> list[str]:
    return [cmd for cmd in _ssh_run_cmds(ssh_client) if _is_ps(cmd) or _is_inspect(cmd)]


def _live_pod_read_calls(ssh_client) -> list:
    return [call for call in ssh_client.run.await_args_list if _is_ps(call.args[0]) or _is_inspect(call.args[0])]


# ---------------------------------------------------------------------------------------------------
# create_container
# ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filler_is_refused_when_a_running_pod_holds_one_of_its_gpus(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)
    payload = _filler_payload(["GPU-1", "GPU-2"])

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP
    assert result.pod_id == payload.pod_id and result.workload_kind == WorkloadKind.FILLER
    assert "pod_cust1" in (result.detail or "") and "GPU-1" in (result.detail or "")
    assert svc.rental_docker_client_factory.client.run_specs == [], "no docker run"
    # the customer's container is never swept to make room for a filler
    svc.clean_existing_containers.assert_not_awaited()
    [event] = _events(caplog)
    # one key for the validator uuid in BOTH halves (lium-platform#608 writes executor_uuid too; its
    # executor_id is the DB PK, which this side does not know)
    assert event["executor_uuid"] == payload.executor_id
    assert "executor_id" not in event
    assert event["filler_pod_id"] == payload.pod_id
    assert event["pod_containers"] == ["pod_cust1"]
    assert event["gpu_overlap"] == ["GPU-1"]


@pytest.mark.asyncio
async def test_filler_starts_on_the_gpus_no_running_pod_holds(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)
    payload = _filler_payload([f"GPU-{index}" for index in range(2, 8)])

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert _created_run_spec(svc) is not None
    assert _events(caplog) == []
    cmds = _live_pod_read_cmds(ssh)
    assert len(cmds) == 2 and _is_ps(cmds[0]) and _is_inspect(cmds[1]), "one ps, one inspect of its ids"
    assert "000000000001" in cmds[1], "inspect is given the ids ps returned"


@pytest.mark.asyncio
async def test_a_whole_host_pod_refuses_every_filler(svc, monkeypatch):
    ssh = _host_with_running_pods(POD_WHOLE_HOST)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _filler_payload(["GPU-5"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP


@pytest.mark.asyncio
async def test_a_whole_node_filler_is_refused_by_any_running_pod(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload([]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP
    [event] = _events(caplog)
    assert event["gpu_overlap"] == ["GPU-0", "GPU-1"]


@pytest.mark.asyncio
async def test_a_host_with_no_running_pod_lets_the_filler_through(svc, monkeypatch):
    ssh = _host_with_running_pods()
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _filler_payload(["GPU-0"]))

    assert isinstance(result, ContainerCreated), result
    cmds = _live_pod_read_cmds(ssh)
    assert len(cmds) == 1 and _is_ps(cmds[0]), "an empty ps (exit 0) is 'no pod'; nothing to inspect"


@pytest.mark.asyncio
async def test_a_customer_create_never_runs_the_filler_guard(svc, monkeypatch):
    # The rent path has its own rule (lium-io#1417 removes every filler_*); a pod create must not be
    # refused, or even slowed, by a listing meant for fillers.
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, _payload(gpu_uuids=["GPU-1"]))

    assert isinstance(result, ContainerCreated), result
    assert _live_pod_read_cmds(ssh) == []


@pytest.mark.asyncio
async def test_a_real_docker_inspect_line_refuses_the_filler(svc, monkeypatch, caplog):
    # The line dockerd actually prints (docker 29.1.3, template with the {{"\t"}} action).
    ssh = _host_with_running_pods(REAL_INSPECT_PINNED, REAL_INSPECT_NO_GPU)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload(["GPU-bbbb2222-0000-0000-0000-000000000002", "GPU-cccc"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == GUARD_STEP
    [event] = _events(caplog)
    assert event["pod_containers"] == ["pod_e187tab"]
    assert event["gpu_overlap"] == ["GPU-bbbb2222-0000-0000-0000-000000000002"]


# ---------------------------------------------------------------------------------------------------
# unreadable host: fail closed under its OWN step, no overlap event
# ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_docker_ps_failure_refuses_the_filler_as_unreadable(svc, monkeypatch, caplog):
    # What a dead dockerd really produces: `docker ps` exits 1 with the error on stderr and NOTHING
    # on stdout. r1 piped ps into `xargs -r docker inspect`, whose exit (0) hid this: fail OPEN.
    ssh = _host_with_running_pods(POD_ON_0_1, ps_exit=1)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload(["GPU-0"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == UNREADABLE_STEP
    assert "Cannot connect to the Docker daemon" in (result.detail or "")
    assert svc.rental_docker_client_factory.client.run_specs == [], "no docker run"
    svc.clean_existing_containers.assert_not_awaited()
    assert _events(caplog) == [], "an unreadable host is not an overlap event"


@pytest.mark.asyncio
async def test_a_docker_inspect_failure_refuses_the_filler_as_unreadable(svc, monkeypatch, caplog):
    ssh = _host_with_running_pods(POD_ON_0_1, inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload(["GPU-5"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == UNREADABLE_STEP
    assert _events(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hang", ["ps", "inspect"])
async def test_a_hung_host_read_refuses_the_filler_as_unreadable(svc, monkeypatch, caplog, hang):
    ssh = _host_with_running_pods(POD_ON_0_1, hang=hang)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload(["GPU-5"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == UNREADABLE_STEP
    assert "timed out" in (result.detail or "")
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_every_host_read_of_the_guard_is_bounded_by_a_timeout(svc, monkeypatch):
    ssh = _host_with_running_pods(POD_ON_0_1)
    _patch_happy(svc, monkeypatch, ssh)

    await _run(svc, _filler_payload(["GPU-5"]))

    calls = _live_pod_read_calls(ssh)
    assert len(calls) == 2
    assert all(call.kwargs.get("timeout") == HOST_READ_TIMEOUT_SECONDS for call in calls), calls


@pytest.mark.asyncio
async def test_a_line_without_the_separator_refuses_the_filler_as_unreadable(svc, monkeypatch, caplog):
    # The r1 template's output: a literal backslash-t. Whatever printed it, the listing cannot be
    # trusted — refuse under the unreadable step, never "no GPU claim".
    ssh = _host_with_running_pods(REAL_INSPECT_R1_LITERAL)
    _patch_happy(svc, monkeypatch, ssh)

    with caplog.at_level(logging.WARNING):
        result = await _run(svc, _filler_payload(["GPU-zzzz"]))

    assert isinstance(result, FailedContainerRequest), result
    assert result.failure_step == UNREADABLE_STEP
    assert _events(caplog) == []


# ---------------------------------------------------------------------------------------------------
# the command / parser / overlap (pure)
# ---------------------------------------------------------------------------------------------------


def test_the_inspect_template_emits_the_tab_through_a_template_action():
    # Go text/template: a string literal inside an action is interpreted ({{"\t"}} → TAB); text
    # OUTSIDE actions is copied verbatim, so '\t' there stays two characters. `docker ps --format`
    # pre-processes \t itself; `docker inspect --format` does not.
    from services.filler_live_pod_guard import LIVE_POD_GPU_SETS_CMD_PREFIX, LIVE_POD_IDS_CMD

    assert '{{.Name}}{{"\\t"}}{{json .HostConfig.DeviceRequests}}' in LIVE_POD_GPU_SETS_CMD_PREFIX
    assert "\\t{{json" not in LIVE_POD_GPU_SETS_CMD_PREFIX, "a bare \\t outside an action is printed literally"
    assert "|" not in LIVE_POD_IDS_CMD and "xargs" not in LIVE_POD_IDS_CMD, "ps runs alone so its exit is seen"
    assert "--filter status=running" in LIVE_POD_IDS_CMD and "--filter name=pod_" in LIVE_POD_IDS_CMD


def test_parse_live_pod_gpu_sets_reads_real_inspect_lines():
    from services.filler_live_pod_guard import parse_live_pod_gpu_sets

    pods = parse_live_pod_gpu_sets("\n".join([REAL_INSPECT_PINNED, REAL_INSPECT_WHOLE_HOST, REAL_INSPECT_NO_GPU, ""]))

    assert pods == {
        "pod_e187tab": frozenset(
            {"GPU-aaaa1111-0000-0000-0000-000000000001", "GPU-bbbb2222-0000-0000-0000-000000000002"}
        ),
        "pod_e187all": None,  # --gpus all: Count=-1, DeviceIDs=null → the whole host
        # pod_e187nogpu: created without --gpus (DeviceRequests null) → no GPU claim, skipped
    }


def test_parse_live_pod_gpu_sets_skips_non_pods():
    from services.filler_live_pod_guard import parse_live_pod_gpu_sets

    listing = "\n".join(
        [
            POD_ON_0_1,
            '/filler_abc\t[{"Driver":"nvidia","Count":0,"DeviceIDs":["GPU-7"],"Capabilities":[["gpu"]]}]',
            "/container_health_check\t[]",
            "/mypod_x\tnull",
            "",
        ]
    )

    assert parse_live_pod_gpu_sets(listing) == {"pod_cust1": frozenset({"GPU-0", "GPU-1"})}


@pytest.mark.parametrize(
    "line",
    [
        REAL_INSPECT_R1_LITERAL,  # no separator (the r1 template's literal \t)
        "/pod_broken\tnot-json",  # separator, unparseable claim
        "/pod_broken\t[42]",  # separator, a claim of the wrong shape
    ],
)
def test_parse_live_pod_gpu_sets_fails_closed_on_a_line_it_cannot_trust(line):
    from services.filler_live_pod_guard import FillerLivePodListingUnreadableError, parse_live_pod_gpu_sets

    with pytest.raises(FillerLivePodListingUnreadableError):
        parse_live_pod_gpu_sets(line + "\n" + POD_ON_0_1)


def test_find_live_pod_gpu_overlap_pinned_sets():
    from services.filler_live_pod_guard import find_live_pod_gpu_overlap

    pods = {"pod_a": frozenset({"GPU-0", "GPU-1"}), "pod_b": frozenset({"GPU-4"})}
    assert find_live_pod_gpu_overlap(["GPU-2", "GPU-3"], pods) is None
    overlap = find_live_pod_gpu_overlap(["GPU-1", "GPU-4"], pods)
    assert overlap is not None
    assert overlap.pod_containers == ["pod_a", "pod_b"]
    assert overlap.gpu_overlap == ["GPU-1", "GPU-4"]


def test_find_live_pod_gpu_overlap_whole_host_on_either_side():
    from services.filler_live_pod_guard import find_live_pod_gpu_overlap

    assert find_live_pod_gpu_overlap(["GPU-5"], {"pod_w": None}).pod_containers == ["pod_w"]
    whole_filler = find_live_pod_gpu_overlap([], {"pod_a": frozenset({"GPU-0"})})
    assert whole_filler is not None and whole_filler.gpu_overlap == ["GPU-0"]
    assert find_live_pod_gpu_overlap([], {}) is None
