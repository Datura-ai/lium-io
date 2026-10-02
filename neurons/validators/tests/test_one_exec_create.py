"""LIUM-65 — volume create → `docker run` → running check as one exec of a fixed script.

Covered here:
- the script itself, run by python3 against a fake dockerd on a unix socket: the happy path's
  step lines and calls, each step's failure with what it removes, the refusals that create
  nothing (missing rental network, a name already taken, unexpected request fields), SIGTERM
  mid-run removes the attempt's own work, and `settle` adopts or removes only its attempt's;
- hostile values (quotes, `$()`, newlines) in image / env / volume / container name reach the host
  only inside the JSON: the command line is a constant, the fake dockerd receives them verbatim;
- the container body is the one docker-py posts for the same run spec;
- `create_container`: the step lines become today's `stream_log` texts and profiler rows; each
  step's failure keeps today's failure step; port collision / no python3 / missing network take
  today's path; a lost reply is settled first (adopt, or today's path only after a confirmed
  removal, else the create fails); the creates that never use it.
"""

from __future__ import annotations

import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import docker
import pytest
from core.config import settings
from payload_models.payloads import (
    ContainerCreated,
    ExternalVolumeInfo,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    ProfilerStepName,
    WorkloadKind,
)
from services import docker_service as ds
from services.docker_service import DockerService, VolumeHostProbe
from services.one_exec_create import (
    ONE_EXEC_CREATE_COMMAND,
    ONE_EXEC_CREATE_SCRIPT,
    build_one_exec_create_request,
    run_one_exec_create,
)
from services.rental_docker_sdk import (
    RENTAL_NETWORK_NAME,
    ContainerRunSpec,
    ContainerUlimit,
    DeviceMount,
    GpuDeviceRequest,
    PortBinding,
    RentalDockerSdkClient,
    VolumeMount,
    build_container_create_body,
)
from test_deploy_optimizations import _payload as _deploy_payload
from test_deploy_optimizations import _run as _run_create_container
from test_deploy_optimizations import _ssh_client as _deploy_ssh_client
from test_prerun_host_probe import _wire

_CONTAINER_ID = "c0ffee"
_HOSTILE = "x'\"; $(touch {marker}) `touch {marker}`\nrm -rf / #"


# ------------------------------------------------------------------
# the script against a fake dockerd
# ------------------------------------------------------------------


class _FakeDockerd:
    """Answers the Engine API calls the script makes; `refuse` names the step that fails."""

    def __init__(self, *, refuse: str | None = None, running: bool = True, vanish: bool = False,
                 network_missing: bool = False, volume_exists: bool = False):
        self.refuse = refuse
        self.running = running
        self.vanish = vanish
        self.network_missing = network_missing
        self.volume_exists = volume_exists
        self.calls: list[tuple[str, str, object]] = []

    def answer(self, method: str, path: str, body):
        self.calls.append((method, path, body))
        container = f"/containers/{_CONTAINER_ID}"
        if method == "GET" and path.startswith("/networks/"):
            if self.network_missing:
                return 404, {"message": "network lium-rentals not found"}
            return 200, {"Driver": "bridge", "Options": {"com.docker.network.bridge.enable_icc": "false"}}
        if method == "GET" and path.startswith("/volumes/"):
            return (200, {}) if self.volume_exists else (404, {"message": "no such volume"})
        if method == "GET" and path.startswith("/containers/container_"):
            return 404, {"message": "No such container"}
        if method == "POST" and path == "/volumes/create":
            return (500, {"message": "disk full"}) if self.refuse == "volume" else (201, {"Name": body["Name"]})
        if method == "POST" and path.startswith("/containers/create?name="):
            return (500, {"message": "no such image"}) if self.refuse == "create" else (201, {"Id": _CONTAINER_ID})
        if method == "POST" and path == f"{container}/start":
            if self.refuse == "start":
                return 500, {"message": "Bind for 0.0.0.0:20001 failed: port is already allocated"}
            return 204, None
        if method == "GET" and path == f"{container}/json":
            if self.vanish:
                return 404, {"message": "No such container"}
            return 200, {"State": {"Running": self.running, "Status": "exited", "ExitCode": 1,
                                   "OOMKilled": False, "Error": ""}}
        if method == "GET" and path.startswith(f"{container}/logs"):
            text = b"entrypoint: boom\n"
            return 200, b"\x02\x00\x00\x00" + len(text).to_bytes(4, "big") + text
        if method == "DELETE":
            return 204, None
        return 404, {"message": f"unexpected {method} {path}"}


class _Handler(BaseHTTPRequestHandler):
    def _serve(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        status, payload = self.server.fake.answer(self.command, self.path, json.loads(raw) if raw else None)
        data = payload if isinstance(payload, bytes) else (b"" if payload is None else json.dumps(payload).encode())
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_DELETE = _serve

    def address_string(self):
        return "docker.sock"

    def log_message(self, *_):
        return None


class _UnixHttpServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@pytest.fixture
def fake_dockerd_socket():
    # macOS caps a unix socket path at 104 bytes; pytest's tmp_path is longer
    directory = tempfile.mkdtemp(dir="/tmp", prefix="oec-")
    path = os.path.join(directory, "docker.sock")
    servers = []

    def start(fake: _FakeDockerd) -> str:
        server = _UnixHttpServer(path, _Handler)
        server.fake = fake
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return path

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()
    shutil.rmtree(directory)


def _script(socket_path: str, running_check_sec: float = 0.3) -> str:
    # the attempt's lock and record go next to the socket instead of the host's /tmp
    return (
        ONE_EXEC_CREATE_SCRIPT.replace('"/var/run/docker.sock"', repr(socket_path))
        .replace("RUNNING_CHECK_SEC = 10", f"RUNNING_CHECK_SEC = {running_check_sec}")
        .replace('"/tmp/lium-one-exec-"', repr(os.path.join(os.path.dirname(socket_path), "attempt-")))
    )


def _run_script(socket_path: str, request: dict) -> tuple[list[dict], int]:
    completed = subprocess.run(
        [sys.executable, "-c", _script(socket_path)], input=json.dumps(request),
        capture_output=True, text=True, timeout=30,
    )
    return [json.loads(line) for line in completed.stdout.splitlines()], completed.returncode


def _settle(socket_path: str, create_request: dict, attempt: str | None = None) -> dict:
    request = {"action": "settle", "attempt": attempt or create_request["attempt"],
               "container_name": create_request["container_name"]}
    lines, _ = _run_script(socket_path, request)
    return lines[-1]


def _run_spec(**over) -> ContainerRunSpec:
    base = dict(
        image="daturaai/pytorch:1.0.0",
        name="container_pod-1",
        environment={"JUPYTER_PASSWORD": "token", "NVIDIA_DRIVER_CAPABILITIES": "all"},
        ports=(PortBinding(22, 20001), PortBinding(8888, 20002)),
        volumes=(VolumeMount("volume_pod-1", "/lium-cipher"),),
        restart_policy="unless-stopped",
        runtime="sysbox-runc",
        cap_add=("NET_ADMIN",),
        sysctls={"net.ipv4.conf.all.src_valid_mark": "1"},
        ulimits=(ContainerUlimit("nofile", 1024, 4096),),
        devices=(DeviceMount("/dev/nvidia0"), DeviceMount("/dev/fuse")),
        device_requests=(GpuDeviceRequest(device_ids=("GPU-1",)),),
        cpu_count=4,
        memory_gb=16,
        storage_limit_gb=20,
        shm_size="8g",
        network=RENTAL_NETWORK_NAME,
    )
    base.update(over)
    return ContainerRunSpec(**base)


def _request(run_spec: ContainerRunSpec | None = None, volume_name: str = "volume_pod-1") -> dict:
    return build_one_exec_create_request(
        run_spec=run_spec or _run_spec(),
        volume_name=volume_name,
        volume_driver="vloopback",
        volume_driver_opts={"size": "10g", "sparse": "true"},
        volume_timeout_s=10,
    )


def _steps(lines: list[dict]) -> list[str]:
    return [line["step"] for line in lines]


def test_script_happy_path_prints_every_step_and_posts_the_request_bodies(fake_dockerd_socket):
    fake = _FakeDockerd()
    request = _request()

    lines, exit_status = _run_script(fake_dockerd_socket(fake), request)

    assert exit_status == 0
    assert _steps(lines) == ["network", "volume_created", "container_created", "started", "running"]
    assert all(isinstance(line["ms"], int) for line in lines)
    assert [(method, path) for method, path, _ in fake.calls] == [
        ("GET", f"/networks/{RENTAL_NETWORK_NAME}"),
        ("GET", "/volumes/volume_pod-1"),
        ("GET", "/containers/container_pod-1/json"),
        ("POST", "/volumes/create"),
        ("POST", "/containers/create?name=container_pod-1"),
        ("POST", f"/containers/{_CONTAINER_ID}/start"),
        ("GET", f"/containers/{_CONTAINER_ID}/json"),
    ]
    assert fake.calls[3][2] == request["volume"]
    assert fake.calls[4][2] == request["container"]


@pytest.mark.parametrize(
    ("fake", "failed_step", "removed"),
    [
        # the name was absent before the attempt, so whatever answers to it now is its own
        (_FakeDockerd(refuse="volume"), "volume_created", ["/volumes/volume_pod-1"]),
        (_FakeDockerd(refuse="create"), "container_created",
         ["/containers/container_pod-1?force=1&v=1", "/volumes/volume_pod-1"]),
        (_FakeDockerd(refuse="start"), "started",
         [f"/containers/{_CONTAINER_ID}?force=1&v=1", "/volumes/volume_pod-1"]),
        (_FakeDockerd(running=False), "running",
         [f"/containers/{_CONTAINER_ID}?force=1&v=1", "/volumes/volume_pod-1"]),
        (_FakeDockerd(vanish=True), "running",
         [f"/containers/{_CONTAINER_ID}?force=1&v=1", "/volumes/volume_pod-1"]),
    ],
    ids=["volume", "create", "start", "not-running", "vanished"],
)
def test_script_step_failure_removes_what_it_created_and_names_the_step(
    fake_dockerd_socket, fake, failed_step, removed
):
    lines, exit_status = _run_script(fake_dockerd_socket(fake), _request())

    assert exit_status == 1
    failure = lines[-1]
    assert failure["step"] == "failed"
    assert failure["failed_step"] == failed_step
    assert failure["cleaned"] is True
    assert [path for method, path, _ in fake.calls if method == "DELETE"] == removed


def test_script_running_failure_reports_the_state_and_the_logs(fake_dockerd_socket):
    lines, _ = _run_script(fake_dockerd_socket(_FakeDockerd(running=False)), _request())

    assert lines[-1]["state"]["ExitCode"] == 1
    assert lines[-1]["logs"] == "entrypoint: boom\n"
    assert "vanished" not in lines[-1]


def test_script_vanished_container_is_reported(fake_dockerd_socket):
    lines, _ = _run_script(fake_dockerd_socket(_FakeDockerd(vanish=True)), _request())

    assert lines[-1]["vanished"] is True


def test_script_missing_network_fails_before_creating_anything(fake_dockerd_socket):
    fake = _FakeDockerd(network_missing=True)

    lines, exit_status = _run_script(fake_dockerd_socket(fake), _request())

    assert exit_status == 1
    assert lines == [{"step": "failed", "failed_step": "network", "error": lines[0]["error"], "cleaned": True}]
    assert [method for method, _, _ in fake.calls] == ["GET"]


def test_script_never_takes_over_a_volume_that_was_there_before(fake_dockerd_socket):
    fake = _FakeDockerd(volume_exists=True)

    lines, exit_status = _run_script(fake_dockerd_socket(fake), _request())

    assert exit_status == 1
    assert lines[-1]["failed_step"] == "volume_exists"
    assert [method for method, _, _ in fake.calls] == ["GET", "GET"]


def test_script_sigterm_mid_run_removes_its_own_work_and_settle_confirms(fake_dockerd_socket):
    fake = _FakeDockerd(running=False)
    socket_path = fake_dockerd_socket(fake)
    request = _request()
    process = subprocess.Popen(
        [sys.executable, "-c", _script(socket_path, running_check_sec=30)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    process.stdin.write(json.dumps(request))
    process.stdin.close()
    while json.loads(process.stdout.readline())["step"] != "started":
        pass

    process.terminate()

    assert process.wait(timeout=10) == 3
    assert [path for method, path, _ in fake.calls if method == "DELETE"] == [
        f"/containers/{_CONTAINER_ID}?force=1&v=1", "/volumes/volume_pod-1",
    ]
    assert _settle(socket_path, request) == {"step": "settled", "result": "gone"}


def test_settle_adopts_the_attempts_running_container(fake_dockerd_socket):
    fake = _FakeDockerd()
    socket_path = fake_dockerd_socket(fake)
    request = _request()
    _run_script(socket_path, request)

    settled = _settle(socket_path, request)

    assert settled == {"step": "settled", "result": "adopt"}
    assert not [call for call in fake.calls if call[0] == "DELETE"]


def test_settle_of_another_attempt_removes_nothing(fake_dockerd_socket):
    fake = _FakeDockerd()
    socket_path = fake_dockerd_socket(fake)
    request = _request()
    _run_script(socket_path, request)

    settled = _settle(socket_path, request, attempt="someone-else")

    assert settled == {"step": "settled", "result": "gone"}
    assert not [call for call in fake.calls if call[0] == "DELETE"]


def test_settle_removes_what_a_killed_attempt_recorded(fake_dockerd_socket):
    fake = _FakeDockerd(running=False)
    socket_path = fake_dockerd_socket(fake)
    request = _request()
    process = subprocess.Popen(
        [sys.executable, "-c", _script(socket_path, running_check_sec=30)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    process.stdin.write(json.dumps(request))
    process.stdin.close()
    while json.loads(process.stdout.readline())["step"] != "started":
        pass
    process.kill()
    process.wait(timeout=10)

    settled = _settle(socket_path, request)

    assert settled == {"step": "settled", "result": "gone"}
    assert [path for method, path, _ in fake.calls if method == "DELETE"] == [
        f"/containers/{_CONTAINER_ID}?force=1&v=1", "/volumes/volume_pod-1",
    ]


def test_script_refuses_a_request_with_unexpected_fields(fake_dockerd_socket):
    fake = _FakeDockerd()

    lines, exit_status = _run_script(fake_dockerd_socket(fake), {**_request(), "privileged": True})

    assert exit_status == 2
    assert lines[0]["failed_step"] == "request"
    assert fake.calls == []


def test_hostile_values_reach_the_host_only_inside_the_json(fake_dockerd_socket, tmp_path):
    marker = tmp_path / "pwned"
    hostile = _HOSTILE.format(marker=marker)
    spec = _run_spec(image=hostile, name=f"container_{hostile}", environment={"EVIL": hostile})
    request = _request(spec, volume_name=f"volume_{hostile}")
    fake = _FakeDockerd()

    lines, exit_status = _run_script(fake_dockerd_socket(fake), request)

    assert hostile not in ONE_EXEC_CREATE_COMMAND
    assert exit_status == 0, lines
    volume_body, container_body = fake.calls[3][2], fake.calls[4][2]
    assert volume_body["Name"] == f"volume_{hostile}"
    assert container_body["Image"] == hostile
    assert f"EVIL={hostile}" in container_body["Env"]
    assert fake.calls[4][1] == "/containers/create?name=" + quote(f"container_{hostile}", safe="")
    assert not marker.exists()


@pytest.mark.asyncio
async def test_connector_sends_the_constant_command_and_the_values_on_stdin_only(tmp_path):
    hostile = _HOSTILE.format(marker=tmp_path / "pwned")
    request = _request(_run_spec(image=hostile, environment={"EVIL": hostile}), volume_name=hostile)
    process = _FakeProcess([])
    ssh_client = Mock(create_process=AsyncMock(return_value=process))

    await run_one_exec_create(ssh_client, request, AsyncMock(), {})

    ssh_client.create_process.assert_awaited_once_with(ONE_EXEC_CREATE_COMMAND)
    (sent,) = [call.args[0] for call in process.stdin.write.call_args_list]
    assert json.loads(sent) == json.loads(json.dumps(request))


def test_container_body_is_the_one_docker_py_posts_for_the_same_spec(monkeypatch):
    spec = _run_spec(command=("bash", "-c", "sleep 1"), entrypoint="/bin/sh")
    api = docker.APIClient(base_url="unix:///tmp/none.sock", version=docker.constants.DEFAULT_DOCKER_API_VERSION)
    api._proxy_configs = docker.utils.proxy.ProxyConfig()
    posted = {}
    monkeypatch.setattr(api, "create_container_from_config", lambda config, name, platform=None: posted.update(config))
    monkeypatch.setattr(api, "start", Mock())
    client = RentalDockerSdkClient(api)
    monkeypatch.setattr(client, "_ensure_rental_network_sync", Mock())

    client._run_container_sync(spec)

    assert build_container_create_body(spec) == posted


# ------------------------------------------------------------------
# create_container
# ------------------------------------------------------------------


class _Lines:
    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeProcess:
    def __init__(self, lines: list[str], exit_status: int = 0, stderr: str = ""):
        self.stdin = Mock()
        self.stdout = _Lines(lines)
        self.stderr = Mock(read=AsyncMock(return_value=stderr))
        self._exit_status = exit_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def wait(self):
        return Mock(exit_status=self._exit_status)


def _line(step: str, ms: int) -> str:
    return json.dumps({"step": step, "ms": ms}) + "\n"


def _failed(failed_step: str, error: str = "boom", **details) -> str:
    return json.dumps({"step": "failed", "failed_step": failed_step, "error": error, "cleaned": True, **details}) + "\n"


_HAPPY_LINES = [
    _line("network", 2), _line("volume_created", 370), _line("container_created", 400),
    _line("started", 1100), _line("running", 3),
]


def _wire_one_exec(svc, monkeypatch, process: _FakeProcess, *, plugin_enabled: bool = True):
    ssh_client = _deploy_ssh_client()
    ssh_client.create_process = AsyncMock(return_value=process)
    _wire(svc, monkeypatch, ssh_client, probe_result=None)
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", False)
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(settings, "VOLUME_MASTER_SECRET", "test-master-secret-32-chars-long!!")
    monkeypatch.setattr(svc, "_image_has_encrypted_volume_label", AsyncMock(return_value=True))
    monkeypatch.setattr(svc, "setup_encrypted_local_volume", AsyncMock())
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", AsyncMock(return_value=False))
    monkeypatch.setattr(svc, "_restore_gpu_power_for_uncapped_pod", AsyncMock())
    # nothing reclaimed: the cleanup left the host as it was
    monkeypatch.setattr(svc, "reclaim_dphn_cache_for_rental", AsyncMock(return_value=[]))
    monkeypatch.setattr(ds, "fetch_docker_hub_digest", AsyncMock(return_value=None))
    monkeypatch.setattr(
        svc,
        "probe_volume_host",
        AsyncMock(return_value=VolumeHostProbe(
            docker_root_dir="/var/lib/docker",
            df_avail_bytes=None,
            vloopback_volume_names=[],
            loopback_plugin_enabled=plugin_enabled,
            loopback_plugin_installed=True,
        )),
    )
    return ssh_client


def _encrypted_payload(**over):
    return _deploy_payload(enable_volume_encryption=True, is_sysbox=True, **over)


def _stream_texts(svc) -> list[str]:
    return [call.args[0] for call in svc.stream_log.await_args_list]


def _rows(result) -> dict[ProfilerStepName, int]:
    return {step.name: step.duration for step in result.profilers}


@pytest.mark.asyncio
async def test_create_container_one_exec_maps_the_step_lines_to_todays_logs_and_rows(svc, monkeypatch):
    _wire_one_exec(svc, monkeypatch, _FakeProcess(_HAPPY_LINES))
    payload = _encrypted_payload()

    result = await _run_create_container(svc, payload)

    assert isinstance(result, ContainerCreated), getattr(result, "msg", "")
    texts = _stream_texts(svc)
    volume_text = f"Creating docker volume volume_{payload.pod_id}"
    assert texts.index(volume_text) < texts.index("Creating docker container") < texts.index("Created Docker Container")
    svc.create_local_volume.assert_not_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    svc.check_container_running.assert_not_awaited()
    rows = _rows(result)
    assert rows[ProfilerStepName.DOCKER_RUN] == 2 + 400 + 1100
    assert rows[ProfilerStepName.CONTAINER_RUNNING_CHECK] == 3
    assert rows[ProfilerStepName.DOCKER_VOLUME_CREATION] >= 0
    svc.setup_encrypted_local_volume.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lines", "failure_step", "error_code"),
    [
        ([_line("network", 2), _failed("volume_created", "500 disk full")],
         "volume_creation", FailedContainerErrorCodes.UnknownError),
        ([_line("network", 2), _line("volume_created", 9), _failed("container_created", "500 no such image")],
         "docker_run", FailedContainerErrorCodes.UnknownError),
        ([_line("network", 2), _line("volume_created", 9), _line("container_created", 9),
          _failed("started", "500 cgroup error")],
         "docker_run", FailedContainerErrorCodes.UnknownError),
        (_HAPPY_LINES[:4] + [_failed("running", "container is not running", state={"ExitCode": 1}, logs="x")],
         "container_health_check", FailedContainerErrorCodes.UnknownError),
        (_HAPPY_LINES[:4] + [_failed("running", "container is gone", vanished=True)],
         "container_health_check", FailedContainerErrorCodes.ContainerVanished),
    ],
    ids=["volume", "create", "start", "not-running", "vanished"],
)
async def test_create_container_one_exec_step_failure_keeps_todays_failure_step(
    svc, monkeypatch, lines, failure_step, error_code
):
    _wire_one_exec(svc, monkeypatch, _FakeProcess(lines, exit_status=1))

    result = await _run_create_container(svc, _encrypted_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == failure_step
    assert result.error_code == error_code
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    # the name-based cleanup stays behind the script's own
    svc.cleanup_failed_container_creation.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_container_one_exec_volume_failure_carries_the_daemon_reason(svc, monkeypatch):
    lines = [_line("network", 2), _failed("volume_created", "500 disk full")]
    _wire_one_exec(svc, monkeypatch, _FakeProcess(lines, exit_status=1))

    result = await _run_create_container(svc, _encrypted_payload())

    assert "disk full" in result.step_detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "process",
    [
        # dockerd refused a host port: the script removed its volume and container and said so
        _FakeProcess(_HAPPY_LINES[:3] + [_failed(
            "started", "500 Bind for 0.0.0.0:20001 failed: port is already allocated")], exit_status=1),
        # no python3 on the host
        _FakeProcess([], exit_status=127, stderr="python3: not found"),
        # the rental network is not there yet
        _FakeProcess([_failed("network", "404 network lium-rentals not found")], exit_status=1),
        # a leftover volume of this pod: today's path decides about it, as before
        _FakeProcess([_line("network", 2), _failed("volume_exists", "volume is already on the host")], exit_status=1),
    ],
    ids=["port-collision", "no-python", "no-network", "volume-exists"],
)
async def test_create_container_one_exec_refusal_takes_todays_path_without_a_settle(svc, monkeypatch, process):
    _wire_one_exec(svc, monkeypatch, process)
    monkeypatch.setattr(ds, "settle_one_exec_create", AsyncMock())

    result = await _run_create_container(svc, _encrypted_payload())

    assert isinstance(result, ContainerCreated), getattr(result, "msg", "")
    ds.settle_one_exec_create.assert_not_awaited()
    svc.create_local_volume.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    svc.check_container_running.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_container_lost_reply_adopts_the_running_container(svc, monkeypatch):
    # the channel broke after `started`: the attempt went on to a running container
    _wire_one_exec(svc, monkeypatch, _FakeProcess(_HAPPY_LINES[:4], exit_status=None))
    monkeypatch.setattr(ds, "settle_one_exec_create", AsyncMock(return_value="adopt"))

    result = await _run_create_container(svc, _encrypted_payload())

    assert isinstance(result, ContainerCreated), getattr(result, "msg", "")
    svc.create_local_volume.assert_not_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    assert "Created Docker Container" in _stream_texts(svc)


@pytest.mark.asyncio
async def test_create_container_lost_reply_takes_todays_path_only_after_a_confirmed_removal(svc, monkeypatch):
    _wire_one_exec(svc, monkeypatch, _FakeProcess(_HAPPY_LINES[:2], exit_status=124))
    monkeypatch.setattr(ds, "settle_one_exec_create", AsyncMock(return_value="gone"))

    result = await _run_create_container(svc, _encrypted_payload())

    assert isinstance(result, ContainerCreated), getattr(result, "msg", "")
    ds.settle_one_exec_create.assert_awaited_once()
    svc.create_local_volume.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("settled", ["not_gone", None])
async def test_create_container_lost_reply_unconfirmed_removal_fails_the_create(svc, monkeypatch, settled):
    _wire_one_exec(svc, monkeypatch, _FakeProcess(_HAPPY_LINES[:2], exit_status=None))
    monkeypatch.setattr(ds, "settle_one_exec_create", AsyncMock(return_value=settled))

    result = await _run_create_container(svc, _encrypted_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run"
    svc.create_local_volume.assert_not_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["filler", "not-encrypted", "pulled", "cleanup-removed", "external-volume", "plugin-disabled", "cvm"],
)
async def test_create_container_without_the_narrow_conditions_never_uses_the_one_exec(svc, monkeypatch, case):
    ssh_client = _wire_one_exec(svc, monkeypatch, _FakeProcess(_HAPPY_LINES), plugin_enabled=case != "plugin-disabled")
    payload = _encrypted_payload()
    executor_tdx = None
    if case == "filler":
        payload = _encrypted_payload(workload_kind=WorkloadKind.FILLER)
    elif case == "not-encrypted":
        payload = _deploy_payload(is_sysbox=True)
    elif case == "pulled":
        svc.rental_docker_client_factory.client.image_exists_result = False
    elif case == "cleanup-removed":
        monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=["filler_1"]))
    elif case == "external-volume":
        payload = _encrypted_payload(external_volume_info=ExternalVolumeInfo(
            name="s3vol", plugin="s3fs", iam_user_access_key="a", iam_user_secret_key="b"))
        monkeypatch.setattr(svc, "create_s3fs_volume", AsyncMock(return_value=(True, "")))
    elif case == "cvm":
        executor_tdx = "quote"
        monkeypatch.setattr(svc, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    if executor_tdx:
        from test_deploy_optimizations import _executor_info

        info = _executor_info(payload)
        info.tdx_quote = executor_tdx
        await svc.create_container(payload=payload, executor_info=info,
                                   keypair=Mock(ss58_address="validator-hotkey"), private_key="encrypted")
    else:
        await _run_create_container(svc, payload)

    ssh_client.create_process.assert_not_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
