"""One Docker API channel per host: the rental adapter's pool is keyed by host, not by URL.

docker-py keys its SSH pool by the full request URL, so every SDK call of a rent opened a new SSH
channel and ran `docker system dial-stdio` (~3 round trips instead of 1). Sharing one pool must not
hand a call a channel another stream owns, a dead channel, or a channel another thread is using.

Each SSH channel here is one end of a socket pair; a thread on the other end plays dockerd.
"""

import json
import os
import select
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import docker
import paramiko
import pytest
from services.rental_docker_sdk import (
    _DOCKER_SDK_SSH_ADAPTER_LOCK,
    RentalDockerSdkClientFactory,
    _create_docker_api_client_with_rental_ssh_adapter,
)

BASE_URL = "ssh://root@203.0.113.10:2222"
RAW_STREAM_BYTES = b"bytes of the exec stream"


class _FakeChannel(socket.socket):
    """A socket with the paramiko Channel surface the adapter and docker-py touch."""

    def exec_command(self, command):
        self.command = command

    def makefile(self, *args, **kwargs):
        reader = super().makefile(*args, **kwargs)
        # docker-py takes a hijacked SSH stream from `response.raw._fp.fp.channel`
        reader.channel = self
        return reader

    def close(self):
        _close_at_once(self)

    @property
    def closed(self) -> bool:
        return self.fileno() == -1

    @property
    def eof_received(self) -> bool:
        return self._peek() == b""

    def recv_ready(self) -> bool:
        return bool(self._peek())

    def _peek(self) -> bytes | None:
        readable, _, _ = select.select([self], [], [], 0)
        return self.recv(1, socket.MSG_PEEK) if readable else None


class _FakeDockerHost:
    """A paramiko transport stand-in: every `open_session()` is a new channel with its own dockerd."""

    def __init__(self, name: str, *, hold_requests: int = 0):
        self.name = name
        self.channels: list[_FakeChannel] = []
        self._daemon_ends: list[socket.socket] = []
        # GET .../hold/... answers only once this many requests wait at once
        self._hold = threading.Barrier(hold_requests) if hold_requests else None
        self._stream_output_due = threading.Event()
        # what reached dockerd on a hijacked channel after the upgrade, set at the stream's end
        self._stream_input: bytes | None = None
        self._stream_ended = threading.Event()
        # what each SSH connect / channel open saw: the adapter lock's state and the open timeout
        self.adapter_lock_held_at_connect: list[bool] = []
        self.adapter_lock_held_at_open: list[bool] = []
        self.open_session_timeouts: list[float | None] = []
        # cleared, the host accepts the session but never answers a channel open
        self.open_session_gate = threading.Event()
        self.open_session_gate.set()

    def is_active(self) -> bool:
        return True

    def set_keepalive(self, interval):
        pass

    def open_session(self, timeout=None) -> _FakeChannel:
        self.open_session_timeouts.append(timeout)
        self.adapter_lock_held_at_open.append(_DOCKER_SDK_SSH_ADAPTER_LOCK.locked())
        self.open_session_gate.wait(timeout=10)
        near_end, daemon_end = socket.socketpair()
        channel = _FakeChannel(fileno=near_end.detach())
        self.channels.append(channel)
        self._daemon_ends.append(daemon_end)
        threading.Thread(
            target=self._serve, args=(daemon_end, len(self.channels)), daemon=True
        ).start()
        return channel

    def drop_channel(self, number: int) -> None:
        # dial-stdio exited: the daemon end of that channel goes away
        _close_at_once(self._daemon_ends[number - 1])

    def send_stream_output(self) -> None:
        self._stream_output_due.set()

    def stream_input(self) -> bytes | None:
        self._stream_ended.wait(timeout=10)
        return self._stream_input

    def close(self) -> None:
        self._stream_output_due.set()
        for daemon_end in self._daemon_ends:
            _close_at_once(daemon_end)

    def _serve(self, daemon_end: socket.socket, channel_number: int) -> None:
        requests_file = daemon_end.makefile("rb")
        try:
            while request_line := requests_file.readline():
                method, path, _ = request_line.decode().split(" ", 2)
                content_length = 0
                while (header := requests_file.readline()) not in (b"\r\n", b""):
                    name, _, value = header.decode().partition(":")
                    if name.lower() == "content-length":
                        content_length = int(value)
                requests_file.read(content_length)
                if path.endswith("/start"):
                    daemon_end.sendall(
                        b"HTTP/1.1 101 UPGRADED\r\n"
                        b"Content-Type: application/vnd.docker.raw-stream\r\n"
                        b"Connection: Upgrade\r\nUpgrade: tcp\r\n\r\n"
                    )
                    # the channel is a raw stream now: no more HTTP on it
                    self._stream_output_due.wait(timeout=10)
                    daemon_end.sendall(RAW_STREAM_BYTES)
                    daemon_end.shutdown(socket.SHUT_WR)
                    self._stream_input = requests_file.read()
                    self._stream_ended.set()
                    return
                if "/hold/" in path:
                    self._hold.wait(timeout=10)
                answer = {"host": self.name, "channel": channel_number, "path": path}
                if path.endswith("/version"):
                    answer["ApiVersion"] = "1.45"
                body = json.dumps(answer).encode()
                daemon_end.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
        except (OSError, ValueError, threading.BrokenBarrierError):
            return


def _close_at_once(end: socket.socket) -> None:
    # a bare close() waits for the socket's makefile() readers; a paramiko channel closes at once.
    # On Linux a close() under a thread blocked in recv() sends no EOF to the peer; shutdown() does
    if end.fileno() != -1:
        with suppress(OSError):
            end.shutdown(socket.SHUT_RDWR)
        os.close(end.detach())


class _FakeSSHClient:
    def __init__(self, host: _FakeDockerHost):
        self._host = host

    def load_host_keys(self, path):
        pass

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, **params):
        self._host.adapter_lock_held_at_connect.append(_DOCKER_SDK_SSH_ADAPTER_LOCK.locked())

    def get_transport(self) -> _FakeDockerHost:
        return self._host

    def close(self):
        pass


class _RoutingSSHClient(_FakeSSHClient):
    """Each build's SSH session lands on the fake host named by the address it connects to."""

    def __init__(self, hosts: dict[str, _FakeDockerHost]):
        self._hosts = hosts
        self._host = None

    def connect(self, **params):
        self._host = self._hosts[params["hostname"]]
        super().connect(**params)


def _rental_api_client(monkeypatch, tmp_path: Path, host: _FakeDockerHost, version="1.45"):
    """The validator's docker-py APIClient with the rental adapter, its SSH session on `host`."""
    monkeypatch.setattr(paramiko, "SSHClient", lambda: _FakeSSHClient(host))
    return _build_client(tmp_path, BASE_URL, version)


def _build_client(tmp_path: Path, base_url: str, version):
    return _create_docker_api_client_with_rental_ssh_adapter(
        docker_module=docker,
        key_path=tmp_path / "id_executor",
        known_hosts_path=tmp_path / "known_hosts",
        base_url=base_url,
        version=version,
        timeout=5,
        use_ssh_client=False,
    )


@pytest.fixture
def host():
    fake_host = _FakeDockerHost("executor-a", hold_requests=12)
    yield fake_host
    fake_host.close()


def test_calls_to_distinct_urls_share_one_channel(monkeypatch, tmp_path, host):
    api = _rental_api_client(monkeypatch, tmp_path, host)

    answers = [api.inspect_container(f"pod-{index}") for index in range(8)]

    assert [answer["path"] for answer in answers] == [
        f"/v1.45/containers/pod-{index}/json" for index in range(8)
    ]
    assert len(host.channels) == 1
    api.close()


def test_a_hijacked_channel_never_carries_a_later_call(monkeypatch, tmp_path, host):
    api = _rental_api_client(monkeypatch, tmp_path, host)
    stream = api.exec_start("exec-1", socket=True)
    # worst case: the hijacked connection goes back to the pool while its stream is still open
    stream._response.raw.release_conn()

    answer = api.inspect_container("pod-1")
    host.send_stream_output()
    stream_output = stream.recv(1024)
    stream.close()

    assert answer == {"host": "executor-a", "channel": 2, "path": "/v1.45/containers/pod-1/json"}
    assert stream_output == RAW_STREAM_BYTES
    assert host.stream_input() == b""
    api.close()


def test_a_dead_channel_is_replaced_not_reused(monkeypatch, tmp_path, host):
    api = _rental_api_client(monkeypatch, tmp_path, host)
    api.inspect_container("pod-1")
    host.drop_channel(1)

    answer = api.inspect_container("pod-1")

    assert answer["channel"] == 2
    assert host.channels[0].closed
    api.close()


def test_concurrent_calls_on_one_client_each_get_their_own_channel(monkeypatch, tmp_path, host):
    # 12 > docker-py's 10 pooled connections: the pool keeps 10 idle, it never makes a call wait
    api = _rental_api_client(monkeypatch, tmp_path, host)

    with ThreadPoolExecutor(max_workers=12) as threads:
        answers = list(threads.map(lambda index: api.inspect_container(f"hold/{index}"), range(12)))

    assert [answer["path"] for answer in answers] == [
        f"/v1.45/containers/hold/{index}/json" for index in range(12)
    ]
    assert sorted(answer["channel"] for answer in answers) == list(range(1, 13))
    api.close()


def test_two_clients_of_one_host_never_share_a_pool(monkeypatch, tmp_path):
    first_rent_host = _FakeDockerHost("first-rent")
    second_rent_host = _FakeDockerHost("second-rent")
    first_api = _rental_api_client(monkeypatch, tmp_path, first_rent_host)
    second_api = _rental_api_client(monkeypatch, tmp_path, second_rent_host)

    first_answer = first_api.inspect_container("pod-1")
    second_answer = second_api.inspect_container("pod-1")

    assert first_answer["host"] == "first-rent"
    assert second_answer["host"] == "second-rent"
    assert len(first_rent_host.channels) == len(second_rent_host.channels) == 1
    first_api.close()
    second_api.close()
    first_rent_host.close()
    second_rent_host.close()


def test_client_build_connects_and_reads_the_daemon_version_outside_the_adapter_lock(
    monkeypatch, tmp_path, host
):
    api = _rental_api_client(monkeypatch, tmp_path, host, version=None)

    assert api.api_version == "1.45"
    assert host.adapter_lock_held_at_connect == [False]
    assert host.adapter_lock_held_at_open == [False]
    assert host.open_session_timeouts == [15]
    api.close()


def test_a_host_that_never_opens_a_channel_does_not_hold_another_hosts_client_build(
    monkeypatch, tmp_path
):
    # 11 Oct 2026 (ticket-0486): one silent host held the adapter lock through paramiko's 1 h
    # open_session wait, every other rental's client build queued behind it, and by the time they
    # ran the executors had purged the validator's 15-minute key, so all failed at ssh_connect
    silent_host = _FakeDockerHost("silent")
    silent_host.open_session_gate.clear()
    healthy_host = _FakeDockerHost("healthy")
    monkeypatch.setattr(
        paramiko,
        "SSHClient",
        lambda: _RoutingSSHClient({"203.0.113.10": silent_host, "203.0.113.11": healthy_host}),
    )

    factory = RentalDockerSdkClientFactory(timeout=5)

    def build(base_url: str):
        # the factory's own client build, the one every rental's connect runs in a docker-sdk thread
        return factory._create_api_client(
            base_url, tmp_path / "id_executor", tmp_path / "known_hosts", []
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        stuck_build = pool.submit(build, "ssh://root@203.0.113.10:2222")
        while not silent_host.open_session_timeouts:
            # a build that raised instead of waiting fails here, not as a hang
            assert not stuck_build.done(), stuck_build.exception()
            threading.Event().wait(0.01)

        healthy_api = build("ssh://root@203.0.113.11:2222")

        assert healthy_api.api_version == "1.45"
        assert not stuck_build.done()
        silent_host.open_session_gate.set()
        stuck_build.result(timeout=10).close()
    healthy_api.close()
    silent_host.close()
    healthy_host.close()
