"""A customer pod on a CVM node reaches the dstack guest agent only through the quote-only broker.

The broker is one nginx container per guest, started by the validator over the guest's docker
socket: it forwards ``GetQuote``/``Info``/``Version`` to ``/var/run/dstack.sock`` and answers
everything else (key derivation, RTMR3 extension) with 403. The pod gets the broker's socket at the
dstack SDK's default path; bare-metal nodes and fillers get nothing.
"""

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreateRequest, CustomOptions, WorkloadKind

from services import cvm_quote_broker as broker
from services.cvm_quote_broker import (
    DSTACK_GUEST_SOCKET_PATH,
    QUOTE_BROKER_ALLOWED_PATH_RE,
    QUOTE_BROKER_CONTAINER_NAME,
    QUOTE_BROKER_IMAGE,
    QUOTE_BROKER_REV_ENV,
    QUOTE_BROKER_SOCKET_DIR,
    QUOTE_BROKER_SOCKET_PATH,
    ensure_quote_broker,
    quote_broker_revision,
    quote_broker_run_spec,
    quote_socket_pod_mount,
)
from services.docker_service import DockerService, _wants_quote_socket
from services.rental_docker_sdk import (
    ContainerExecResult,
    GpuDockerConfig,
    RentalDockerOperationError,
    _binds,
)


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _make_payload(workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        workload_kind=workload_kind,
        active_container_names=[],
    )


def _build_run_spec(docker_service: DockerService, *, quote_socket: bool):
    return docker_service._build_rental_container_run_spec(
        payload=_make_payload(),
        container_name="pod_x",
        custom_options=CustomOptions(),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
        quote_socket=quote_socket,
    )


def _inspect_output(*, running: bool, revision: str | None) -> str:
    env = f"[PATH=/usr/sbin {QUOTE_BROKER_REV_ENV}={revision}]" if revision else "[PATH=/usr/sbin]"
    return f"{'true' if running else 'false'} {env}\n"


def _fake_clients(*, inspect_outputs: list[str], image_exists: bool = True, socket_exit_statuses=(0,)):
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=[SimpleNamespace(stdout=out) for out in inspect_outputs])
    docker_client = Mock()
    docker_client.image_exists = AsyncMock(return_value=image_exists)
    docker_client.pull = AsyncMock()
    docker_client.run_container = AsyncMock()
    docker_client.exec_in_container = AsyncMock(
        side_effect=[ContainerExecResult(exit_status=status) for status in socket_exit_statuses]
    )
    return docker_client, ssh_client


def _ssh_commands(ssh_client) -> list[str]:
    return [call.args[0] for call in ssh_client.run.await_args_list]


# --- what the broker lets through -------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/GetQuote", "/Info", "/Version", "/prpc/GetQuote", "/prpc/DstackGuest.Info", "/DstackGuest.Version"],
)
def test_allow_list_passes_quote_and_info_paths(path):
    # "/<Method>" is what the dstack SDK posts; the prpc spellings are the agent's own.
    assert re.match(QUOTE_BROKER_ALLOWED_PATH_RE, path)


@pytest.mark.parametrize(
    "path",
    [
        "/GetKey",  # derives the guest's app keys, shared with the executor and other pods
        "/GetTlsKey",
        "/Sign",
        "/EmitEvent",  # extends RTMR3 — would change the node's own measurement
        "/Attest",
        "/Verify",
        "/prpc/DstackGuest.GetKey",
        "/GetQuoteX",
        "/GetQuote/extra",
        "/",
    ],
)
def test_allow_list_rejects_everything_else(path):
    assert re.match(QUOTE_BROKER_ALLOWED_PATH_RE, path) is None


def test_nginx_conf_bridges_the_two_sockets_with_the_allow_list():
    conf = broker._NGINX_CONF

    assert f"listen unix:{QUOTE_BROKER_SOCKET_PATH};" in conf
    assert f"server unix:{DSTACK_GUEST_SOCKET_PATH};" in conf
    assert f"location ~ {QUOTE_BROKER_ALLOWED_PATH_RE} " in conf
    assert "return 403" in conf


# --- the broker container ----------------------------------------------------------------------


def test_run_spec_pins_the_image_by_digest_and_binds_both_sockets():
    spec = quote_broker_run_spec()

    assert spec.name == QUOTE_BROKER_CONTAINER_NAME
    assert spec.image == QUOTE_BROKER_IMAGE
    assert "@sha256:" in spec.image and ":1." not in spec.image
    # the real agent socket is read-only even for the broker: a root nginx must not be able to
    # replace the socket the executor and every other pod on the guest depend on
    assert [(m.source, m.target, m.read_only) for m in spec.volumes] == [
        (DSTACK_GUEST_SOCKET_PATH, DSTACK_GUEST_SOCKET_PATH, True),
        (QUOTE_BROKER_SOCKET_DIR, QUOTE_BROKER_SOCKET_DIR, False),
    ]
    assert spec.restart_policy == "unless-stopped"
    # a filter needs no GPU, no privileges and no resource caps (--cpus is broken inside a CVM)
    assert spec.devices == () and spec.device_requests == () and spec.cap_add == ()
    assert spec.cpu_count is None and spec.memory_gb is None


def test_run_spec_entrypoint_clears_the_socket_path_and_writes_the_conf():
    spec = quote_broker_run_spec()

    assert spec.entrypoint == "/bin/sh"
    script = spec.command[1]
    assert spec.command[0] == "-c"
    assert script.startswith(f"rm -rf {QUOTE_BROKER_SOCKET_PATH}")
    assert "> /etc/nginx/nginx.conf" in script
    assert script.endswith("exec nginx -g 'daemon off;'")
    assert spec.environment["LIUM_QUOTE_BROKER_NGINX_CONF"] == broker._NGINX_CONF
    assert spec.environment[QUOTE_BROKER_REV_ENV] == quote_broker_revision()


def test_revision_follows_the_conf(monkeypatch):
    before = quote_broker_revision()
    monkeypatch.setattr(broker, "_NGINX_CONF", broker._NGINX_CONF + "# changed\n")

    assert quote_broker_revision() != before


# --- ensure_quote_broker -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_keeps_a_broker_already_running_at_this_revision():
    docker_client, ssh_client = _fake_clients(
        inspect_outputs=[_inspect_output(running=True, revision=quote_broker_revision())]
    )

    await ensure_quote_broker(docker_client, ssh_client, log_extra={})

    assert len(_ssh_commands(ssh_client)) == 1
    docker_client.run_container.assert_not_awaited()
    docker_client.pull.assert_not_awaited()
    # "Running" only says the entrypoint shell started — the socket is still checked
    docker_client.exec_in_container.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_starts_the_broker_and_waits_for_its_socket():
    docker_client, ssh_client = _fake_clients(
        inspect_outputs=[_inspect_output(running=False, revision=None), "\n"],
        image_exists=False,
        socket_exit_statuses=(1, 0),
    )

    await ensure_quote_broker(docker_client, ssh_client, log_extra={})

    commands = _ssh_commands(ssh_client)
    assert commands[1].startswith(f"/usr/bin/docker rm -fv {QUOTE_BROKER_CONTAINER_NAME}")
    docker_client.pull.assert_awaited_once_with(image=QUOTE_BROKER_IMAGE)
    docker_client.run_container.assert_awaited_once()
    assert docker_client.run_container.await_args.args[0].name == QUOTE_BROKER_CONTAINER_NAME
    probe = docker_client.exec_in_container.await_args.args[0]
    assert probe.argv == ("test", "-S", QUOTE_BROKER_SOCKET_PATH)
    assert docker_client.exec_in_container.await_count == 2


@pytest.mark.asyncio
async def test_ensure_replaces_a_broker_from_an_older_release():
    docker_client, ssh_client = _fake_clients(
        inspect_outputs=[_inspect_output(running=True, revision="0000000000000000"), "\n"]
    )

    await ensure_quote_broker(docker_client, ssh_client, log_extra={})

    assert any(cmd.startswith("/usr/bin/docker rm -fv") for cmd in _ssh_commands(ssh_client))
    docker_client.run_container.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_tolerates_losing_the_race_to_a_parallel_pod_create():
    # two pods on one guest: the second `docker run` hits a name conflict, but the broker
    # the first one started is the same broker
    docker_client, ssh_client = _fake_clients(
        inspect_outputs=[
            _inspect_output(running=False, revision=None),
            "\n",
            _inspect_output(running=True, revision=quote_broker_revision()),
        ]
    )
    docker_client.run_container.side_effect = RentalDockerOperationError("Conflict: name in use")

    await ensure_quote_broker(docker_client, ssh_client, log_extra={})

    docker_client.exec_in_container.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_raises_when_the_broker_cannot_be_started():
    docker_client, ssh_client = _fake_clients(
        inspect_outputs=[_inspect_output(running=False, revision=None), "\n", "\n"]
    )
    docker_client.run_container.side_effect = RentalDockerOperationError("no space left on device")

    with pytest.raises(RentalDockerOperationError):
        await ensure_quote_broker(docker_client, ssh_client, log_extra={})


@pytest.mark.asyncio
async def test_ensure_raises_when_the_socket_never_appears(monkeypatch):
    # the deadline bounds the whole wait, including a slow exec, not just the sleeps between probes
    monkeypatch.setattr(broker, "QUOTE_BROKER_SOCKET_WAIT_SECONDS", 0.05)
    docker_client, ssh_client = _fake_clients(
        inspect_outputs=[_inspect_output(running=False, revision=None), "\n"],
    )

    async def slow_exec(spec):
        await asyncio.sleep(10)

    docker_client.exec_in_container = AsyncMock(side_effect=slow_exec)

    with pytest.raises(RuntimeError, match="did not appear"):
        await ensure_quote_broker(docker_client, ssh_client, log_extra={})


# --- what the pod gets --------------------------------------------------------------------------


def test_pod_mount_is_the_broker_socket_at_the_sdk_default_path_read_only():
    mount = quote_socket_pod_mount()

    assert (mount.source, mount.target, mount.read_only) == (
        QUOTE_BROKER_SOCKET_PATH,
        "/var/run/dstack.sock",
        True,
    )


def test_run_spec_appends_the_quote_socket_after_the_rental_volume(docker_service):
    run_spec = _build_run_spec(docker_service, quote_socket=True)

    assert [(m.source, m.target) for m in run_spec.volumes] == [
        ("volume_pod", "/root"),
        (QUOTE_BROKER_SOCKET_PATH, DSTACK_GUEST_SOCKET_PATH),
    ]
    # what actually reaches dockerd
    assert _binds(run_spec.volumes)[-1] == "/var/run/lium-dstack/dstack.sock:/var/run/dstack.sock:ro"


def test_run_spec_default_is_no_socket(docker_service):
    run_spec = _build_run_spec(docker_service, quote_socket=False)

    assert [(m.source, m.target) for m in run_spec.volumes] == [("volume_pod", "/root")]


@pytest.mark.parametrize(
    ("enabled", "in_cvm", "workload_kind", "expected"),
    [
        (True, True, WorkloadKind.CUSTOMER_RENTAL, True),
        (True, False, WorkloadKind.CUSTOMER_RENTAL, False),  # bare metal: no guest agent
        (True, True, WorkloadKind.FILLER, False),  # DPHN/PEARL have no attestation use
        (False, True, WorkloadKind.CUSTOMER_RENTAL, False),  # kill switch
    ],
)
def test_wants_quote_socket(monkeypatch, enabled, in_cvm, workload_kind, expected):
    monkeypatch.setattr("services.docker_service.settings.ENABLE_CVM_POD_QUOTE_SOCKET", enabled)

    assert _wants_quote_socket(_make_payload(workload_kind), in_cvm=in_cvm) is expected
