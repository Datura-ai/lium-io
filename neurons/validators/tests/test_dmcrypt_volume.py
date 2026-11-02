"""dm-crypt (LUKS2) pod volumes: method choice and fallback, key only on stdin, open before start,
cleanup on delete and on a failed create, disk accounting."""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    FailedContainerRequest,
    PayloadPortMapping,
    VolumeEncryptionStatus,
    WorkloadKind,
)

import services.docker_service as docker_service_module
from services import dmcrypt_volume as dm
from services.docker_service import _LIUM_CIPHER_MOUNT, _parse_volume_host_probe
from services.volume_keys import derive_dmcrypt_key, derive_volume_passphrase
from tests.test_delete_drops_dirty_volume import _delete, _delete_payload, _host_answers, _wire_delete
from tests.test_docker_service import (  # noqa: F401
    FAKE_SSH_HOST_KEY,
    DummySSHConnectionManager,
    _make_ssh_command_result,
    _patch_create_container_happy_path,
    docker_service,
    mock_dependencies,
)

_MASTER = "test-master-secret-32-chars-long!!"
# dockerd's answer to a start whose dm-crypt mapping is closed (stage 1, after a host reboot)
_CLOSED_MAPPING_ERROR = (
    "error while mounting volume '/var/lib/docker/volumes/volume_x/_data': failed to mount local volume: "
    "mount /dev/mapper/lium-{pod}:/var/lib/docker/volumes/volume_x/_data, flags: 0x6: no such file or directory"
)


def test_dmcrypt_key_is_512_bits_and_not_the_gocryptfs_passphrase():
    pod = str(uuid4())
    key = derive_dmcrypt_key(_MASTER, pod)
    assert key == derive_dmcrypt_key(_MASTER, pod)
    assert key != derive_dmcrypt_key(_MASTER, str(uuid4()))
    assert len(key) == 86  # 64 bytes, base64url without padding
    assert key != derive_volume_passphrase(_MASTER, pod)


def test_commands_never_carry_the_key_and_run_in_the_host_mount_namespace():
    pod = str(uuid4())
    key = derive_dmcrypt_key(_MASTER, pod)
    for command in (
        dm.dmcrypt_create_command(pod, 20, sparse=False),
        dm.dmcrypt_open_command(pod),
        dm.dmcrypt_close_command(pod),
        dm.dmcrypt_delete_command(pod),
    ):
        assert key not in command
        assert "nsenter -t 1 -m -- sh -c" in command
        assert pod in command
    assert dm.dmcrypt_create_command(pod, 20, sparse=True).endswith(f"{pod} 20G 1")
    assert dm.dmcrypt_create_command(pod, 20, sparse=False).endswith(f"{pod} 20G 0")


def test_commands_refuse_anything_but_a_pod_uuid():
    for bad in ("", "pod-1", "x; rm -rf /", str(uuid4()).upper()):
        with pytest.raises(ValueError):
            dm.dmcrypt_open_command(bad)
    with pytest.raises(ValueError):
        dm.dmcrypt_create_command(str(uuid4()), 0, sparse=False)


def test_capability_probe_is_the_same_in_the_scrape():
    # machine_scrape runs alone on the executor (importing it runs the scrape): read its constant
    source = (Path(dm.__file__).parents[1] / "miner_jobs" / "machine_scrape.py").read_text()
    node = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "DMCRYPT_CAPABILITY_PROBE"
    )
    assert ast.literal_eval(node.value) == dm.DMCRYPT_CAPABILITY_PROBE


def test_device_listing_and_closed_mapping_error():
    pod = str(uuid4())
    assert dm.dmcrypt_pod_ids(f"<no value>\n/dev/mapper/lium-{pod}\n/dev/mapper/other\n") == [pod]
    assert dm.is_closed_dmcrypt_mapping_error(_CLOSED_MAPPING_ERROR.format(pod=pod))
    assert not dm.is_closed_dmcrypt_mapping_error("failed to mount local volume: mount /dev/sdb1")


def test_volume_host_probe_reads_the_backing_bytes():
    stdout = "ROOT\t/var/lib/docker\nVOLS\t0\nCRYPT\t1073741824\nPLUGIN\ttrue\n"
    assert _parse_volume_host_probe(stdout, with_df=False).dmcrypt_backing_bytes == 1 << 30
    stdout = "ROOT\t/var/lib/docker\nVOLS\t0\nCRYPT\t\nPLUGIN\ttrue\n"
    assert _parse_volume_host_probe(stdout, with_df=False).dmcrypt_backing_bytes == 0


@pytest.mark.asyncio
async def test_create_dmcrypt_volume_sends_the_key_on_stdin_only(docker_service, monkeypatch):
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", _MASTER)
    pod = str(uuid4())
    ssh_client = Mock(run=AsyncMock(return_value=_make_ssh_command_result()))
    docker_client = docker_service.rental_docker_client_factory.client

    await docker_service.create_dmcrypt_volume(
        ssh_client, docker_client, local_volume=f"volume_{pod}", pod_id=pod, size_gb=20, sparse=False, log_extra={}
    )

    key = derive_dmcrypt_key(_MASTER, pod)
    command = ssh_client.run.await_args.args[0]
    assert command == dm.dmcrypt_create_command(pod, 20, sparse=False)
    assert ssh_client.run.await_args.kwargs["input"] == f"{key}\n"
    [volume] = docker_client.created_volumes
    assert volume["volume_name"] == f"volume_{pod}"
    assert volume["driver"] == "local"
    assert volume["driver_opts"] == {"type": "ext4", "device": f"/dev/mapper/lium-{pod}"}


def _payload(**overrides) -> ContainerCreateRequest:
    fields = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        docker_image="daturaai/dlph:test",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        is_sysbox=True,
        enable_volume_encryption=True,
        volume_encryption_method=dm.DMCRYPT_METHOD,
        available_ports=[PayloadPortMapping(internal_port=20020, external_port=20020)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    fields.update(overrides)
    return ContainerCreateRequest(**fields)


async def _create(docker_service, payload):
    return await docker_service.create_container(
        payload=payload,
        executor_info=ExecutorSSHInfo(
            uuid=payload.executor_id,
            address="127.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=2200,
            python_path="/usr/bin/python",
            root_dir="/root/app",
            ssh_host_key=FAKE_SSH_HOST_KEY,
        ),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _wire_create(docker_service, monkeypatch, *, capable: bool):
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", _MASTER)
    ssh_client = _patch_create_container_happy_path(docker_service, monkeypatch)
    spies = {
        "dmcrypt_capable": AsyncMock(return_value=capable),
        "create_dmcrypt_volume": AsyncMock(),
        "delete_dmcrypt_volume": AsyncMock(),
        "setup_encrypted_local_volume": AsyncMock(),
        "_image_has_encrypted_volume_label": AsyncMock(return_value=True),
        "_pick_loopback_driver_for_new_volume": AsyncMock(return_value="vloopback:v2"),
    }
    for name, spy in spies.items():
        monkeypatch.setattr(docker_service, name, spy)
    return ssh_client, spies


@pytest.mark.asyncio
async def test_create_makes_a_dmcrypt_volume_when_asked_and_capable(docker_service, monkeypatch):
    _, spies = _wire_create(docker_service, monkeypatch, capable=True)
    payload = _payload()

    result = await _create(docker_service, payload)

    assert isinstance(result, ContainerCreated)
    assert result.volume_encryption_status == VolumeEncryptionStatus.ENABLED
    assert result.volume_encryption_method == dm.DMCRYPT_METHOD
    spies["create_dmcrypt_volume"].assert_awaited_once()
    kwargs = spies["create_dmcrypt_volume"].await_args.kwargs
    assert (kwargs["pod_id"], kwargs["size_gb"], kwargs["sparse"]) == (payload.pod_id, 2, False)
    # no gocryptfs: no setup exec, no image label, no /dev/fuse, the volume mounted at the workspace itself
    spies["setup_encrypted_local_volume"].assert_not_awaited()
    spies["_image_has_encrypted_volume_label"].assert_not_awaited()
    docker_service.create_local_volume.assert_not_awaited()
    run_spec = docker_service.rental_docker_client_factory.client.run_specs[-1]
    assert not any(device.path_on_host == "/dev/fuse" for device in run_spec.devices)
    assert not any(volume.target == _LIUM_CIPHER_MOUNT for volume in run_spec.volumes)
    assert any(volume.source == f"volume_{payload.pod_id}" for volume in run_spec.volumes)
    # a successful create keeps its volume
    spies["delete_dmcrypt_volume"].assert_not_awaited()


@pytest.mark.asyncio
async def test_create_falls_back_to_gocryptfs_on_a_host_that_cannot(docker_service, monkeypatch):
    _, spies = _wire_create(docker_service, monkeypatch, capable=False)

    result = await _create(docker_service, _payload())

    assert isinstance(result, ContainerCreated)
    assert result.volume_encryption_method == dm.GOCRYPTFS_METHOD
    spies["create_dmcrypt_volume"].assert_not_awaited()
    spies["setup_encrypted_local_volume"].assert_awaited_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"volume_encryption_method": None},  # the backend did not ask
        {"volume_encryption_method": "gocryptfs"},
        {"enable_volume_encryption": False},  # no encryption at all
        {"workload_kind": WorkloadKind.FILLER},
    ],
)
@pytest.mark.asyncio
async def test_create_without_a_dmcrypt_ask_never_probes(docker_service, monkeypatch, overrides):
    _, spies = _wire_create(docker_service, monkeypatch, capable=True)

    await _create(docker_service, _payload(**overrides))

    spies["dmcrypt_capable"].assert_not_awaited()
    spies["create_dmcrypt_volume"].assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_create_removes_the_dmcrypt_volume(docker_service, monkeypatch):
    ssh_client, spies = _wire_create(docker_service, monkeypatch, capable=True)
    docker_service.rental_docker_client_factory.client.run_error = RuntimeError("docker run failed")
    payload = _payload()

    result = await _create(docker_service, payload)

    assert isinstance(result, FailedContainerRequest)
    spies["delete_dmcrypt_volume"].assert_awaited_once()
    assert spies["delete_dmcrypt_volume"].await_args.args[:2] == (ssh_client, payload.pod_id)


@pytest.mark.asyncio
async def test_a_create_failing_after_docker_run_removes_its_container_before_the_volume(docker_service, monkeypatch):
    # the running container pins the volume and keeps the mapping mounted: it must go first, also when
    # the create's own container cleanup did not remove it
    ssh_client, spies = _wire_create(docker_service, monkeypatch, capable=True)
    monkeypatch.setattr(docker_service, "_run_rental_docker_create_with_port_retry", AsyncMock(return_value="cid"))
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(side_effect=RuntimeError("not healthy")))
    monkeypatch.setattr(docker_service, "cleanup_failed_container_creation", AsyncMock(return_value=False))
    events: list[str] = []
    ssh_client.run.side_effect = lambda command, **kwargs: events.append(command) or _make_ssh_command_result()
    spies["delete_dmcrypt_volume"].side_effect = lambda *args: events.append("dmcrypt delete")
    payload = _payload()

    result = await _create(docker_service, payload)

    assert isinstance(result, FailedContainerRequest)
    teardown = [e for e in events if e.startswith(("/usr/bin/docker rm -fv cid", "/usr/bin/docker volume rm")) or e == "dmcrypt delete"]
    assert teardown == [
        "/usr/bin/docker rm -fv cid >/dev/null 2>&1",
        f"/usr/bin/docker volume rm volume_{payload.pod_id} >/dev/null 2>&1",
        "dmcrypt delete",
    ]


@pytest.mark.asyncio
async def test_the_stale_sweep_removes_an_orphaned_dmcrypt_volume_and_its_mapping(docker_service, monkeypatch):
    orphan, mounted, plain = str(uuid4()), str(uuid4()), str(uuid4())
    delete = AsyncMock()
    monkeypatch.setattr(docker_service, "delete_dmcrypt_volume", delete)
    commands: list[str] = []

    async def run(command, **kwargs):
        commands.append(command)
        if command.startswith("/usr/bin/docker volume ls"):
            return _make_ssh_command_result(
                stdout=f"volume_{orphan} local\nvolume_{mounted} local\nvolume_{plain} local\n"
            )
        if "xargs -r /usr/bin/docker inspect" in command:
            return _make_ssh_command_result(stdout=f"volume_{mounted}\n")
        if command.startswith("/usr/bin/docker volume inspect"):
            return _make_ssh_command_result(
                stdout=f"volume_{orphan} /dev/mapper/lium-{orphan}\nvolume_{plain} <no value>\n"
            )
        return _make_ssh_command_result()

    ssh_client = Mock(run=AsyncMock(side_effect=run))

    removed = await docker_service.clean_stale_vloopback_volumes(ssh_client=ssh_client, default_extra={})

    assert removed == [f"volume_{orphan}"]
    assert f"volume_{mounted}" not in commands[2]  # a mounted volume is never even inspected
    assert commands[3:] == [f"/usr/bin/docker volume rm volume_{orphan}"]
    assert delete.await_args.args[:2] == (ssh_client, orphan)


@pytest.mark.asyncio
async def test_a_filler_sweep_never_touches_a_dmcrypt_volume(docker_service, monkeypatch):
    pod = str(uuid4())
    monkeypatch.setattr(docker_service, "delete_dmcrypt_volume", AsyncMock())
    ssh_client = Mock(
        run=AsyncMock(side_effect=[_make_ssh_command_result(stdout=f"volume_{pod} local\n"), _make_ssh_command_result()])
    )

    assert await docker_service.clean_stale_vloopback_volumes(ssh_client=ssh_client, default_extra={}, filler_volumes_only=True) == []
    docker_service.delete_dmcrypt_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_edit_of_a_dmcrypt_pod_opens_its_volume_and_skips_gocryptfs(docker_service, monkeypatch):
    ssh_client, spies = _wire_create(docker_service, monkeypatch, capable=True)
    payload = _payload()
    payload = _payload(pod_id=payload.pod_id, local_volume=f"volume_{payload.pod_id}")
    opened = AsyncMock(return_value=[payload.pod_id])
    monkeypatch.setattr(docker_service, "open_dmcrypt_volumes", opened)

    result = await _create(docker_service, payload)

    assert isinstance(result, ContainerCreated)
    assert result.volume_encryption_method == dm.DMCRYPT_METHOD
    assert opened.await_args.args[1] == dm.volume_devices_command([payload.local_volume])
    spies["create_dmcrypt_volume"].assert_not_awaited()
    spies["setup_encrypted_local_volume"].assert_not_awaited()


@pytest.mark.asyncio
async def test_open_sends_the_key_on_stdin_for_each_dmcrypt_volume(docker_service, monkeypatch):
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", _MASTER)
    pod = str(uuid4())

    async def run(command, **kwargs):
        if command == dm.mounted_volume_devices_command("pod_x"):
            return _make_ssh_command_result(stdout=f"<no value>\n/dev/mapper/lium-{pod}\n")
        return _make_ssh_command_result()

    ssh_client = Mock(run=AsyncMock(side_effect=run))

    assert await docker_service.open_dmcrypt_volumes(ssh_client, dm.mounted_volume_devices_command("pod_x"), {}) == [pod]

    open_call = ssh_client.run.await_args_list[-1]
    assert open_call.args[0] == dm.dmcrypt_open_command(pod)
    assert open_call.kwargs["input"] == f"{derive_dmcrypt_key(_MASTER, pod)}\n"


@pytest.mark.asyncio
async def test_a_failed_open_stops_the_start(docker_service, monkeypatch):
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", _MASTER)
    pod = str(uuid4())
    ssh_client = Mock(
        run=AsyncMock(
            side_effect=[
                _make_ssh_command_result(stdout=f"/dev/mapper/lium-{pod}\n"),
                _make_ssh_command_result(exit_status=2, stderr="No key available with this passphrase."),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="dm-crypt volume open failed"):
        await docker_service.open_dmcrypt_volumes(ssh_client, dm.volume_devices_command(["v"]), {})


@pytest.mark.asyncio
async def test_start_after_a_reboot_opens_the_mapping_then_starts(docker_service, monkeypatch):
    pod = str(uuid4())
    ssh_client = Mock(run=AsyncMock(return_value=_make_ssh_command_result()))
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock(return_value="pkey"))
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect", Mock(return_value=DummySSHConnectionManager(ssh_client))
    )
    client = docker_service.rental_docker_client_factory.client
    events: list[str] = []
    starts = iter([RuntimeError(_CLOSED_MAPPING_ERROR.format(pod=pod)), None])

    async def start(*, container_name, network=None):
        events.append("start")
        error = next(starts)
        if error:
            raise error

    async def open_volumes(ssh, listing, extra):
        events.append("open")
        assert listing == dm.mounted_volume_devices_command("pod_x")
        return [pod]

    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr(docker_service, "open_dmcrypt_volumes", open_volumes)
    monkeypatch.setattr(docker_service, "_restore_mount_and_sshd_after_start", AsyncMock())

    await docker_service.start_existing_container(
        executor_info=Mock(address="127.0.0.1", ssh_port=2200, ssh_username="root"),
        private_key="k",
        known_hosts_policy=None,
        container_name="pod_x",
        local_volume_path="/root",
        pod_id=pod,
        default_extra={},
    )

    # the first start is refused at the mount (fail closed): nothing ran without the mapping
    assert events == ["start", "open", "start"]


@pytest.mark.parametrize("dmcrypt", [True, False])
@pytest.mark.asyncio
async def test_the_edit_swap_undo_opens_a_closed_dmcrypt_mapping_before_the_start(docker_service, monkeypatch, dmcrypt):
    """A failed edit brings the parked container back (_EditSwap.restore -> _bring_up_existing_container). Park
    stopped it; the mapping may be closed by then (a reboot meanwhile). A dm-crypt pod opens its volume, key on
    stdin only, before the one start; a gocryptfs pod starts exactly as before and runs nothing over SSH first."""
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", _MASTER)
    pod = str(uuid4())
    key = derive_dmcrypt_key(_MASTER, pod)
    events: list[str] = []

    async def run(command, **kwargs):
        events.append("open" if command == dm.dmcrypt_open_command(pod) else "list")
        if command == dm.dmcrypt_open_command(pod):
            assert kwargs["input"] == f"{key}\n"
        assert key not in command
        if command == dm.mounted_volume_devices_command("pod_x"):
            return _make_ssh_command_result(stdout=f"/dev/mapper/lium-{pod}\n")
        return _make_ssh_command_result()

    async def start(*, container_name, network=None):
        events.append("start")

    docker_client = Mock(start=start)
    monkeypatch.setattr(docker_service, "_restore_mount_and_sshd_after_start", AsyncMock())

    await docker_service._bring_up_existing_container(
        docker_client=docker_client,
        ssh_client=Mock(run=AsyncMock(side_effect=run)),
        container_name="pod_x",
        local_volume_path="/root",
        pod_id=pod,
        default_extra={},
        dmcrypt=dmcrypt,
    )

    assert events == (["list", "open", "start"] if dmcrypt else ["start"])
    docker_service._restore_mount_and_sshd_after_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_any_other_start_failure_does_not_open_anything(docker_service, monkeypatch):
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock(return_value="pkey"))
    client = docker_service.rental_docker_client_factory.client
    client.start_error = RuntimeError("No such container: pod_x")
    opened = AsyncMock()
    monkeypatch.setattr(docker_service, "open_dmcrypt_volumes", opened)

    with pytest.raises(RuntimeError):
        await docker_service.start_existing_container(
            executor_info=Mock(address="127.0.0.1", ssh_port=2200, ssh_username="root"),
            private_key="k",
            known_hosts_policy=None,
            container_name="pod_x",
            local_volume_path="/root",
            pod_id=str(uuid4()),
            default_extra={},
        )
    opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_reboot_recovery_takes_a_closed_mapping(docker_service, monkeypatch):
    pod = str(uuid4())
    recover = AsyncMock(return_value=True)
    monkeypatch.setattr(docker_service, "_recover_pod", recover)

    assert await docker_service.recover_pod_after_stale_vloopback_mount(
        ssh_client=Mock(),
        executor_info=Mock(),
        miner_hotkey="miner",
        private_key="k",
        container_name="pod_x",
        pod_id=pod,
        container_error=_CLOSED_MAPPING_ERROR.format(pod=pod),
        local_volume_path="/root",
        default_extra={},
    )
    assert recover.await_args.kwargs["dmcrypt_closed"] is True


@pytest.mark.asyncio
async def test_delete_removes_the_backing_file_after_the_volume(docker_service, monkeypatch):
    order = _wire_delete(docker_service, monkeypatch, _host_answers())
    client = docker_service.rental_docker_client_factory.client
    client.container_statuses = [None]
    payload = _delete_payload()

    await _delete(docker_service, payload)

    assert ("ssh", dm.dmcrypt_delete_command(payload.pod_id)) in order
    assert client.removed_volumes[0]["volume_name"] == "volume_dirty"


@pytest.mark.asyncio
async def test_delete_dmcrypt_volume_skips_a_non_uuid_pod(docker_service):
    ssh_client = Mock(run=AsyncMock())
    await docker_service.delete_dmcrypt_volume(ssh_client, "filler-1", {})
    ssh_client.run.assert_not_awaited()


def test_close_keeps_the_file_and_a_second_close_is_a_no_op(tmp_path):
    """The host script's close against stub tools: the first closes the mapping and detaches the loop
    device, the second finds nothing to do and still succeeds; the backing file stays."""
    pod = str(uuid4())
    bin_dir, state, crypt = tmp_path / "bin", tmp_path / "state", tmp_path / "lium-crypt"
    bin_dir.mkdir(), state.mkdir(), crypt.mkdir()
    (crypt / f"{pod}.img").write_bytes(b"")
    (state / "mapped").touch()
    (state / "loop").touch()
    stubs = {
        "dmsetup": '[ -e "$S/mapped" ]',
        "cryptsetup": 'echo "$*" >> "$S/calls"; [ "$1" = close ] && rm "$S/mapped"',
        "losetup": 'echo "losetup $*" >> "$S/calls"; case $1 in -j) [ ! -e "$S/loop" ] || echo "/dev/loop7: x";; -d) rm "$S/loop";; esac',
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    script = dm._SCRIPT.replace(dm._PATH, f"export PATH={bin_dir}:/usr/bin:/bin S={state}")

    for _ in range(2):
        done = subprocess.run(["sh", "-c", script, "lium-dmcrypt", "close", str(crypt), pod], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    assert (state / "calls").read_text().splitlines().count(f"close lium-{pod}") == 1
    assert "losetup -d /dev/loop7" in (state / "calls").read_text()
    assert (crypt / f"{pod}.img").exists()


def _lifecycle(cls, pod):
    return cls(miner_hotkey="miner", miner_address="127.0.0.1", miner_port=8000, executor_id=str(uuid4()),
               pod_id=pod, container_name="pod_x", **({"local_volume_path": "/root"} if cls is ContainerStartRequest else {}))


def _wire_lifecycle(docker_service, monkeypatch, pod, events, *, close_exit=0):
    """Stub host: tracks whether the pod's mapping is open; a start with it closed is refused at the mount."""
    host = {"open": True}

    async def run(command, **kwargs):
        if command == dm.mounted_volume_devices_command("pod_x"):
            events.append("list")
            return _make_ssh_command_result(stdout=f"<no value>\n/dev/mapper/lium-{pod}\n")
        if command == dm.dmcrypt_close_command(pod):
            events.append("close")
            host["open"] = host["open"] and close_exit != 0
            return _make_ssh_command_result(exit_status=close_exit, stderr="Device is still in use.")
        if command == dm.dmcrypt_open_command(pod):
            events.append("open")
            host["open"] = True
        return _make_ssh_command_result()

    async def start(*, container_name, network=None):
        events.append("start")
        if not host["open"]:
            raise RuntimeError(_CLOSED_MAPPING_ERROR.format(pod=pod))

    async def stop(*, container_name, stop_grace_seconds=None):
        events.append("stop")

    client = docker_service.rental_docker_client_factory.client
    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr(client, "stop", stop)
    monkeypatch.setattr(docker_service_module.settings, "VOLUME_MASTER_SECRET", _MASTER)
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock(return_value="pkey"))
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(side_effect=lambda **kw: DummySSHConnectionManager(Mock(run=AsyncMock(side_effect=run)))),
    )
    docker_service.ssh_service.decrypt_payload.return_value = "private-key"
    docker_service._prepare_known_hosts_policy = AsyncMock(return_value=None)
    monkeypatch.setattr(docker_service, "_restore_mount_and_sshd_after_start", AsyncMock())
    return host


_EXECUTOR = ExecutorSSHInfo(
    uuid=str(uuid4()), address="127.0.0.1", port=8001, ssh_username="root", ssh_port=2200,
    python_path="/usr/bin/python3", root_dir="/root/app", ssh_host_key=FAKE_SSH_HOST_KEY,
)


@pytest.mark.asyncio
async def test_stop_closes_the_mapping_and_the_next_start_reopens_it(docker_service, monkeypatch):
    pod = str(uuid4())
    events: list[str] = []
    host = _wire_lifecycle(docker_service, monkeypatch, pod, events)
    keypair = Mock(ss58_address="validator")

    assert await docker_service.stop_container(_lifecycle(ContainerStopRequest, pod), _EXECUTOR, keypair, "k") is None
    assert events == ["stop", "close"]
    assert host["open"] is False  # a stopped dm-crypt pod is not decrypted on the host

    events.clear()
    assert await docker_service.start_container(_lifecycle(ContainerStartRequest, pod), _EXECUTOR, keypair, "k") is None
    # the start is refused at the mount until the mapping is reopened: nothing runs on the closed volume
    assert events == ["start", "list", "open", "start"]
    assert host["open"] is True


@pytest.mark.asyncio
async def test_a_failed_close_fails_the_stop(docker_service, monkeypatch):
    pod = str(uuid4())
    events: list[str] = []
    _wire_lifecycle(docker_service, monkeypatch, pod, events, close_exit=5)

    result = await docker_service.stop_container(
        _lifecycle(ContainerStopRequest, pod), _EXECUTOR, Mock(ss58_address="validator"), "k"
    )

    # the container is down, but the volume would stay decrypted: report the stop failed
    assert isinstance(result, FailedContainerRequest)
    assert events == ["stop", "close"]
