"""DAH-3980: a customer create adopts the pre-started ("warm") pod of the same id.

The fast path inspects `pod_<id>` once; only a running container with this pod's label, the rent's
image, GPUs and ports, and the root-fs quota this rent gets is adopted, in a fixed order: fillers removed and
confirmed gone, GPU power restored, volume grown and limits updated and read back, then one exec
that checks the gocryptfs mount and writes the renter's keys. Anything else removes the warm pod
(confirmed) and runs the normal create.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    CustomOptions,
    PayloadPortMapping,
    VolumeEncryptionStatus,
)
from services.docker_service import WARM_POD_LABEL, DockerService
from services.prerun_host_probe import DOCKER_PS_ALL_NAMES_CMD
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_result

GIB = 1024**3
JUPYTER_TOKEN = "warm-token"


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _rent_payload(**over):
    base = dict(
        is_sysbox=True,
        enable_volume_encryption=True,
        enable_jupyter=True,
        ships_sshd=True,
        cpu_count=4,
        memory_gb=16,
        disk_share=1.0,
        gpu_uuids=["GPU-a", "GPU-b"],
        user_public_keys=["ssh-ed25519 renter-key-1", "ssh-ed25519 renter-key-2"],
        pod_mapping=[
            PayloadPortMapping(docker_port=22, internal_port=40022, external_port=50022),
            PayloadPortMapping(docker_port=8888, internal_port=40888, external_port=50888),
            PayloadPortMapping(docker_port=20000, internal_port=40000, external_port=50000),
        ],
    )
    base.update(over)
    return _payload(**base)


def _warm_container(payload) -> dict:
    return {
        "Name": f"/pod_{payload.pod_id}",
        "Image": "sha256:image",
        "State": {"Running": True},
        "Config": {
            "Labels": {WARM_POD_LABEL: payload.pod_id},
            "Env": [f"JUPYTER_PASSWORD={JUPYTER_TOKEN}", "NVIDIA_DRIVER_CAPABILITIES=all"],
        },
        "HostConfig": {
            "DeviceRequests": [{"Driver": "", "DeviceIDs": ["GPU-a", "GPU-b"]}],
            # resolve_volume_sizing is stubbed to storage_limit_gb=20
            "StorageOpt": {"size": "20g"},
            "PortBindings": {
                "22/tcp": [{"HostIp": "", "HostPort": "40022"}],
                "8888/tcp": [{"HostIp": "", "HostPort": "40888"}],
                "20000/tcp": [{"HostIp": "", "HostPort": "40000"}],
            },
            "NanoCpus": 4_000_000_000,
            "Memory": 16 * GIB,
        },
    }


class _FakeHost:
    """The executor host behind the SSH session: a warm pod, fillers, and every command in order."""

    def __init__(self, container: dict | None, fillers: tuple[str, ...] = ()):
        self.container = container
        self.image = {"Id": "sha256:image", "RepoTags": ["daturaai/pytorch:1.0.0"], "RepoDigests": []}
        self.names = [*([container["Name"][1:]] if container else []), *fillers]
        self.events: list[str] = []
        self.keys_exit = 0
        self.grow_exit = 0
        self.slow_update = False
        self.ssh_client = AsyncMock()
        self.ssh_client.run = AsyncMock(side_effect=self.run)
        self.ssh_client.image_exists_result = True
        self.ssh_client.image_exists_error = None

    async def run(self, cmd, *args, check=False, **kwargs):
        if cmd.startswith("/usr/bin/docker inspect pod_") and " -f " not in cmd:
            inspected = [obj for obj in (self.container, self.image) if obj]
            return _ssh_result(exit_status=0 if self.container else 1, stdout=json.dumps(inspected))
        if cmd == DOCKER_PS_ALL_NAMES_CMD:
            return _ssh_result(stdout="\n".join(self.names))
        if "nsenter -t 1 -m" in cmd:
            return self._finish("grow", self.grow_exit, check)
        if "docker update" in cmd:
            if self.slow_update:
                await asyncio.sleep(0.05)
            return self._finish("update", 0, check)
        if "size-max" in cmd:
            return _ssh_result(stdout=f"{10 * GIB}\n{4_000_000_000} {16 * GIB}")
        if "docker exec -u 0 -i" in cmd and "authorized_keys" in cmd:
            self.events.append("keys")
            return _ssh_result(exit_status=self.keys_exit)
        if cmd.startswith("/usr/bin/docker rm -fv pod_"):
            self.events.append("remove_warm_pod")
            self.names = [name for name in self.names if not name.startswith("pod_")]
        return _ssh_result()

    def _finish(self, step: str, exit_status: int, check: bool):
        self.events.append(f"{step} finished")
        if check and exit_status:
            raise RuntimeError(f"{step} exited {exit_status}")
        return _ssh_result(exit_status=exit_status)

    def commands(self) -> list[str]:
        return [call.args[0] for call in self.ssh_client.run.await_args_list]

    def keys_exec_calls(self):
        return [
            call for call in self.ssh_client.run.await_args_list
            if "docker exec -u 0 -i" in call.args[0] and "authorized_keys" in call.args[0]
        ]


class _DockerConnectRecorder:
    """The Docker-SDK-over-SSH factory, noting on the host's timeline when the create connects it."""

    def __init__(self, client, host: _FakeHost):
        self.client = client
        self.host = host

    def connect(self, **kwargs):
        return self

    async def __aenter__(self):
        self.host.events.append("docker connected")
        return self.client

    async def __aexit__(self, *exc):
        return None


def _patch_host(svc, monkeypatch, host: _FakeHost) -> None:
    _patch_happy(svc, monkeypatch, host.ssh_client)
    svc.rental_docker_client_factory = _DockerConnectRecorder(svc.rental_docker_client_factory.client, host)
    # what the normal create's port step makes of the rent's pod_mapping
    monkeypatch.setattr(
        svc,
        "generate_portMappings",
        AsyncMock(return_value=(
            [(22, 40022, 50022), (8888, 40888, 50888), (20000, 40000, 50000)],
            (8888, 50888),
        )),
    )
    monkeypatch.setattr(ds_module.settings, "ENABLE_VOLUME_ENCRYPTION", True)
    monkeypatch.setattr(svc, "_image_has_encrypted_volume_label", AsyncMock(return_value=True))
    monkeypatch.setattr(svc, "setup_encrypted_local_volume", AsyncMock())

    async def restore_power(*args, **kwargs):
        host.events.append("power restored")

    monkeypatch.setattr(svc, "_restore_gpu_power_for_uncapped_pod", AsyncMock(side_effect=restore_power))

    async def remove_fillers(*args, **kwargs):
        host.events.append("fillers removed")
        host.names = [name for name in host.names if not name.startswith("filler_")]
        return []

    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(side_effect=remove_fillers))


def _adoptable(svc, monkeypatch, payload, fillers: tuple[str, ...] = ()) -> _FakeHost:
    host = _FakeHost(_warm_container(payload), fillers)
    _patch_host(svc, monkeypatch, host)
    return host


async def _assert_normal_create_after_removal(svc, host: _FakeHost, payload) -> None:
    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert "remove_warm_pod" in host.events
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    svc.create_local_volume.assert_awaited_once()


@pytest.mark.asyncio
async def test_matching_warm_pod_is_adopted_without_docker_run_or_volume_create(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    svc.create_local_volume.assert_not_awaited()
    assert "remove_warm_pod" not in host.events


@pytest.mark.asyncio
async def test_adoption_grows_the_volume_to_the_size_from_resolve_volume_sizing(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)

    await _run(svc, payload)

    grow = next(cmd for cmd in host.commands() if "nsenter -t 1 -m" in cmd)
    assert " grow 10G " in grow  # resolve_volume_sizing is stubbed to 10 GB
    assert f"volume_{payload.pod_id}" in grow
    svc.resolve_volume_sizing.assert_awaited_once()


@pytest.mark.asyncio
async def test_adoption_keys_exec_writes_exactly_the_request_keys(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)

    await _run(svc, payload)

    (keys_exec,) = host.keys_exec_calls()
    assert keys_exec.kwargs["input"] == "ssh-ed25519 renter-key-1\nssh-ed25519 renter-key-2\n"
    assert "cat > /root/.ssh/authorized_keys" in keys_exec.args[0]
    assert "fuse.gocryptfs" in keys_exec.args[0]
    assert "renter-key" not in keys_exec.args[0]


@pytest.mark.asyncio
async def test_adoption_reply_has_every_field_of_the_normal_reply(svc, monkeypatch):
    payload = _rent_payload()
    _adoptable(svc, monkeypatch, payload)

    result = await _run(svc, payload)

    assert result.container_name == f"pod_{payload.pod_id}"
    assert result.volume_name == f"volume_{payload.pod_id}"
    assert result.port_maps == [(22, 50022), (8888, 50888), (20000, 50000)]
    assert result.jupyter_url == f"http://127.0.0.1:50888/lab?token={JUPYTER_TOKEN}"
    assert result.local_volume_path == "/root"
    assert result.volume_encryption_status == VolumeEncryptionStatus.ENABLED
    assert result.volume_limit_gb == 10
    assert result.storage_limit_gb == 20
    assert result.warnings == []
    assert result.profilers[-1].name.value == "Finished in subnet."
    svc.redis_service.add_rented_pod.assert_awaited_once()


@pytest.mark.asyncio
async def test_keys_exec_runs_after_fillers_power_grow_and_update(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload, fillers=("filler_x",))
    host.container["HostConfig"]["Memory"] = 8 * GIB

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    keys_at = host.events.index("keys")
    for earlier in ("fillers removed", "power restored", "grow finished", "update finished"):
        assert host.events.index(earlier) < keys_at, host.events
    assert host.events.index("fillers removed") < host.events.index("power restored")


@pytest.mark.asyncio
async def test_no_docker_update_when_the_limits_match(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)

    await _run(svc, payload)

    assert not any("docker update" in cmd for cmd in host.commands())


@pytest.mark.asyncio
async def test_docker_update_when_the_limits_differ(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["HostConfig"]["Memory"] = 8 * GIB

    await _run(svc, payload)

    (update,) = [cmd for cmd in host.commands() if "docker update" in cmd]
    assert "--cpus 4 --memory 16g --memory-swap 32g" in update


@pytest.mark.asyncio
async def test_no_filler_removal_when_no_filler_runs(svc, monkeypatch):
    payload = _rent_payload()
    _adoptable(svc, monkeypatch, payload)

    await _run(svc, payload)

    svc.clean_existing_containers.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_pod_without_the_label_is_not_adopted(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["Config"]["Labels"] = {}

    await _assert_normal_create_after_removal(svc, host, payload)
    assert not host.keys_exec_calls()


@pytest.mark.asyncio
async def test_warm_pod_labelled_for_another_pod_is_not_adopted(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["Config"]["Labels"] = {WARM_POD_LABEL: "another-pod"}

    await _assert_normal_create_after_removal(svc, host, payload)
    assert not host.keys_exec_calls()


@pytest.mark.asyncio
async def test_stopped_warm_pod_is_not_adopted(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["State"]["Running"] = False

    await _assert_normal_create_after_removal(svc, host, payload)


@pytest.mark.asyncio
async def test_warm_pod_of_another_image_is_not_adopted(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["Image"] = "sha256:older-image"

    await _assert_normal_create_after_removal(svc, host, payload)


@pytest.mark.asyncio
async def test_warm_pod_with_another_gpu_set_is_not_adopted(svc, monkeypatch):
    payload = _rent_payload(gpu_uuids=["GPU-a"])
    host = _adoptable(svc, monkeypatch, payload)

    await _assert_normal_create_after_removal(svc, host, payload)


@pytest.mark.asyncio
async def test_warm_pod_with_the_rent_s_root_fs_quota_is_adopted_and_the_reply_carries_it(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["HostConfig"]["StorageOpt"] = {"size": "3g"}
    svc.resolve_volume_sizing.return_value = Mock(volume_limit_gb=10, storage_limit_gb=3)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert "remove_warm_pod" not in host.events
    assert result.storage_limit_gb == 3


@pytest.mark.asyncio
async def test_warm_pod_with_another_root_fs_quota_falls_back_with_reason_storage_opt(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["HostConfig"]["StorageOpt"] = {"size": "3g"}
    log_not_adopted = Mock()
    monkeypatch.setattr(svc, "_log_warm_pod_not_adopted", log_not_adopted)

    await _assert_normal_create_after_removal(svc, host, payload)
    assert log_not_adopted.call_args.args[1] == "storage_opt"
    assert not host.keys_exec_calls()


@pytest.mark.asyncio
async def test_warm_pod_and_rent_both_without_a_root_fs_quota_is_adopted(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["HostConfig"]["StorageOpt"] = None
    svc.resolve_volume_sizing.return_value = Mock(volume_limit_gb=10, storage_limit_gb=None)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert "remove_warm_pod" not in host.events
    assert result.storage_limit_gb is None


@pytest.mark.asyncio
async def test_warm_pod_with_other_ports_than_pod_mapping_is_not_adopted(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["HostConfig"]["PortBindings"]["20000/tcp"] = [{"HostIp": "", "HostPort": "40999"}]

    await _assert_normal_create_after_removal(svc, host, payload)


@pytest.mark.asyncio
async def test_rent_with_a_startup_command_takes_the_normal_create(svc, monkeypatch):
    payload = _rent_payload(custom_options=CustomOptions(startup_commands="python train.py"))
    host = _adoptable(svc, monkeypatch, payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert not any(cmd.startswith("/usr/bin/docker inspect pod_") for cmd in host.commands())
    assert not host.keys_exec_calls()


@pytest.mark.asyncio
async def test_unconfirmed_filler_removal_falls_back_without_writing_keys(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload, fillers=("filler_x",))
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=[]))  # filler survives

    await _assert_normal_create_after_removal(svc, host, payload)
    assert not host.keys_exec_calls()
    assert "grow finished" not in host.events


@pytest.mark.asyncio
async def test_lost_gocryptfs_mount_removes_the_warm_pod_and_runs_the_normal_create(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.keys_exit = 92

    await _assert_normal_create_after_removal(svc, host, payload)
    assert host.events.index("keys") < host.events.index("remove_warm_pod")
    removal = next(cmd for cmd in host.commands() if cmd.startswith("/usr/bin/docker rm -fv pod_"))
    assert f"docker volume rm volume_{payload.pod_id}" in removal


@pytest.mark.asyncio
async def test_failed_grow_removes_the_warm_pod_only_after_the_update_finished(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["HostConfig"]["NanoCpus"] = 2_000_000_000
    host.grow_exit = 1
    host.slow_update = True

    await _assert_normal_create_after_removal(svc, host, payload)
    assert host.events.index("update finished") < host.events.index("remove_warm_pod")
    assert not host.keys_exec_calls()


@pytest.mark.asyncio
async def test_rent_without_a_warm_pod_takes_the_normal_create_after_one_inspect(svc, monkeypatch):
    payload = _rent_payload()
    host = _FakeHost(container=None)
    _patch_host(svc, monkeypatch, host)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert "remove_warm_pod" not in host.events
    assert len([cmd for cmd in host.commands() if cmd.startswith("/usr/bin/docker inspect pod_")]) == 1


@pytest.mark.asyncio
async def test_grow_runs_under_the_host_timeout_shorter_than_the_local_wait(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)

    await _run(svc, payload)

    grow = next(cmd for cmd in host.commands() if "nsenter -t 1 -m" in cmd)
    assert grow.startswith("timeout -k 5 20 sh -c ")
    assert 20 + 5 < ds_module.WARM_POD_ADOPTION_COMMAND_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_warm_pod_adopted_is_not_logged_when_a_delete_cancelled_the_create(svc, monkeypatch, caplog):
    payload = _rent_payload()
    _adoptable(svc, monkeypatch, payload)
    monkeypatch.setattr(
        svc,
        "_abort_if_cancelled_by_delete",
        AsyncMock(side_effect=[None, ds_module._CreateCancelledByDelete("delete arrived")]),
    )
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", AsyncMock())

    with caplog.at_level("INFO"):
        result = await _run(svc, payload)

    assert not isinstance(result, ContainerCreated)
    assert not any("Warm pod adopted" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_adopted_rent_replies_before_the_inspector_start_and_a_delete_waits_for_it(svc, monkeypatch):
    payload = _rent_payload()
    _adoptable(svc, monkeypatch, payload)
    monkeypatch.setattr(ds_module.settings, "ENABLE_INSPECTOR", True)
    inspector_may_finish = asyncio.Event()

    async def slow_inspector_start(**kwargs):
        await inspector_may_finish.wait()

    inspector_start = AsyncMock(side_effect=slow_inspector_start)
    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start)

    # a reply that waited for the inspector would never come
    result = await asyncio.wait_for(_run(svc, payload), 1)
    # what delete_container awaits before its teardown
    delete_wait = asyncio.create_task(ds_module.create_steps_after_reply.wait_until_done(payload.pod_id, 5))
    await asyncio.sleep(0.01)
    started_before_the_delete_went_on = inspector_start.await_count
    delete_waited = not delete_wait.done()
    inspector_may_finish.set()

    assert isinstance(result, ContainerCreated), result
    assert started_before_the_delete_went_on == 1
    assert delete_waited
    assert await delete_wait is True
    assert inspector_start.await_args.kwargs["action"] == "start"
    assert [step.name.value for step in result.profilers][-2:] == [
        "Inspector collector start runs after the reply",
        "Finished in subnet.",
    ]


# The bench rent as the backend serialized it (LIUM-44 replay, untracked/scripts/lium44/rent.json),
# claiming warm pod e5cc9735-b344-4c90-afa3-7899350d94b1; available_ports cut to 2 of its 300, the
# eligibility gate does not read them.
BENCH_RENT_REQUEST_JSON = """
{
    "message_type": "ContainerCreateRequest",
    "miner_hotkey": "miner-hotkey",
    "executor_id": "d61ffebd-32b5-4429-a4c4-4dabd20a1347",
    "miner_address": null,
    "miner_port": null,
    "pod_id": "e5cc9735-b344-4c90-afa3-7899350d94b1",
    "workload_kind": "CUSTOMER_RENTAL",
    "docker_image": "daturaai/pytorch:2.11.0-py3.12-cuda12.8-devel-ubuntu24.04-dind-lium1",
    "user_public_keys": [
        "ssh-ed25519 synthetic-test-key"
    ],
    "gpu_uuids": [
        "GPU-0"
    ],
    "cpu_count": 4,
    "memory_gb": 16,
    "custom_options": {
        "volumes": null,
        "environment": null,
        "entrypoint": null,
        "internal_ports": [],
        "startup_commands": null,
        "shm_size": "6g",
        "initial_port_count": null
    },
    "debug": false,
    "local_volume": null,
    "volume_limit_gb": 60,
    "storage_limit_gb": 3,
    "disk_share": 1.0,
    "min_volume_gb": null,
    "external_volume_info": null,
    "is_sysbox": true,
    "docker_username": null,
    "docker_password": null,
    "timestamp": 1790894156110,
    "pre_dispatch_profilers": [],
    "backup_log_id": null,
    "restore_path": null,
    "bootstrap_restore": null,
    "enable_jupyter": null,
    "enable_volume_encryption": true,
    "available_ports": [
        {
            "docker_port": null,
            "internal_port": 20001,
            "external_port": 20001
        },
        {
            "docker_port": null,
            "internal_port": 20002,
            "external_port": 20002
        }
    ],
    "pod_mapping": [
        {
            "docker_port": 22,
            "internal_port": 20300,
            "external_port": 20300
        },
        {
            "docker_port": 8888,
            "internal_port": 20299,
            "external_port": 20299
        },
        {
            "docker_port": 20001,
            "internal_port": 20001,
            "external_port": 20001
        },
        {
            "docker_port": 20002,
            "internal_port": 20002,
            "external_port": 20002
        },
        {
            "docker_port": 20003,
            "internal_port": 20003,
            "external_port": 20003
        },
        {
            "docker_port": 20004,
            "internal_port": 20004,
            "external_port": 20004
        },
        {
            "docker_port": 20005,
            "internal_port": 20005,
            "external_port": 20005
        },
        {
            "docker_port": 20006,
            "internal_port": 20006,
            "external_port": 20006
        },
        {
            "docker_port": 20007,
            "internal_port": 20007,
            "external_port": 20007
        },
        {
            "docker_port": 20008,
            "internal_port": 20008,
            "external_port": 20008
        },
        {
            "docker_port": 20009,
            "internal_port": 20009,
            "external_port": 20009
        },
        {
            "docker_port": 20010,
            "internal_port": 20010,
            "external_port": 20010
        }
    ],
    "active_container_names": null,
    "active_volume_names": null,
    "cluster_membership": null,
    "dockerfile_content": null,
    "ships_sshd": true,
    "gpu_power_limits": null,
    "cache_volumes": null
}
"""


def test_bench_rent_without_the_jupyter_flag_fits_a_warm_pod(monkeypatch):
    monkeypatch.setattr(ds_module.settings, "ENABLE_VOLUME_ENCRYPTION", True)
    payload = ContainerCreateRequest.model_validate_json(BENCH_RENT_REQUEST_JSON)

    unfit_reason = ds_module._rent_unfit_for_warm_pod(payload, payload.custom_options, in_cvm=False)

    assert payload.pod_id == "e5cc9735-b344-4c90-afa3-7899350d94b1"
    assert payload.enable_jupyter is None
    assert unfit_reason is None


def test_rent_with_jupyter_explicitly_off_is_unfit_for_a_warm_pod(monkeypatch):
    monkeypatch.setattr(ds_module.settings, "ENABLE_VOLUME_ENCRYPTION", True)
    payload = ContainerCreateRequest.model_validate_json(BENCH_RENT_REQUEST_JSON).model_copy(
        update={"enable_jupyter": False}
    )

    unfit_reason = ds_module._rent_unfit_for_warm_pod(payload, payload.custom_options, in_cvm=False)

    assert unfit_reason == "jupyter_off"


@pytest.mark.asyncio
async def test_adoption_takes_over_the_warm_pod_without_connecting_the_docker_sdk(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    assert "keys" in host.events
    assert "docker connected" not in host.events


@pytest.mark.asyncio
async def test_fallback_connects_the_docker_sdk_after_the_warm_pod_removal_and_creates(svc, monkeypatch):
    payload = _rent_payload()
    host = _adoptable(svc, monkeypatch, payload)
    host.container["Config"]["Labels"] = {}

    await _assert_normal_create_after_removal(svc, host, payload)
    assert host.events.index("remove_warm_pod") < host.events.index("docker connected")


@pytest.mark.asyncio
async def test_unfit_rent_connects_ssh_and_docker_sdk_together_as_before(svc, monkeypatch):
    payload = _rent_payload(custom_options=CustomOptions(startup_commands="python train.py"))
    _adoptable(svc, monkeypatch, payload)
    connect_both = AsyncMock(wraps=svc._connect_ssh_and_docker)
    monkeypatch.setattr(svc, "_connect_ssh_and_docker", connect_both)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated), result
    connect_both.assert_awaited_once()
