"""A sysbox pod's /var/lib/docker is a capped, sparse vloopback volume, not an uncapped host volume."""

from unittest.mock import AsyncMock

import pytest
from payload_models.payloads import ContainerCreateRequest, CustomOptions
from services.docker_service import _parse_volume_host_probe
from services.rental_docker_sdk import GpuDockerConfig, _build_host_config_kwargs

from tests.test_deploy_optimizations import svc  # noqa: F401  (pytest fixture)


def _run_spec(docker_service, storage_limit_gb, local_volume_path="/root"):
    payload = ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:dind",
        gpu_uuids=["g0"],
        is_sysbox=True,
    )
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_test",
        custom_options=CustomOptions(),
        port_maps=[(22, 30022, 40022)],
        local_volume="volume_pod",
        local_volume_path=local_volume_path,
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=storage_limit_gb,
        cpu_count=None,
        docker_data_volume_driver="vloopback:v2",
    )


def test_sysbox_pod_mounts_a_capped_sparse_loopback_volume_at_var_lib_docker(svc):  # noqa: F811
    kwargs = _build_host_config_kwargs(_run_spec(svc, 40))

    [mount] = kwargs["mounts"]
    assert (
        mount["Target"] == "/var/lib/docker" and mount["Type"] == "volume" and not mount["Source"]
    )
    assert mount["VolumeOptions"]["DriverConfig"] == {
        "Name": "vloopback:v2",
        "Options": {"size": "20g", "sparse": "true"},
    }


@pytest.mark.parametrize(
    "path,storage,dind,writable",
    [
        ("/root", 40, "20g", 20),  # DinD and the writable layer split the one allowance
        ("/root", 41, "20.5g", 20.5),
        ("/root", 2, "1g", 1),
        ("/root", 1, "0.5g", 0.5),  # a 1 GiB allowance still caps DinD, not drops its mount
        ("/root", 32766, "16383g", 16383),
        ("/root", 40946, "16383g", 24563),  # DinD held at the ext4 16 TiB file limit
        ("/root", None, None, None),  # storage-opt unsupported: no cap to share
        ("/var/lib/docker/", 40, None, 40),  # the workspace already sits there: no duplicate mount
    ],
)
def test_dind_mount(svc, path, storage, dind, writable):  # noqa: F811
    spec = _run_spec(svc, storage, path)
    kwargs = _build_host_config_kwargs(spec)
    assert spec.storage_limit_gb == writable
    if dind is None:
        assert "mounts" not in kwargs
    else:
        [mount] = kwargs["mounts"]
        assert mount["VolumeOptions"]["DriverConfig"]["Options"]["size"] == dind
        assert kwargs["storage_opt"] == {"size": f"{writable}g"}


_DIND = "a1" * 32


@pytest.mark.parametrize("order", [("volume_pod", _DIND), (_DIND, "volume_pod")])
@pytest.mark.asyncio
async def test_stale_mount_repair_visits_every_volume_on_every_beat(svc, order):  # noqa: F811
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=AsyncMock(exit_status=0, stdout="\n".join(order) + "\n"))
    repair = AsyncMock(return_value=True)
    svc.repair_stale_vloopback_mountpoint = repair
    for _ in range(2):
        repair.reset_mock()
        assert (
            await svc._repair_stale_mountpoint_of_container_volume(ssh, "pod_test", {}) == order[0]
        )
        assert [call.args[1] for call in repair.await_args_list] == list(order)


def test_sizing_ignores_anonymous_vloopback_volumes():
    df = "Filesystem 1-blocks Used Available Capacity Mounted on\r/dev/vda1 1000 500 966367641600 80% /hostfs\r"
    stdout = (
        f"ROOT\t/var/lib/docker\nDF\t{df}\n"
        f"VOL\tvolume_abc\tvloopback\nVOL\t{'a1' * 32}\tvloopback\nVOLS\t0\nPLUGIN\ttrue\n"
    )

    assert _parse_volume_host_probe(stdout, with_df=True).vloopback_volume_names == ["volume_abc"]


@pytest.mark.parametrize(
    "exit_status,stdout,expected",
    [
        (0, "vloopback:v2\n", "vloopback:v2"),
        (0, "vloopback\n", "vloopback"),
        (0, "local\n", None),  # an unlimited workspace: nothing to put inner Docker on
        (1, "", None),
    ],
)
@pytest.mark.asyncio
async def test_reused_volume_driver(svc, exit_status, stdout, expected):  # noqa: F811
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=AsyncMock(exit_status=exit_status, stdout=stdout))
    assert await svc._vloopback_driver_of_volume(ssh, "volume_pod", {}) == expected
