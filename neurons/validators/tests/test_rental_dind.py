"""DAH-3796: Docker-in-Docker defaults for a rented sysbox pod."""

import io
import ipaddress
import json
import tarfile
from unittest.mock import Mock

import pytest
from docker.errors import APIError, NotFound

from core.config import settings
from payload_models.payloads import ContainerCreateRequest, CustomOptions, WorkloadKind
from services.docker_service import DockerService
from services.rental_dind import (
    RESERVED_POD_RANGES,
    AddressPool,
    dind_companion_volume_names,
    merge_inner_daemon_config,
    parse_address_pools,
    with_dind_companion_volumes,
)
from services.rental_docker_sdk import ContainerRunSpec, GpuDockerConfig, RentalDockerSdkClient

# computenet-docker-images templates/pytorch/daemon.json, the /etc/docker/daemon.json of every
# default "PyTorch (CUDA + DinD)" template image
PYTORCH_TEMPLATE_DAEMON_JSON = b"""{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
"""
POOLS = (AddressPool(base=ipaddress.IPv4Network("10.200.0.0/14"), size=24),)


# --- the pools -------------------------------------------------------------------------------


def test_the_default_pools_give_a_pod_1024_networks_clear_of_every_range_it_uses():
    pools = parse_address_pools(settings.RENTAL_DIND_ADDRESS_POOLS)

    assert [pool.as_daemon_json() for pool in pools] == [{"base": "10.200.0.0/14", "size": 24}]
    assert sum(pool.network_count for pool in pools) == 1024
    for pool in pools:
        assert not any(pool.base.overlaps(reserved) for reserved in RESERVED_POD_RANGES)


@pytest.mark.parametrize(
    "raw, reason",
    [
        # the host's lium-rentals pool, the sysbox installer's bip, a cluster pod's WireGuard
        ('[{"base": "172.31.0.0/16", "size": 24}]', "overlaps 172.16.0.0/12"),
        ('[{"base": "172.20.0.0/16", "size": 24}]', "overlaps 172.16.0.0/12"),
        ('[{"base": "192.168.0.0/16", "size": 20}]', "overlaps 192.168.0.0/16"),
        ('[{"base": "10.42.0.0/16", "size": 24}]', "overlaps 10.42.0.0/24"),
        ('[{"base": "10.200.0.0/14", "size": 12}]', "size must be between"),
        ('[{"base": "10.200.0.0/14", "size": 30}]', "size must be between"),
        ('[{"base": "10.200.0.0/14", "size": true}]', "size an integer"),
        ('[{"base": "10.200.0.0/14"}]', "exactly 'base' and 'size'"),
        ('[{"base": "10.200.0.0/14", "size": 24, "x": 1}]', "exactly 'base' and 'size'"),
        ('[{"base": "10.200.0.1/14", "size": 24}]', "has host bits set"),
        (
            '[{"base": "10.200.0.0/14", "size": 24}, {"base": "10.201.0.0/16", "size": 24}]',
            "overlaps pool",
        ),
        ("[]", "non-empty"),
        ('{"base": "10.200.0.0/14", "size": 24}', "non-empty JSON list"),
        ("10.200.0.0/14", "not JSON"),
    ],
)
def test_pools_that_dockerd_would_refuse_or_that_shadow_the_pod_are_rejected(raw, reason):
    with pytest.raises(ValueError, match=reason):
        parse_address_pools(raw)


# --- the pod's daemon.json -------------------------------------------------------------------


def test_the_template_daemon_json_keeps_its_nvidia_runtime_and_gains_the_pools():
    merged = json.loads(merge_inner_daemon_config(PYTORCH_TEMPLATE_DAEMON_JSON, POOLS))

    assert merged == {
        "runtimes": {"nvidia": {"args": [], "path": "nvidia-container-runtime"}},
        "default-address-pools": [{"base": "10.200.0.0/14", "size": 24}],
    }


@pytest.mark.parametrize("existing", [None, b"", b"  \n"])
def test_an_image_without_a_daemon_json_gets_one_with_only_the_pools(existing):
    assert json.loads(merge_inner_daemon_config(existing, POOLS)) == {
        "default-address-pools": [{"base": "10.200.0.0/14", "size": 24}]
    }


def test_the_lium_cluster_config_is_kept_when_its_entrypoint_merges_after_us():
    cluster = json.dumps(
        {
            "runtimes": {
                "nvidia": {"path": "nvidia-container-runtime"},
                "lium-rdma": {"path": "/usr/local/bin/lium-rdma-runc"},
            },
            "default-runtime": "lium-rdma",
        }
    ).encode()

    merged = json.loads(merge_inner_daemon_config(cluster, POOLS))

    assert merged["default-runtime"] == "lium-rdma"
    assert set(merged["runtimes"]) == {"nvidia", "lium-rdma"}
    assert merged["default-address-pools"] == [{"base": "10.200.0.0/14", "size": 24}]


@pytest.mark.parametrize(
    "existing",
    [
        b'{"default-address-pools": [{"base": "10.10.0.0/16", "size": 24}]}',  # the image chose
        b'{"runtimes": {',  # dockerd would refuse it; rewriting would hide that
        b'["not", "an", "object"]',
        b"\xff\xfe not utf-8",
    ],
)
def test_a_daemon_json_with_its_own_pools_or_that_is_not_an_object_is_left_alone(existing):
    assert merge_inner_daemon_config(existing, POOLS) is None


def test_no_pools_means_no_write():
    assert merge_inner_daemon_config(PYTORCH_TEMPLATE_DAEMON_JSON, ()) is None


# --- the companion volumes -------------------------------------------------------------------


def test_companion_volumes_are_named_after_the_pods_own_volume():
    assert dind_companion_volume_names("volume_abc") == (
        "volume_abc_docker",
        "volume_abc_workspace",
    )
    assert with_dind_companion_volumes(["volume_a", "", "volume_b"]) == [
        "volume_a",
        "volume_b",
        "volume_a_docker",
        "volume_a_workspace",
        "volume_b_docker",
        "volume_b_workspace",
    ]


# --- the SDK: seeding daemon.json between create and start -----------------------------------


def _tar_of(name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo(name)
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()


def _tar_members(data: bytes) -> dict[str, tuple[str, int, bytes | None]]:
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {
            member.name: (
                "dir" if member.isdir() else "file",
                member.mode,
                archive.extractfile(member).read() if member.isfile() else None,
            )
            for member in archive.getmembers()
        }


class _ArchiveApiClient:
    """What `_run_container_sync` calls: create, the archive endpoints, start — in that order."""

    def __init__(
        self,
        *,
        daemon_json: bytes | None = PYTORCH_TEMPLATE_DAEMON_JSON,
        get_error=None,
        put_error=None,
    ):
        self.events = []
        self.daemon_json = daemon_json
        self.get_error = get_error
        self.put_error = put_error
        self.put = []

    def create_host_config(self, **kwargs):
        return kwargs

    def create_container(self, **kwargs):
        self.events.append("create_container")
        return {"Id": "container-id"}

    def get_archive(self, container, path):
        self.events.append(("get_archive", container, path))
        if self.get_error is not None:
            raise self.get_error
        if self.daemon_json is None:
            raise NotFound(f"Could not find the file {path} in container {container}")
        return iter([_tar_of("daemon.json", self.daemon_json)]), {"name": "daemon.json"}

    def put_archive(self, container, path, data):
        self.events.append(("put_archive", container, path))
        if self.put_error is not None:
            raise self.put_error
        self.put.append((path, _tar_members(data)))
        return True

    def start(self, container):
        self.events.append("start")


def _spec(pools=POOLS) -> ContainerRunSpec:
    return ContainerRunSpec(
        image="daturaai/pytorch:dind", name="pod_x", inner_daemon_address_pools=pools
    )


@pytest.mark.asyncio
async def test_the_pools_are_merged_into_the_images_daemon_json_before_the_first_start():
    api = _ArchiveApiClient()

    await RentalDockerSdkClient(api).run_container(_spec())

    assert api.events == [
        "create_container",
        ("get_archive", "pod_x", "/etc/docker/daemon.json"),
        ("put_archive", "pod_x", "/etc/docker"),
        "start",
    ]
    [(path, members)] = api.put
    kind, mode, content = members["daemon.json"]
    assert (kind, mode) == ("file", 0o644)
    assert json.loads(content)["runtimes"]["nvidia"]["path"] == "nvidia-container-runtime"
    assert json.loads(content)["default-address-pools"] == [{"base": "10.200.0.0/14", "size": 24}]


@pytest.mark.asyncio
async def test_an_image_without_etc_docker_gets_the_directory_and_the_file():
    api = _ArchiveApiClient(daemon_json=None)

    await RentalDockerSdkClient(api).run_container(_spec())

    [(path, members)] = api.put
    assert path == "/"
    assert members["etc/docker"][:2] == ("dir", 0o755)
    assert json.loads(members["etc/docker/daemon.json"][2]) == {
        "default-address-pools": [{"base": "10.200.0.0/14", "size": 24}]
    }
    assert api.events[-1] == "start"


@pytest.mark.asyncio
async def test_an_image_that_names_its_own_pools_is_not_written():
    api = _ArchiveApiClient(
        daemon_json=b'{"default-address-pools": [{"base": "10.9.0.0/16", "size": 24}]}'
    )

    await RentalDockerSdkClient(api).run_container(_spec())

    assert api.put == []
    assert api.events[-1] == "start"


@pytest.mark.parametrize(
    "api",
    [
        _ArchiveApiClient(get_error=APIError("500 Server Error: archive read failed")),
        _ArchiveApiClient(put_error=APIError("500 Server Error: archive write failed")),
    ],
)
@pytest.mark.asyncio
async def test_a_daemon_json_that_cannot_be_read_or_written_never_fails_the_create(api, caplog):
    await RentalDockerSdkClient(api).run_container(_spec())

    assert api.events[-1] == "start"
    [record] = [r for r in caplog.records if r.getMessage() == "Inner Docker daemon address pools"]
    assert record.levelname == "WARNING"
    assert record.msg.extra["outcome"].startswith("failed: APIError")


@pytest.mark.asyncio
async def test_without_pools_the_create_does_not_touch_the_containers_filesystem():
    api = _ArchiveApiClient()

    await RentalDockerSdkClient(api).run_container(_spec(pools=()))

    assert api.events == ["create_container", "start"]


# --- the rental run spec ---------------------------------------------------------------------


@pytest.fixture
def docker_service() -> DockerService:
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


@pytest.fixture
def dind_flags(monkeypatch):
    def set_flags(*, pools=False, store=False, workspace=False, pools_json=None):
        monkeypatch.setattr(settings, "RENTAL_DIND_ADDRESS_POOLS_ENABLED", pools)
        monkeypatch.setattr(settings, "RENTAL_DIND_PERSISTENT_STORE_ENABLED", store)
        monkeypatch.setattr(settings, "RENTAL_DIND_WORKSPACE_VOLUME_ENABLED", workspace)
        if pools_json is not None:
            monkeypatch.setattr(settings, "RENTAL_DIND_ADDRESS_POOLS", pools_json)

    return set_flags


def _payload(
    *, is_sysbox=True, workload_kind=WorkloadKind.CUSTOMER_RENTAL
) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="daturaai/pytorch:dind",
        gpu_uuids=["g0"],
        is_sysbox=is_sysbox,
        workload_kind=workload_kind,
    )


def _run_spec(
    docker_service, payload, *, encrypted=False, local_volume_path="/root", external_volume=None
):
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_pod",
        custom_options=CustomOptions(),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path=local_volume_path,
        encrypted_local_volume=encrypted,
        external_volume_name=external_volume,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )


def _mounts(spec: ContainerRunSpec) -> list[tuple[str, str]]:
    return [(volume.source, volume.target) for volume in spec.volumes]


def test_with_every_flag_off_the_rental_is_built_exactly_as_before(docker_service, dind_flags):
    dind_flags()

    spec = _run_spec(docker_service, _payload(), encrypted=True)

    assert _mounts(spec) == [("volume_pod", "/lium-cipher")]
    assert spec.inner_daemon_address_pools == ()


def test_the_flags_on_give_a_plain_pod_the_pools_and_a_persistent_store(docker_service, dind_flags):
    dind_flags(pools=True, store=True, workspace=True)

    spec = _run_spec(docker_service, _payload())

    assert _mounts(spec) == [("volume_pod", "/root"), ("volume_pod_docker", "/var/lib/docker")]
    assert [pool.as_daemon_json() for pool in spec.inner_daemon_address_pools] == [
        {"base": "10.200.0.0/14", "size": 24}
    ]


def test_an_encrypted_pod_also_gets_a_bind_mountable_workspace(docker_service, dind_flags):
    dind_flags(store=True, workspace=True)

    spec = _run_spec(docker_service, _payload(), encrypted=True)

    assert _mounts(spec) == [
        ("volume_pod", "/lium-cipher"),
        ("volume_pod_docker", "/var/lib/docker"),
        ("volume_pod_workspace", "/workspace"),
    ]


def test_a_pod_whose_own_volume_is_at_workspace_keeps_it(docker_service, dind_flags):
    dind_flags(store=True, workspace=True)

    spec = _run_spec(docker_service, _payload(), local_volume_path="/workspace")

    assert _mounts(spec) == [("volume_pod", "/workspace"), ("volume_pod_docker", "/var/lib/docker")]


@pytest.mark.parametrize(
    "payload",
    [_payload(is_sysbox=False), _payload(workload_kind=WorkloadKind.FILLER)],
    ids=["runc-host", "filler"],
)
def test_no_inner_docker_no_dind_defaults(docker_service, dind_flags, payload):
    dind_flags(pools=True, store=True, workspace=True)

    spec = _run_spec(docker_service, payload, encrypted=True)

    assert _mounts(spec) == [("volume_pod", "/lium-cipher")]
    assert spec.inner_daemon_address_pools == ()


def test_invalid_pools_in_the_env_leave_docker_its_own_and_say_so(
    docker_service, dind_flags, caplog
):
    dind_flags(pools=True, pools_json='[{"base": "172.31.0.0/16", "size": 24}]')

    spec = _run_spec(docker_service, _payload())

    assert spec.inner_daemon_address_pools == ()
    assert "RENTAL_DIND_ADDRESS_POOLS is invalid" in caplog.text
