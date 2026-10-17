"""DAH-3257 — rental pre-run host probe (RENTAL_PRERUN_HOST_PROBE_ENABLED).

Before `docker run`, a rent read eight host listings over eight serial SSH commands (containers,
volumes, mounted volumes, volume names again, GPU minor map, shared device nodes, nvidia-smi power
state, the image's encryption label). With the flag on, one probe command returns them all; every
consumer reads its section and keeps its removals / writes as they are. Covered here (device, renter-isolation and filler-removal cases only):

- the shared device-node command and its parser section; the GPU device config from a probe
  matches the live one (partial and whole-host rentals, missing UUID, failed section);
- a multi-line image label cannot forge another probe section;
- a filler's GPU power cap never survives into the renter's docker run;
- a customer create that removed only a filler, cleanly, keeps the probe's listings and relists
  nothing, but sizes the volume on a fresh volume probe; any other removal, a survivor or a
  failed rm relists as before; a removal that times out fails the create at the cleanup step;
- a customer create removes its fillers as soon as the SSH session is up; the cleanup step awaits
  that removal and never removes those fillers again.
"""

from __future__ import annotations

import os
import stat
import subprocess
import asyncio
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest
from core.config import settings
from payload_models.payloads import WorkloadKind
from services import nvidia_devices as nd
from services.docker_service import (
    DOCKER_FILLER_VOLUME_LS_NAME_DRIVER_CMD,
    DockerService,
    _ENCRYPTED_VOLUME_IMAGE_LABEL,
    VolumeHostProbe,
    _remove_and_list_containers_command,
)
import services.docker_service as ds_module
from services.nvidia_devices import (
    PROC_GPU_INFO_CMD,
    build_gpu_docker_config_for_executor,
    shared_device_nodes_command,
)
from services.prerun_host_probe import (
    DOCKER_MOUNTED_VOLUME_NAMES_CMD,
    DOCKER_PS_ALL_NAMES_IDS_CMD,
    DOCKER_VOLUME_LS_NAME_DRIVER_CMD,
    PrerunHostProbe,
    ProbedVolume,
    parse_prerun_host_probe,
    port_check_containers_command,
    prerun_host_probe_command,
)
from test_deploy_optimizations import (
    _docker_client,
    _patch_happy,
    _payload as _deploy_payload,
    _run as _run_create_container,
    _ssh_client as _deploy_ssh_client,
    _ssh_result,
)

_IMAGE = "daturaai/pytorch:1.0.0"
_HOTKEY = "5HotkeyOfTheMiner"


def _probe(**over) -> PrerunHostProbe:
    base = dict(
        container_names=(),
        volumes=(),
        mounted_volume_names=(),
        gpu_minor_map_stdout="",
        gpu_device_nodes=(),
        shared_nodes=(),
        shared_nodes_whole_host_only=(),
        power_state_stdout="",
        image_label_value="",
        port_check_container_names=(),
    )
    base.update(over)
    return PrerunHostProbe(**base)


def _tagged(tag: str, *lines: str, rc: int = 0) -> str:
    return "".join(f"{tag}\t{line}\n" for line in lines) + f"{tag}_RC\t{rc}\n"


def _stdout(
    *,
    ps: tuple[str, ...] = ("pod_a", "other"),
    ps_rc: int = 0,
    vol: tuple[str, ...] = ("volume_a vloopback:latest", "dphn_cache_x local"),
    vol_rc: int = 0,
    mnt: tuple[str, ...] = ("volume_a",),
    mnt_rc: int = 0,
    gpu_minor_map: tuple[str, ...] = ("GPU-1, 0", "GPU-2, 1"),
    gpu_minor_map_rc: int = 0,
    gpudev: tuple[str, ...] = ("/dev/nvidia0", "/dev/nvidia1"),
    shared: tuple[str, ...] = ("/dev/nvidiactl", "/dev/nvidia-uvm"),
    sharedw: tuple[str, ...] = ("/dev/infiniband/uverbs0", "/dev/nvidia-caps/nvidia-cap1"),
    power: tuple[str, ...] | None = ("GPU-1, 300.00, 450.00, 100.00, 450.00",),
    power_rc: int = 0,
    label: tuple[str, ...] = ("1",),
    label_rc: int = 0,
    port_check: tuple[str, ...] = (),
    port_check_rc: int = 0,
) -> str:
    out = (
        _tagged("PS", *ps, rc=ps_rc)
        + _tagged("VOL", *vol, rc=vol_rc)
        + _tagged("MNT", *mnt, rc=mnt_rc)
        + _tagged("GPUMINORMAP", *gpu_minor_map, rc=gpu_minor_map_rc)
        + _tagged("GPUDEV", *gpudev)
        + _tagged("SHARED", *shared)
        + _tagged("SHAREDW", *sharedw)
    )
    if power is not None:
        out += _tagged("POWER", *power, rc=power_rc)
    out += _tagged("LABEL", *label, rc=label_rc)
    out += _tagged("PORTCHECK", *port_check, rc=port_check_rc)
    return out


def _ssh(*results):
    client = AsyncMock()
    client.run = AsyncMock(side_effect=list(results))
    return client


def _cmds(client) -> list[str]:
    return [c.args[0] for c in client.run.await_args_list if c.args]


# ------------------------------------------------------------------
# the command
# ------------------------------------------------------------------


def test_shared_device_nodes_command_whole_host_only_is_the_tail_of_the_whole_host_command():
    partial = shared_device_nodes_command(is_whole_host_rental=False)
    whole = shared_device_nodes_command(is_whole_host_rental=True)
    tail = shared_device_nodes_command(is_whole_host_rental=True, whole_host_only=True)
    assert "/dev/infiniband" not in partial and "nvidia-caps" not in partial
    assert (
        "/dev/infiniband/uverbs[0-9]* /dev/infiniband/rdma_cm" in whole
        and "find /dev/nvidia-caps" in whole
    )
    assert tail.startswith("for p in /dev/infiniband/uverbs[0-9]* /dev/infiniband/rdma_cm; do")
    assert "find /dev/nvidia-caps" in tail
    assert "/dev/nvidiactl" not in tail


# ------------------------------------------------------------------
# the parser
# ------------------------------------------------------------------


def test_parser_device_node_sections_ignore_the_exit_status_like_the_per_command_path():
    # `for p in …; do [ -e "$p" ] && …; done` exits 1 when the last node is absent; the live path
    # never read that status, so an exit 1 with nodes listed is still a listing (and an empty one
    # is "no nodes", not a failure)
    stdout = (
        _stdout()
        .replace("GPUDEV_RC\t0", "GPUDEV_RC\t2")
        .replace("SHARED_RC\t0", "SHARED_RC\t1")
        .replace("SHAREDW_RC\t0", "SHAREDW_RC\t1")
    )
    probe = parse_prerun_host_probe(stdout, with_power=True)
    assert probe.gpu_device_nodes == ("/dev/nvidia0", "/dev/nvidia1")
    assert probe.shared_nodes_for(is_whole_host_rental=True) == (
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/infiniband/uverbs0",
        "/dev/nvidia-caps/nvidia-cap1",
    )


# ------------------------------------------------------------------
# the command and the parser together (sh -c, docker stub)
# ------------------------------------------------------------------


_DOCKER_STUB = """#!/bin/sh
case "$1 $2" in
  "ps -a")
    if [ "$3" = "-q" ]; then printf 'c1\\nc2\\n'; else printf 'pod_a\\nfiller_b\\nother\\n'; fi ;;
  "volume ls") printf 'volume_a vloopback:latest\\nvolume_b local\\n' ;;
  "inspect --format") printf 'volume_a\\n\\nvolume_a\\n' ;;
  "image inspect") printf '%s\\n' "$LABEL_VALUE" ;;
  "ps --format") printf 'health_check_1\\n' ;;
  *) echo "unexpected: $*" >&2; exit 9 ;;
esac
"""


_NVIDIA_SMI_STUB = (
    "#!/bin/sh\nexit 1\n"  # a host without a working driver: the section fails, never lists
)


def _run_probe_in_sh(tmp_path, *, label_value: str = "1", with_power: bool = True) -> str:
    stub = tmp_path / "docker"
    stub.write_text(_DOCKER_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    # PATH starts with tmp_path, so this stub shadows a real nvidia-smi on a GPU test host
    smi = tmp_path / "nvidia-smi"
    smi.write_text(_NVIDIA_SMI_STUB)
    smi.chmod(smi.stat().st_mode | stat.S_IXUSR)
    cmd = prerun_host_probe_command(
        docker_image=_IMAGE,
        image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL,
        miner_hotkey=_HOTKEY,
        with_power=with_power
    )
    cmd = cmd.replace("/usr/bin/docker", str(stub))
    env = {**os.environ, "LABEL_VALUE": label_value, "PATH": f"{tmp_path}:/usr/bin:/bin"}
    return subprocess.run(
        ["sh", "-c", cmd], capture_output=True, text=True, env=env, check=False
    ).stdout


def test_probe_through_sh_multiline_label_cannot_forge_another_section(tmp_path):
    # every line of a command's output gets that command's tag — an image label cannot inject a PS line
    probe = parse_prerun_host_probe(
        _run_probe_in_sh(tmp_path, label_value="1\nPS\tpod_evil"), with_power=True
    )
    assert probe.container_names == ("pod_a", "filler_b", "other")
    assert probe.image_label_value == "1\nPS\tpod_evil"
    assert probe.image_label_value != "1"


# ------------------------------------------------------------------
# consumers: same result, no listing command
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpu_config_with_probe_matches_live_partial_rental(monkeypatch):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False)
    proc = "GPU-1, 0\nGPU-2, 1\n"
    live = _ssh(_ssh_result(stdout=proc), _ssh_result(stdout="/dev/nvidiactl\n/dev/nvidia-uvm\n"))
    from_live = await build_gpu_docker_config_for_executor(live, ["GPU-2"])
    probe = _probe(
        gpu_minor_map_stdout=proc,
        shared_nodes=("/dev/nvidiactl", "/dev/nvidia-uvm"),
        shared_nodes_whole_host_only=("/dev/infiniband/uverbs0", "/dev/nvidia-caps/nvidia-cap1"),
    )
    probed = _ssh()
    from_probe = await build_gpu_docker_config_for_executor(probed, ["GPU-2"], host_probe=probe)
    assert from_probe == from_live
    assert [d.path_on_host for d in from_probe.device_mounts] == [
        "/dev/nvidia1",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
    ]
    assert len(_cmds(live)) == 2 and _cmds(probed) == []


@pytest.mark.asyncio
async def test_gpu_config_with_probe_whole_host_gets_the_whole_host_nodes(monkeypatch):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False)
    probe = _probe(
        gpu_minor_map_stdout="GPU-1, 0\nGPU-2, 1\n",
        gpu_device_nodes=("/dev/nvidia0", "/dev/nvidia1"),
        shared_nodes=("/dev/nvidiactl",),
        shared_nodes_whole_host_only=("/dev/infiniband/uverbs0", "/dev/nvidia-caps/nvidia-cap1"),
    )
    ssh = _ssh()
    by_uuid = await build_gpu_docker_config_for_executor(ssh, ["GPU-1", "GPU-2"], host_probe=probe)
    whole_node = await build_gpu_docker_config_for_executor(ssh, None, host_probe=probe)
    expected = [
        "/dev/nvidia0",
        "/dev/nvidia1",
        "/dev/nvidiactl",
        "/dev/infiniband/uverbs0",
        "/dev/nvidia-caps/nvidia-cap1",
    ]
    assert [d.path_on_host for d in by_uuid.device_mounts] == expected
    assert [d.path_on_host for d in whole_node.device_mounts] == expected
    assert _cmds(ssh) == []


@pytest.mark.asyncio
async def test_gpu_config_with_probe_missing_uuid_still_consults_xml_live(monkeypatch):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False)
    probe = _probe(gpu_minor_map_stdout="GPU-1, 0\n", shared_nodes=("/dev/nvidiactl",))
    ssh = _ssh(_ssh_result(exit_status=1, stderr="no nvidia-smi"))
    config = await build_gpu_docker_config_for_executor(ssh, ["GPU-9"], host_probe=probe)
    # the kernel map lacks GPU-9 → nvidia-smi XML is asked live → fails → legacy --gpus-only fallback
    assert _cmds(ssh) == ["nvidia-smi -q -x"]
    assert config.device_mounts == ()


@pytest.mark.asyncio
async def test_gpu_config_failed_proc_section_queries_live(monkeypatch):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False)
    probe = _probe(gpu_minor_map_stdout=None, shared_nodes=("/dev/nvidiactl",))
    ssh = _ssh(_ssh_result(stdout="GPU-1, 0\n"))
    config = await build_gpu_docker_config_for_executor(ssh, ["GPU-1"], host_probe=probe)
    assert _cmds(ssh) == [PROC_GPU_INFO_CMD]
    assert [d.path_on_host for d in config.device_mounts] == ["/dev/nvidia0", "/dev/nvidiactl"]


# ------------------------------------------------------------------
# create_container wiring
# ------------------------------------------------------------------


def _wire(svc, monkeypatch, ssh_client, *, probe_result):
    _patch_happy(svc, monkeypatch, ssh_client)
    # so a payload with is_sysbox + enable_volume_encryption reaches the label check
    monkeypatch.setattr(settings, "ENABLE_VOLUME_ENCRYPTION", True)
    monkeypatch.setattr(svc, "probe_prerun_host", AsyncMock(return_value=probe_result))
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "clean_stale_vloopback_volumes", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "sweep_stale_cache_volumes", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "select_affordable_cache_volumes", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "reclaim_dphn_cache_for_rental", AsyncMock())
    monkeypatch.setattr(svc, "_image_has_encrypted_volume_label", AsyncMock(return_value=False))
    monkeypatch.setattr(
        "services.docker_service.restore_tracked_gpu_power_limits", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        "services.docker_service.restore_all_host_gpu_power_limits", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        "services.docker_service.raise_low_power_limits_to_default", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(svc, "add_ssh_public_keys_with_rental_docker", AsyncMock())
    monkeypatch.setattr(
        svc, "add_environment_variables_with_rental_docker", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", AsyncMock())


def _probe_kwarg(mock) -> object:
    return mock.await_args.kwargs.get("host_probe")


@pytest.mark.asyncio
async def test_a_filler_cap_after_the_early_gpu_power_restore_is_restored_before_docker_run(svc_fixture, monkeypatch):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    _wire(svc, monkeypatch, _deploy_ssh_client(), probe_result=_probe())
    events: list[str] = []
    early_restore_done = asyncio.Event()

    async def restore(*_args, **_kwargs) -> int:
        events.append("restore")
        early_restore_done.set()
        return 0

    async def volume_create_while_a_filler_caps(*_args, **_kwargs) -> None:
        await asyncio.wait_for(early_restore_done.wait(), 1)
        events.append("filler cap")

    monkeypatch.setattr("services.docker_service.restore_tracked_gpu_power_limits", AsyncMock(side_effect=restore))
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock(side_effect=volume_create_while_a_filler_caps))
    svc._run_rental_docker_create_with_port_retry.side_effect = lambda *_a, **_k: events.append("docker run")

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert events == ["restore", "filler cap", "restore", "docker run"]


@pytest.mark.asyncio
async def test_a_power_limit_lowered_without_a_record_after_the_early_raise_is_raised_before_docker_run(
    svc_fixture, monkeypatch
):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    _wire(svc, monkeypatch, _deploy_ssh_client(), probe_result=_probe())
    events: list[str] = []
    early_raise_done = asyncio.Event()

    async def live_raise(*_args, **_kwargs) -> int:
        events.append("live raise")
        early_raise_done.set()
        return 0

    async def volume_create_while_the_host_lowers_a_limit(*_args, **_kwargs) -> None:
        await asyncio.wait_for(early_raise_done.wait(), 1)
        events.append("limit lowered, no record")

    monkeypatch.setattr("services.docker_service.raise_low_power_limits_to_default", AsyncMock(side_effect=live_raise))
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock(side_effect=volume_create_while_the_host_lowers_a_limit))
    svc._run_rental_docker_create_with_port_retry.side_effect = lambda *_a, **_k: events.append("docker run")

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert events == ["live raise", "limit lowered, no record", "live raise", "docker run"]


def _wire_early_probes(svc, monkeypatch, *, image_present: bool) -> list[int]:
    """Both probes on; returns how many host probes had started when the image inspect answered."""
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    _wire(svc, monkeypatch, _deploy_ssh_client(inspect_exit=0 if image_present else 1), probe_result=_probe())
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=None))
    monkeypatch.setattr(svc, "reclaim_dphn_cache_for_rental", AsyncMock(return_value=False))
    docker_client = _docker_client(svc)
    inspect_image = docker_client.local_image_repo_digests
    host_probes_started_at_inspect: list[int] = []

    async def slow_inspect(*, image):
        await asyncio.sleep(0.01)
        host_probes_started_at_inspect.append(svc.probe_prerun_host.await_count)
        return await inspect_image(image=image)

    docker_client.local_image_repo_digests = slow_inspect
    return host_probes_started_at_inspect


def _wire_customer_create_over_the_host(
    svc,
    monkeypatch,
    *,
    probe: PrerunHostProbe,
    docker_rm_exit: int = 0,
    ps_after_rm: str = "",
    docker_rm_raises: Exception | None = None,
    listing_after_rm_exit: int = 0,
    docker_rm_seconds: float = 0,
    containers_on_host: tuple[str, ...] | None = None,
    container_ids: dict[str, str] | None = None,
    listings_that_answer: int | None = None,
) -> AsyncMock:
    """Both early probes on; the cleanup, the sweeps and the port-check wait are the real
    ones over a stub SSH client, so every listing they run is a command on it. The host lists the
    probe's containers until the first rm, then ``ps_after_rm``."""
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    answer_as_the_deploy_host = ssh_client.run.side_effect
    names_on_host = list(probe.container_names or () if containers_on_host is None else containers_on_host)
    listings = 0

    async def answer(cmd, *args, **kwargs):
        nonlocal names_on_host, listings
        if "/usr/bin/docker rm -fv" in cmd:
            await asyncio.sleep(docker_rm_seconds)
            if docker_rm_raises is not None:
                raise docker_rm_raises
            names_on_host = ps_after_rm.split()
            # the removal command reports the rm's status and the names left after it
            names_after = "".join(f"NAME\t{name}\n" for name in names_on_host)
            return _ssh_result(stdout=f"RM\t{docker_rm_exit}\n{names_after}PS\t{listing_after_rm_exit}\n")
        if cmd == DOCKER_PS_ALL_NAMES_IDS_CMD:
            listings += 1
            if listings_that_answer is not None and listings > listings_that_answer:
                return _ssh_result(exit_status=1)
            ids = container_ids or {}
            return _ssh_result(stdout="".join(f"{name} {ids.get(name, '')}\n" for name in names_on_host))
        return answer_as_the_deploy_host(cmd, *args, **kwargs)

    ssh_client.run.side_effect = answer
    _wire(svc, monkeypatch, ssh_client, probe_result=probe)
    monkeypatch.setattr(svc, "probe_volume_host", AsyncMock(return_value=None))
    for real in (
        "clean_existing_containers",
        "clean_stale_vloopback_volumes",
        "reclaim_dphn_cache_for_rental",
        "wait_for_port_check_containers",
    ):
        monkeypatch.delattr(svc, real)
    return ssh_client


def _probe_with_containers(*names: str) -> PrerunHostProbe:
    # each container mounts its volume_<id>, a vloopback volume
    volume_names = [f"volume_{name.split('_', 1)[1]}" for name in names]
    return _probe(
        container_names=names,
        volumes=tuple(ProbedVolume(name=name, driver="vloopback:latest") for name in volume_names),
        mounted_volume_names=tuple(volume_names),
    )


def _relisting_commands(ssh_client) -> list[str]:
    return [
        cmd
        for cmd in _cmds(ssh_client)
        if "docker volume ls" in cmd
        or cmd == DOCKER_MOUNTED_VOLUME_NAMES_CMD
        or cmd == port_check_containers_command("miner")
    ]


def _fresh_sizing_payload(**over):
    # the DAH-2183 contract: the volume is sized on the host's free disk
    return _deploy_payload(disk_share=0.5, volume_limit_gb=1000, min_volume_gb=2, **over)


@pytest.mark.asyncio
# the last row: the removal at SSH connect ran its rm, then a MaxSessions=1 sshd busy with the probes refused it a channel
@pytest.mark.parametrize(("probe_saw_the_filler", "removal_lost_its_channel"), [(True, False), (False, False), (False, True)],
                         ids=["probe_listed_the_filler", "probe_listed_after_the_removal", "removal_lost_its_channel_after_its_rm"])
async def test_customer_create_keeps_the_listings_but_not_the_volume_facts_after_removing_only_a_filler(
    svc_fixture, monkeypatch, probe_saw_the_filler, removal_lost_its_channel
):
    """Nothing is listed again: whether the probe listed the host before or after the removal at SSH connect, it hands
    on a listing without the filler. The rm freed disk, so the volume is sized on a live probe, never the early one."""
    svc = svc_fixture
    gb = ds_module._FRESH_SIZING_GB_BYTES
    probe = _probe_with_containers(*(["filler_x"] if probe_saw_the_filler else []), "pod_keep")
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=probe, containers_on_host=("filler_x", "pod_keep")
    )
    svc.probe_volume_host = AsyncMock(
        side_effect=[VolumeHostProbe("/var/lib/docker", size * gb, [], True) for size in (100, 140)]
    )
    monkeypatch.delattr(svc, "resolve_volume_sizing")
    if removal_lost_its_channel:
        svc.remove_fillers_at_ssh_connect = AsyncMock(side_effect=asyncssh.ChannelOpenError(1, "open failed"))
    # the backend protects a preempted filler's volume until its own filler delete
    payload = _fresh_sizing_payload(
        active_container_names=["pod_keep"], active_volume_names=["volume_keep", "volume_x"]
    )

    result = await _run_create_container(svc, payload)

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    # the fail-closed pid_max read every rental makes first, then the listing that finds the filler and the
    # rm with its confirming listing in one command — the removal starts only once the host reads succeed
    assert _cmds(ssh_client) == [
        "cat /proc/sys/kernel/pid_max",
        *([] if removal_lost_its_channel else [DOCKER_PS_ALL_NAMES_IDS_CMD, _FILLER_X_REMOVAL]),
    ]
    assert _relisting_commands(ssh_client) == []
    handed_on = _probe_kwarg(svc.select_affordable_cache_volumes)
    assert (handed_on.container_names, handed_on.mounted_volume_names) == (("pod_keep",), ("volume_keep",))
    svc.probe_prerun_host.assert_awaited_once()
    # 0.5 x (free - 20 GB overhead), two thirds of it the volume: the early 100 GB -> 26, the live 140 GB -> 40
    assert svc.create_local_volume.await_args.kwargs["limit"] == 40


_FILLER_X_REMOVAL = _remove_and_list_containers_command(["filler_x"], [])
_POD_OLD_REMOVAL = _remove_and_list_containers_command(["pod_old"], ["volume_old"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("containers", "active_volume_names", "docker_rm_exit", "ps_after_rm", "listing_after_rm_exit", "removals"),
    [
        (("filler_x", "pod_old"), ["volume_x"], 0, "", 0, [_FILLER_X_REMOVAL, _POD_OLD_REMOVAL]),  # a pod beside the filler
        (("filler_x",), [], 0, "", 0, [_remove_and_list_containers_command(["filler_x"], ["volume_x"])]),  # the filler's volume
        (("filler_x",), ["volume_x"], 0, "filler_x\n", 0, [_FILLER_X_REMOVAL]),  # the filler survived the rm
        (("filler_x",), ["volume_x"], 1, "", 0, [_FILLER_X_REMOVAL]),  # the rm failed although the filler is gone
        (("filler_x",), ["volume_x"], 0, "", 1, [_FILLER_X_REMOVAL]),  # the confirming listing failed
    ],
)
async def test_customer_create_relists_after_any_other_removal(
    svc_fixture, monkeypatch, containers, active_volume_names, docker_rm_exit, ps_after_rm, listing_after_rm_exit, removals
):
    """Every removal but a clean filler-only one lists the host again; nothing is removed twice."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc,
        monkeypatch,
        probe=_probe_with_containers(*containers),
        docker_rm_exit=docker_rm_exit,
        ps_after_rm=ps_after_rm,
        listing_after_rm_exit=listing_after_rm_exit,
    )

    result = await _run_create_container(
        svc, _deploy_payload(active_volume_names=active_volume_names)
    )

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _removal_commands(ssh_client) == removals
    # the stub host lists no volume, so the vloopback sweep stops before its mounted-volume listing
    assert _relisting_commands(ssh_client) == [
        DOCKER_VOLUME_LS_NAME_DRIVER_CMD,
        '/usr/bin/docker volume ls --format "{{.Name}}"',
        port_check_containers_command("miner"),
    ]
    assert svc.probe_volume_host.await_count == 2


@pytest.mark.asyncio
async def test_customer_create_whose_filler_removal_times_out_fails_at_the_cleanup_step(
    svc_fixture, monkeypatch
):
    """A hung dockerd fails the create where a failed cleanup does, never hangs it."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc,
        monkeypatch,
        probe=_probe_with_containers("filler_x"),
        docker_rm_raises=asyncssh.TimeoutError(None, None, None, None, None, None, "", ""),
    )

    result = await _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "FailedContainerRequest"
    assert result.failure_step == "container_cleanup"
    [removal] = [call for call in ssh_client.run.await_args_list if call.args[0].startswith("/usr/bin/docker rm -fv")]
    assert removal.kwargs["timeout"] == ds_module._CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS


def _removal_commands(ssh_client) -> list[str]:
    return [cmd for cmd in _cmds(ssh_client) if cmd.startswith("/usr/bin/docker rm -fv")]


_REMOVED_AT_SSH_CONNECT_ID = "a" * 64
_RECREATED_ID = "b" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("id_in_cleanup_listing", "removed_ids", "docker_rm_seconds", "probes_seconds", "after_the_first_removal"),
    [
        # the instance removed at SSH connect, in a listing read before that removal: not removed twice
        (_REMOVED_AT_SSH_CONNECT_ID, [_REMOVED_AT_SSH_CONNECT_ID], 0, 0, []),
        # a listing without IDs: the removal at SSH connect still targets the ID it listed, not the name
        (None, [_REMOVED_AT_SSH_CONNECT_ID], 0, 0, []),
        # a filler created again under the same name since: it goes too, by its own ID
        (_RECREATED_ID, [_REMOVED_AT_SSH_CONNECT_ID, _RECREATED_ID], 0, 0, []),
        # the kill is issued at SSH connect: confirmed before the image inspect and the probes finish
        (None, [_REMOVED_AT_SSH_CONNECT_ID], 0, 0.05, ["host probed", "image inspected"]),
        # safety order: a PEARL filler is gone before its cap is lifted (beside the volume create, and again
        # right before docker run), and the Dolphin cache is reclaimed only once the filler that mounts it is gone
        (None, [_REMOVED_AT_SSH_CONNECT_ID], 0.05, 0, ["cache reclaim", "docker run", "power restore", "power restore"]),
    ],
    ids=["same_instance", "cleanup_listing_without_ids", "recreated_filler", "before_the_image_inspect_and_the_probes_finish",
         "before_power_restore_cache_reclaim_and_docker_run"],
)
async def test_the_filler_removed_at_ssh_connect_goes_first_and_only_once(
    svc_fixture, monkeypatch, id_in_cleanup_listing, removed_ids, docker_rm_seconds, probes_seconds, after_the_first_removal
):
    svc = svc_fixture
    probe = _probe_with_containers("filler_x")
    if id_in_cleanup_listing:
        probe.container_ids["filler_x"] = id_in_cleanup_listing
    ssh_client = _wire_customer_create_over_the_host(
        svc,
        monkeypatch,
        probe=probe,
        container_ids={"filler_x": _REMOVED_AT_SSH_CONNECT_ID},
        docker_rm_seconds=docker_rm_seconds,
    )
    events: list[str] = []  # each removal command and step, once it answered
    host = ssh_client.run.side_effect

    async def run(cmd, *args, **kwargs):
        answer = await host(cmd, *args, **kwargs)
        events.extend([cmd] if cmd.startswith("/usr/bin/docker rm -fv") else [])
        return answer

    def recorded(event: str, call, seconds: float = 0):
        async def record(*args, **kwargs):
            await asyncio.sleep(seconds)
            answer = await call(*args, **kwargs)
            events.append(event)
            return answer

        return record

    ssh_client.run.side_effect = run
    docker_client = _docker_client(svc)
    docker_client.local_image_repo_digests = recorded(
        "image inspected", docker_client.local_image_repo_digests, probes_seconds
    )
    svc.probe_prerun_host = AsyncMock(side_effect=recorded("host probed", svc.probe_prerun_host, probes_seconds))
    svc._restore_gpu_power_for_uncapped_pod = recorded("power restore", AsyncMock())
    svc.reclaim_dphn_cache_for_rental = recorded("cache reclaim", AsyncMock(return_value=False))
    svc._run_rental_docker_create_with_port_retry = recorded("docker run", AsyncMock())

    result = await _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    removals = [_remove_and_list_containers_command([removed_id], []) for removed_id in removed_ids]
    assert [event for event in events if event.startswith("/usr/bin/docker rm -fv")] == removals
    assert events.index("docker run") > events.index(removals[-1])
    after = events[events.index(removals[0]) + 1 :]
    assert sorted(event for event in after if event in after_the_first_removal) == after_the_first_removal


@pytest.mark.asyncio
async def test_a_create_cancelled_at_ssh_connect_never_starts_the_filler_removal(svc_fixture, monkeypatch, caplog):
    """The removal is irreversible, so it starts only after the fail-closed host reads. A create cancelled
    at SSH connect (before those reads) must never have removed the filler, or a cancel would strand the
    host with neither workload — the ordering the review required at 65b84b3."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(svc, monkeypatch, probe=_probe_with_containers("filler_x"))

    async def deleted(*args, **kwargs):
        raise RuntimeError("deleted")

    monkeypatch.setattr(svc, "_abort_if_cancelled_by_delete", AsyncMock(side_effect=deleted))

    result = await asyncio.wait_for(
        _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"])), timeout=2
    )

    assert type(result).__name__ == "FailedContainerRequest"
    assert result.failure_step == "ssh_connect"
    # the filler's docker rm never ran, no removal task lingers, and nothing is logged about a removal
    assert not [cmd for cmd in _cmds(ssh_client) if "/usr/bin/docker rm -fv" in cmd]
    assert not [
        task for task in asyncio.all_tasks() if "remove_fillers_at_ssh_connect" in repr(task.get_coro()) and not task.done()
    ]
    assert not [
        record for record in caplog.records
        if getattr(record.msg, "message", None) == "Filler removal at SSH connect failed before the cleanup step"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("container", "payload_fields", "bootstrap_restore", "commands"),
    [
        # nothing to remove: the commands of before, plus the removal's one read-only listing beside the probes
        ("pod_keep", {"active_container_names": ["pod_keep"], "active_volume_names": ["volume_keep"]}, False, [DOCKER_PS_ALL_NAMES_IDS_CMD]),
        # a filler create keeps protecting its sibling bundle (DAH-2465) and lists nothing at SSH connect; its
        # volume sweep lists the filler-labelled volumes, which the probe's listing cannot tell apart
        ("filler_sibling", {"workload_kind": WorkloadKind.FILLER, "active_container_names": ["filler_sibling"], "active_volume_names": ["volume_sibling"]}, False, [DOCKER_FILLER_VOLUME_LS_NAME_DRIVER_CMD]),
        # a bootstrap restore removes the filler at the cleanup step
        ("filler_x", {"active_volume_names": ["volume_x"]}, True, [_FILLER_X_REMOVAL]),
    ],
    ids=["customer_without_a_filler", "filler_create", "bootstrap_restore"],
)
async def test_only_a_customer_create_removes_fillers_at_ssh_connect(
    svc_fixture, monkeypatch, container, payload_fields, bootstrap_restore, commands
):
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(svc, monkeypatch, probe=_probe_with_containers(container))
    monkeypatch.setattr(svc, "_run_bootstrap_restore", AsyncMock())
    payload = _deploy_payload(**payload_fields)
    payload.bootstrap_restore = Mock(restore_log_id="restore-log") if bootstrap_restore else None

    result = await _run_create_container(svc, payload)

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    # the fail-closed pid_max read every rental makes comes first; only then does the removal run (at SSH
    # connect for a customer create, at the cleanup step for a bootstrap restore), so it is always after
    assert _cmds(ssh_client) == ["cat /proc/sys/kernel/pid_max", *commands]
    svc.probe_prerun_host.assert_awaited_once()
    # the restore's removal at the cleanup step changed the host: its volume facts are probed again
    assert svc.probe_volume_host.await_count == (2 if bootstrap_restore else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bootstrap_restore", "raised_during_volume_creation", "live_raises"), [(False, 1, 2), (True, 0, 1)]
)
async def test_uncapped_pod_gets_gpu_power_back_while_its_volume_is_created(
    svc_fixture, monkeypatch, bootstrap_restore, raised_during_volume_creation, live_raises
):
    """DAH-3980: the live power query overlaps the volume create, unless a restore runs first."""
    svc = svc_fixture
    _wire_early_probes(svc, monkeypatch, image_present=True)
    monkeypatch.setattr(svc, "_run_bootstrap_restore", AsyncMock())
    raise_low = AsyncMock(return_value=0)
    monkeypatch.setattr("services.docker_service.raise_low_power_limits_to_default", raise_low)
    raised_at_volume_creation: list[int] = []

    async def slow_create_local_volume(**kwargs):
        await asyncio.sleep(0.01)
        raised_at_volume_creation.append(raise_low.await_count)

    svc.create_local_volume = slow_create_local_volume
    payload = _deploy_payload()
    if bootstrap_restore:
        payload.bootstrap_restore = Mock(restore_log_id="restore-log")

    result = await _run_create_container(svc, payload)

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert raised_at_volume_creation == [raised_during_volume_creation]
    assert raise_low.await_count == live_raises


@pytest.fixture
def svc_fixture():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )
