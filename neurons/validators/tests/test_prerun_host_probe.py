"""DAH-3257 — rental pre-run host probe (RENTAL_PRERUN_HOST_PROBE_ENABLED).

Before `docker run`, a rent read eight host listings over eight serial SSH commands (containers,
volumes, mounted volumes, volume names again, GPU minor map, shared device nodes, nvidia-smi power
state, the image's encryption label). With the flag on, one probe command returns them all; every
consumer reads its section and keeps its removals / writes as they are. Covered here:

- the probe command carries every section and the exact per-command texts (`_PROC_GPU_INFO_CMD`,
  `_POWER_STATE_CMD`, the shared-node loop, the label inspect); `with_power=False` has no nvidia-smi;
- the parser: every section, a failed section → None, a missing `_RC` / an untagged line / an
  unknown tag → raise, the label's whole-stdout semantics;
- the command and the parser together through `sh -c` with a docker stub (containers, volumes,
  mounts, label; nvidia-smi absent → power None);
- `probe_prerun_host` is never fatal (SSH error or garbage → None);
- each consumer with a probe runs no listing command and produces the same result / the same
  removal commands as the per-command path for the same host facts; a failed section falls back;
- `create_container`: flag off → never probes; flag on → one probe handed to every consumer; the
  docker listings are withdrawn after a removal; the power state is withdrawn after a restore.
"""

from __future__ import annotations

import os
import stat
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from services import nvidia_devices as nd
from services.docker_service import DockerService, _ENCRYPTED_VOLUME_IMAGE_LABEL
from services.gpu_power_limit import (
    _POWER_STATE_CMD,
    raise_low_power_limits_to_default,
    restore_tracked_gpu_power_limits,
)
from services.nvidia_devices import (
    _GPU_DEVICE_NODES_CMD,
    _PROC_GPU_INFO_CMD,
    build_gpu_docker_config_for_executor,
    shared_device_nodes_command,
)
from services.prerun_host_probe import (
    DOCKER_MOUNTED_VOLUME_NAMES_CMD,
    DOCKER_PS_ALL_NAMES_CMD,
    DOCKER_VOLUME_LS_NAME_DRIVER_CMD,
    PREFIX_FAILED_MARKER,
    PrerunHostProbe,
    PrerunHostProbeParseError,
    image_label_command,
    parse_prerun_host_probe,
    prerun_host_probe_command,
)
from test_deploy_optimizations import (
    _patch_happy,
    _payload as _deploy_payload,
    _run as _run_create_container,
    _ssh_client as _deploy_ssh_client,
    _ssh_result,
)

_IMAGE = "daturaai/pytorch:1.0.0"


def _probe(**over) -> PrerunHostProbe:
    base = dict(
        container_names=(),
        volumes=(),
        mounted_volume_names=(),
        gpu_proc_stdout="",
        gpu_device_nodes=(),
        shared_nodes=(),
        shared_nodes_whole_host_only=(),
        power_state_stdout="",
        image_label_value="",
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
    gpuproc: tuple[str, ...] = ("GPU-1, 0", "GPU-2, 1"),
    gpuproc_rc: int = 0,
    gpudev: tuple[str, ...] = ("/dev/nvidia0", "/dev/nvidia1"),
    shared: tuple[str, ...] = ("/dev/nvidiactl", "/dev/nvidia-uvm"),
    sharedw: tuple[str, ...] = ("/dev/infiniband/uverbs0", "/dev/nvidia-caps/nvidia-cap1"),
    power: tuple[str, ...] | None = ("GPU-1, 300.00, 450.00, 100.00, 450.00",),
    power_rc: int = 0,
    label: tuple[str, ...] = ("1",),
    label_rc: int = 0,
) -> str:
    out = (
        _tagged("PS", *ps, rc=ps_rc)
        + _tagged("VOL", *vol, rc=vol_rc)
        + _tagged("MNT", *mnt, rc=mnt_rc)
        + _tagged("GPUPROC", *gpuproc, rc=gpuproc_rc)
        + _tagged("GPUDEV", *gpudev)
        + _tagged("SHARED", *shared)
        + _tagged("SHAREDW", *sharedw)
    )
    if power is not None:
        out += _tagged("POWER", *power, rc=power_rc)
    out += _tagged("LABEL", *label, rc=label_rc)
    return out


@pytest.fixture
def docker_service():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


def _ssh(*results):
    client = AsyncMock()
    client.run = AsyncMock(side_effect=list(results))
    return client


def _cmds(client) -> list[str]:
    return [c.args[0] for c in client.run.await_args_list if c.args]


# ------------------------------------------------------------------
# the command
# ------------------------------------------------------------------


def test_probe_command_carries_every_section_and_the_per_command_texts():
    cmd = prerun_host_probe_command(
        docker_image=_IMAGE, image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL, with_power=True
    )
    assert "\n" not in cmd
    for tag in ("PS", "VOL", "MNT", "GPUPROC", "GPUDEV", "SHARED", "SHAREDW", "POWER", "LABEL"):
        assert f"t {tag} " in cmd
    # shlex.quote wraps each section command; the quoted forms of the shared texts are inside
    import shlex

    assert shlex.quote(DOCKER_PS_ALL_NAMES_CMD) in cmd
    assert shlex.quote(DOCKER_VOLUME_LS_NAME_DRIVER_CMD) in cmd
    assert shlex.quote(DOCKER_MOUNTED_VOLUME_NAMES_CMD) in cmd
    assert f"|| echo {PREFIX_FAILED_MARKER}" in cmd

    assert shlex.quote(_PROC_GPU_INFO_CMD) in cmd
    assert shlex.quote(_GPU_DEVICE_NODES_CMD) in cmd
    assert shlex.quote(_POWER_STATE_CMD) in cmd
    assert shlex.quote(shared_device_nodes_command(is_whole_host_rental=False)) in cmd
    assert (
        shlex.quote(shared_device_nodes_command(is_whole_host_rental=True, whole_host_only=True))
        in cmd
    )
    assert shlex.quote(image_label_command(_IMAGE, _ENCRYPTED_VOLUME_IMAGE_LABEL)) in cmd
    # the probe reads; it never removes, installs or writes
    for verb in (" rm ", "volume rm", "plugin install", "nvidia-smi -pl", "-pm 1"):
        assert verb not in cmd


def test_probe_command_without_power_has_no_nvidia_smi():
    cmd = prerun_host_probe_command(
        docker_image=_IMAGE, image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL, with_power=False
    )
    assert "nvidia-smi" not in cmd
    assert "t POWER " not in cmd


def test_image_label_command_is_the_one_the_label_check_ran():
    cmd = image_label_command("a b/c:1", _ENCRYPTED_VOLUME_IMAGE_LABEL)
    assert cmd == (
        "/usr/bin/docker image inspect --format "
        f"'{{{{index .Config.Labels \"{_ENCRYPTED_VOLUME_IMAGE_LABEL}\"}}}}' 'a b/c:1'"
    )


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


def test_parser_reads_every_section():
    probe = parse_prerun_host_probe(_stdout(), with_power=True)
    assert probe.container_names == ("pod_a", "other")
    assert probe.volumes == (("volume_a", "vloopback:latest"), ("dphn_cache_x", "local"))
    assert probe.volume_names == ("volume_a", "dphn_cache_x")
    assert probe.mounted_volume_names == ("volume_a",)
    assert probe.gpu_proc_stdout == "GPU-1, 0\nGPU-2, 1"
    assert probe.gpu_device_nodes == ("/dev/nvidia0", "/dev/nvidia1")
    assert probe.shared_nodes_for(is_whole_host_rental=False) == (
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
    )
    assert probe.shared_nodes_for(is_whole_host_rental=True) == (
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/infiniband/uverbs0",
        "/dev/nvidia-caps/nvidia-cap1",
    )
    assert probe.power_state_stdout == "GPU-1, 300.00, 450.00, 100.00, 450.00"
    assert probe.image_label_value == "1"


def test_parser_empty_sections_are_empty_not_none():
    probe = parse_prerun_host_probe(
        _stdout(ps=(), vol=(), mnt=(), gpudev=(), shared=(), sharedw=()), with_power=True
    )
    assert probe.container_names == ()
    assert probe.volumes == ()
    assert probe.mounted_volume_names == ()
    assert probe.gpu_device_nodes == ()
    assert probe.shared_nodes_for(is_whole_host_rental=True) == ()


@pytest.mark.parametrize(
    "kwargs, attr",
    [
        ({"ps_rc": 1}, "container_names"),
        ({"vol_rc": 125}, "volumes"),
        ({"mnt_rc": 123}, "mounted_volume_names"),
        ({"gpuproc_rc": 2}, "gpu_proc_stdout"),
        ({"power_rc": 127}, "power_state_stdout"),
        ({"label_rc": 1}, "image_label_value"),
    ],
)
def test_parser_failed_section_is_none_and_the_rest_survive(kwargs, attr):
    probe = parse_prerun_host_probe(_stdout(**kwargs), with_power=True)
    assert getattr(probe, attr) is None
    others = {f for f in probe.__dataclass_fields__ if f != attr}
    assert all(getattr(probe, f) is not None for f in others)


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


def test_parser_failed_volume_listing_hides_volume_names_too():
    probe = parse_prerun_host_probe(_stdout(vol_rc=1), with_power=True)
    assert probe.volumes is None and probe.volume_names is None


def test_parser_shared_nodes_for_whole_host_needs_both_sections():
    probe = _probe(shared_nodes=("/dev/nvidiactl",), shared_nodes_whole_host_only=None)
    assert probe.shared_nodes_for(is_whole_host_rental=False) == ("/dev/nvidiactl",)
    assert probe.shared_nodes_for(is_whole_host_rental=True) is None


def test_parser_without_power_ignores_the_power_section_requirement():
    probe = parse_prerun_host_probe(_stdout(power=None), with_power=False)
    assert probe.power_state_stdout is None
    assert probe.container_names == ("pod_a", "other")


def test_parser_missing_rc_line_raises():
    stdout = _stdout().replace("MNT_RC\t0\n", "")
    with pytest.raises(PrerunHostProbeParseError, match="no MNT_RC"):
        parse_prerun_host_probe(stdout, with_power=True)


def test_parser_missing_power_rc_raises_only_when_power_was_asked():
    stdout = _stdout(power=None)
    with pytest.raises(PrerunHostProbeParseError, match="no POWER_RC"):
        parse_prerun_host_probe(stdout, with_power=True)
    parse_prerun_host_probe(stdout, with_power=False)


def test_parser_untagged_line_raises():
    with pytest.raises(PrerunHostProbeParseError, match="untagged line"):
        parse_prerun_host_probe("garbage\n" + _stdout(), with_power=True)


def test_parser_unknown_tag_raises():
    with pytest.raises(PrerunHostProbeParseError, match="unknown section"):
        parse_prerun_host_probe(_stdout() + "NEW\tx\nNEW_RC\t0\n", with_power=True)


def test_parser_non_numeric_rc_raises():
    with pytest.raises(PrerunHostProbeParseError, match="PS_RC is 'x'"):
        parse_prerun_host_probe(_stdout().replace("PS_RC\t0", "PS_RC\tx"), with_power=True)


def test_parser_error_message_is_capped():
    with pytest.raises(PrerunHostProbeParseError) as exc:
        parse_prerun_host_probe(
            _stdout(ps=("x" * 5000,)).replace("MNT_RC\t0\n", ""), with_power=True
        )
    assert len(str(exc.value)) < 800


@pytest.mark.parametrize(
    "lines, expected",
    [
        (("1",), "1"),
        (("",), ""),  # no such label → `{{index …}}` prints an empty line
        (("<no value>",), "<no value>"),
        (
            ("", "1"),
            "1",
        ),  # a value with a leading newline arrives as two LABEL lines; strip() as before
        (("1", "x"), "1\nx"),
        (
            ("1\tx",),
            "1\tx",
        ),  # a tab in the value stays in the value: only the first tab separates the tag
    ],
)
def test_parser_label_value_matches_the_stripped_stdout_semantics(lines, expected):
    probe = parse_prerun_host_probe(_stdout(label=lines), with_power=True)
    assert probe.image_label_value == expected
    assert (probe.image_label_value == "1") is (expected == "1")


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
        docker_image=_IMAGE, image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL, with_power=with_power
    )
    cmd = cmd.replace("/usr/bin/docker", str(stub))
    env = {**os.environ, "LABEL_VALUE": label_value, "PATH": f"{tmp_path}:/usr/bin:/bin"}
    return subprocess.run(
        ["sh", "-c", cmd], capture_output=True, text=True, env=env, check=False
    ).stdout


def test_probe_through_sh_lists_docker_sections_and_marks_absent_nvidia_smi(tmp_path):
    out = _run_probe_in_sh(tmp_path)
    probe = parse_prerun_host_probe(out, with_power=True)
    assert probe.container_names == ("pod_a", "filler_b", "other")
    assert probe.volumes == (("volume_a", "vloopback:latest"), ("volume_b", "local"))
    assert probe.mounted_volume_names == ("volume_a", "volume_a")  # blank lines dropped as before
    assert probe.image_label_value == "1"
    # the GPU / device-node sections list whatever the test host has (a GPU workstation has
    # /dev/nvidia*, a CI runner may have /dev/infiniband/*) — they are listings, never failures
    assert probe.gpu_proc_stdout is not None and probe.gpu_device_nodes is not None
    assert probe.shared_nodes_for(is_whole_host_rental=True) is not None
    # the nvidia-smi stub exits 1 → the section failed → None → the live query would run
    assert probe.power_state_stdout is None


def test_probe_through_sh_label_absent_is_not_encrypted(tmp_path):
    probe = parse_prerun_host_probe(_run_probe_in_sh(tmp_path, label_value=""), with_power=True)
    assert probe.image_label_value == ""


def test_probe_through_sh_multiline_label_cannot_forge_another_section(tmp_path):
    # every line of a command's output gets that command's tag — an image label cannot inject a PS line
    probe = parse_prerun_host_probe(
        _run_probe_in_sh(tmp_path, label_value="1\nPS\tpod_evil"), with_power=True
    )
    assert probe.container_names == ("pod_a", "filler_b", "other")
    assert probe.image_label_value == "1\nPS\tpod_evil"
    assert probe.image_label_value != "1"


def test_probe_through_sh_prefix_failure_is_a_whole_probe_fallback(tmp_path):
    # awk missing on the host → the marker line (untagged) → the parser raises → probe None →
    # every step on its own command; never an empty listing
    stub = tmp_path / "docker"
    stub.write_text(_DOCKER_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    broken_awk = tmp_path / "awk"
    broken_awk.write_text("#!/bin/sh\nexit 1\n")
    broken_awk.chmod(broken_awk.stat().st_mode | stat.S_IXUSR)
    cmd = prerun_host_probe_command(
        docker_image=_IMAGE, image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL, with_power=False
    ).replace("/usr/bin/docker", str(stub))
    env = {**os.environ, "LABEL_VALUE": "1", "PATH": f"{tmp_path}:/usr/bin:/bin"}
    out = subprocess.run(
        ["sh", "-c", cmd], capture_output=True, text=True, env=env, check=False
    ).stdout
    assert PREFIX_FAILED_MARKER in out
    with pytest.raises(PrerunHostProbeParseError, match="untagged line"):
        parse_prerun_host_probe(out, with_power=False)


# ------------------------------------------------------------------
# probe_prerun_host is never fatal
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_prerun_host_runs_one_command_and_parses(docker_service):
    ssh = _ssh(_ssh_result(stdout=_stdout()))
    probe = await docker_service.probe_prerun_host(
        ssh, docker_image=_IMAGE, with_power=True, log_extra={}
    )
    assert probe is not None and probe.container_names == ("pod_a", "other")
    assert _cmds(ssh) == [
        prerun_host_probe_command(
            docker_image=_IMAGE, image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL, with_power=True
        )
    ]


@pytest.mark.asyncio
async def test_probe_prerun_host_ssh_error_is_none(docker_service):
    ssh = _ssh(ConnectionError("channel closed"))
    assert (
        await docker_service.probe_prerun_host(
            ssh, docker_image=_IMAGE, with_power=True, log_extra={}
        )
        is None
    )


@pytest.mark.asyncio
async def test_probe_prerun_host_is_bounded_and_a_timeout_is_none(docker_service):
    import asyncio

    from services.docker_service import _PRERUN_HOST_PROBE_TIMEOUT_SECONDS

    ssh = _ssh(asyncio.TimeoutError())
    assert (
        await docker_service.probe_prerun_host(
            ssh, docker_image=_IMAGE, with_power=True, log_extra={}
        )
        is None
    )
    assert ssh.run.await_args.kwargs["timeout"] == _PRERUN_HOST_PROBE_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_probe_prerun_host_garbage_is_none(docker_service):
    ssh = _ssh(_ssh_result(stdout="sh: t: not found\n"))
    assert (
        await docker_service.probe_prerun_host(
            ssh, docker_image=_IMAGE, with_power=True, log_extra={}
        )
        is None
    )


# ------------------------------------------------------------------
# consumers: same result, no listing command
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_existing_containers_with_probe_removes_the_same_and_lists_nothing(
    docker_service, monkeypatch
):
    ran: list[str] = []

    async def _retry(ssh, command, _tag):
        ran.append(command)

    monkeypatch.setattr("services.docker_service.retry_ssh_command", _retry)
    live = _ssh(_ssh_result(stdout="pod_new\npod_old\nfiller_x\nsomething\n"))
    removed_live = await docker_service.clean_existing_containers(
        ssh_client=live,
        default_extra={},
        pod_name="pod_new",
        active_container_names=["pod_keep"],
        active_volume_names=["volume_old"],
    )
    live_cmds = list(ran)
    ran.clear()
    probe = _probe(container_names=("pod_new", "pod_old", "filler_x", "something"))
    probed = _ssh()
    removed_probed = await docker_service.clean_existing_containers(
        ssh_client=probed,
        default_extra={},
        pod_name="pod_new",
        active_container_names=["pod_keep"],
        active_volume_names=["volume_old"],
        host_probe=probe,
    )
    assert removed_live == removed_probed == ["pod_new", "pod_old", "filler_x"]
    assert (
        ran
        == live_cmds
        == [
            "/usr/bin/docker rm -fv pod_new pod_old filler_x",
            "/usr/bin/docker volume rm volume_new volume_x 2>/dev/null || true",
        ]
    )
    assert _cmds(live) == ['/usr/bin/docker ps -a --format "{{.Names}}"']
    assert _cmds(probed) == []


@pytest.mark.asyncio
async def test_clean_existing_containers_nothing_stale_returns_empty(docker_service):
    probe = _probe(container_names=("other", "pod_keep"))
    ssh = _ssh()
    assert (
        await docker_service.clean_existing_containers(
            ssh_client=ssh,
            default_extra={},
            pod_name="pod_new",
            active_container_names=["pod_keep"],
            host_probe=probe,
        )
        == []
    )
    assert _cmds(ssh) == []


@pytest.mark.asyncio
async def test_clean_existing_containers_failed_ps_section_lists_itself(docker_service):
    probe = _probe(container_names=None)
    ssh = _ssh(_ssh_result(stdout=""))
    assert (
        await docker_service.clean_existing_containers(
            ssh_client=ssh,
            default_extra={},
            pod_name="pod_new",
            host_probe=probe,
        )
        == []
    )
    assert _cmds(ssh) == ['/usr/bin/docker ps -a --format "{{.Names}}"']


@pytest.mark.asyncio
async def test_clean_stale_vloopback_with_probe_removes_the_same_and_lists_nothing(
    docker_service, monkeypatch
):
    ran: list[str] = []

    async def _retry(ssh, command, _tag):
        ran.append(command)

    monkeypatch.setattr("services.docker_service.retry_ssh_command", _retry)
    volumes = "volume_a vloopback:latest\nvolume_b vloopback\nvolume_c local\nvolume_d vloopback:latest\nother vloopback\n"
    live = _ssh(_ssh_result(stdout=volumes), _ssh_result(stdout="volume_a\n\nvolume_c\n"))
    removed_live = await docker_service.clean_stale_vloopback_volumes(
        ssh_client=live,
        default_extra={},
        skip_volume_names={"volume_d"},
    )
    live_cmds = list(ran)
    ran.clear()
    probe = _probe(
        volumes=(
            ("volume_a", "vloopback:latest"),
            ("volume_b", "vloopback"),
            ("volume_c", "local"),
            ("volume_d", "vloopback:latest"),
            ("other", "vloopback"),
        ),
        mounted_volume_names=("volume_a", "volume_c"),
    )
    probed = _ssh()
    removed_probed = await docker_service.clean_stale_vloopback_volumes(
        ssh_client=probed,
        default_extra={},
        skip_volume_names={"volume_d"},
        host_probe=probe,
    )
    assert removed_live == removed_probed == ["volume_b"]
    assert ran == live_cmds == ["/usr/bin/docker volume rm volume_b 2>/dev/null || true"]
    assert len(_cmds(live)) == 2 and _cmds(probed) == []


@pytest.mark.asyncio
async def test_clean_stale_vloopback_probe_without_vloopback_volumes_runs_nothing(docker_service):
    ssh = _ssh()
    assert (
        await docker_service.clean_stale_vloopback_volumes(
            ssh_client=ssh,
            default_extra={},
            host_probe=_probe(volumes=(("volume_c", "local"),)),
        )
        == []
    )
    assert _cmds(ssh) == []


@pytest.mark.asyncio
async def test_clean_stale_vloopback_failed_mount_section_inspects_itself(
    docker_service, monkeypatch
):
    monkeypatch.setattr("services.docker_service.retry_ssh_command", AsyncMock())
    probe = _probe(volumes=(("volume_a", "vloopback"),), mounted_volume_names=None)
    ssh = _ssh(_ssh_result(stdout="volume_a\n"))
    assert (
        await docker_service.clean_stale_vloopback_volumes(
            ssh_client=ssh, default_extra={}, host_probe=probe
        )
        == []
    )
    assert len(_cmds(ssh)) == 1 and "docker inspect" in _cmds(ssh)[0]


@pytest.mark.asyncio
async def test_find_cache_volumes_with_probe_matches_live(docker_service):
    live = _ssh(_ssh_result(stdout="dphn_cache_old\ndphn_cache_new\nvolume_a\n"))
    from_live = await docker_service._find_cache_volumes_to_sweep(live, {"dphn_cache_new"}, {})
    probe = _probe(
        volumes=(
            ("dphn_cache_old", "local"),
            ("dphn_cache_new", "local"),
            ("volume_a", "vloopback"),
        )
    )
    probed = _ssh()
    from_probe = await docker_service._find_cache_volumes_to_sweep(
        probed, {"dphn_cache_new"}, {}, host_probe=probe
    )
    assert from_live == from_probe == ["dphn_cache_old"]
    assert _cmds(probed) == []


@pytest.mark.asyncio
async def test_find_cache_volumes_failed_volume_section_lists_itself(docker_service):
    ssh = _ssh(_ssh_result(stdout="dphn_cache_old\n"))
    assert await docker_service._find_cache_volumes_to_sweep(
        ssh, set(), {}, host_probe=_probe(volumes=None)
    ) == ["dphn_cache_old"]
    assert _cmds(ssh) == ['/usr/bin/docker volume ls --format "{{.Name}}"']


@pytest.mark.asyncio
async def test_image_label_with_probe_runs_nothing_and_matches(docker_service):
    ssh = _ssh()
    assert await docker_service._image_has_encrypted_volume_label(
        ssh, _IMAGE, host_probe=_probe(image_label_value="1")
    )
    assert not await docker_service._image_has_encrypted_volume_label(
        ssh, _IMAGE, host_probe=_probe(image_label_value="")
    )
    assert not await docker_service._image_has_encrypted_volume_label(
        ssh, _IMAGE, host_probe=_probe(image_label_value="<no value>")
    )
    assert _cmds(ssh) == []


@pytest.mark.asyncio
async def test_image_label_failed_section_inspects_itself(docker_service):
    ssh = _ssh(_ssh_result(stdout="1\n"))
    assert await docker_service._image_has_encrypted_volume_label(
        ssh, _IMAGE, host_probe=_probe(image_label_value=None)
    )
    assert _cmds(ssh) == [image_label_command(_IMAGE, _ENCRYPTED_VOLUME_IMAGE_LABEL)]


@pytest.mark.asyncio
async def test_gpu_config_with_probe_matches_live_partial_rental(monkeypatch):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False)
    proc = "GPU-1, 0\nGPU-2, 1\n"
    live = _ssh(_ssh_result(stdout=proc), _ssh_result(stdout="/dev/nvidiactl\n/dev/nvidia-uvm\n"))
    from_live = await build_gpu_docker_config_for_executor(live, ["GPU-2"])
    probe = _probe(
        gpu_proc_stdout=proc,
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
        gpu_proc_stdout="GPU-1, 0\nGPU-2, 1\n",
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
    probe = _probe(gpu_proc_stdout="GPU-1, 0\n", shared_nodes=("/dev/nvidiactl",))
    ssh = _ssh(_ssh_result(exit_status=1, stderr="no nvidia-smi"))
    config = await build_gpu_docker_config_for_executor(ssh, ["GPU-9"], host_probe=probe)
    # the kernel map lacks GPU-9 → nvidia-smi XML is asked live → fails → legacy --gpus-only fallback
    assert _cmds(ssh) == ["nvidia-smi -q -x"]
    assert config.device_mounts == ()


@pytest.mark.asyncio
async def test_gpu_config_failed_proc_section_queries_live(monkeypatch):
    monkeypatch.setattr(nd.settings, "KERNEL_GPU_VERDICT_CHECK_ENABLED", False)
    probe = _probe(gpu_proc_stdout=None, shared_nodes=("/dev/nvidiactl",))
    ssh = _ssh(_ssh_result(stdout="GPU-1, 0\n"))
    config = await build_gpu_docker_config_for_executor(ssh, ["GPU-1"], host_probe=probe)
    assert _cmds(ssh) == [_PROC_GPU_INFO_CMD]
    assert [d.path_on_host for d in config.device_mounts] == ["/dev/nvidia0", "/dev/nvidiactl"]


@pytest.mark.asyncio
async def test_raise_low_power_limits_with_probe_queries_nothing_and_still_raises():
    state = "GPU-1, 100.00, 450.00, 100.00, 450.00\nGPU-2, 450.00, 450.00, 100.00, 450.00\n"
    probe = _probe(power_state_stdout=state)
    # -pm 1, -pl 450, readback
    ssh = _ssh(
        _ssh_result(stdout=""),
        _ssh_result(stdout=""),
        _ssh_result(stdout="450, Enabled\n"),
    )
    raised = await raise_low_power_limits_to_default(
        ssh, "exec-1", ["GPU-1", "GPU-2"], host_probe=probe
    )
    assert raised == 1
    assert _POWER_STATE_CMD not in _cmds(ssh)
    assert any("-pl 450" in c and "GPU-1" in c for c in _cmds(ssh))


@pytest.mark.asyncio
async def test_raise_low_power_limits_failed_power_section_queries_live():
    ssh = _ssh(_ssh_result(stdout="GPU-1, 450.00, 450.00, 100.00, 450.00\n"))
    assert (
        await raise_low_power_limits_to_default(
            ssh, "exec-1", ["GPU-1"], host_probe=_probe(power_state_stdout=None)
        )
        == 0
    )
    assert _cmds(ssh) == [_POWER_STATE_CMD]


@pytest.mark.asyncio
async def test_restore_tracked_limits_with_probe_uses_it_for_before_values():
    redis = AsyncMock()
    redis.get = AsyncMock(
        return_value='{"gpu_uuid":"GPU-1","watts":400,"pod_id":"p","executor_id":"e","capped_at":1.0}'
    )
    redis.delete = AsyncMock()
    probe = _probe(power_state_stdout="GPU-1, 250.00, 450.00, 100.00, 450.00\n")
    ssh = _ssh(_ssh_result(stdout=""), _ssh_result(stdout=""), _ssh_result(stdout="400, Enabled\n"))
    assert await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-1"], host_probe=probe) == 1
    assert _POWER_STATE_CMD not in _cmds(ssh)


@pytest.mark.asyncio
async def test_restore_counts_a_written_limit_even_when_its_record_cannot_be_cleared():
    # create_container withdraws the probe's power state from the raise when restore_* returns > 0;
    # a limit that was written but whose Redis record could not be deleted still changed the GPU
    redis = AsyncMock()
    redis.get = AsyncMock(
        return_value='{"gpu_uuid":"GPU-1","watts":400,"pod_id":"p","executor_id":"e","capped_at":1.0}'
    )
    redis.delete = AsyncMock(side_effect=ConnectionError("redis gone"))
    ssh = _ssh(
        _ssh_result(stdout="GPU-1, 250.00, 450.00, 100.00, 450.00\n"),
        _ssh_result(stdout=""),
        _ssh_result(stdout=""),
        _ssh_result(stdout="400, Enabled\n"),
    )
    assert await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-1"]) == 1


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
async def test_create_container_flag_off_never_probes(svc_fixture, monkeypatch):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", False)
    ssh_client = _deploy_ssh_client()
    _wire(svc, monkeypatch, ssh_client, probe_result=_probe())
    payload = _deploy_payload(enable_volume_encryption=True, is_sysbox=True)
    result = await _run_create_container(svc, payload)
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    svc.probe_prerun_host.assert_not_awaited()
    for m in (
        svc.clean_existing_containers,
        svc.clean_stale_vloopback_volumes,
        svc.sweep_stale_cache_volumes,
        svc.select_affordable_cache_volumes,
        svc.reclaim_dphn_cache_for_rental,
    ):
        assert _probe_kwarg(m) is None
    from services import docker_service as ds

    assert _probe_kwarg(ds.build_gpu_docker_config_for_executor) is None
    assert _probe_kwarg(ds.raise_low_power_limits_to_default) is None


@pytest.mark.asyncio
async def test_create_container_flag_on_probes_once_and_hands_it_to_every_consumer(
    svc_fixture, monkeypatch
):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    probe = _probe()
    _wire(svc, monkeypatch, ssh_client, probe_result=probe)
    payload = _deploy_payload(enable_volume_encryption=True, is_sysbox=True)
    result = await _run_create_container(svc, payload)
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    svc.probe_prerun_host.assert_awaited_once()
    assert svc.probe_prerun_host.await_args.kwargs["docker_image"] == payload.docker_image
    assert svc.probe_prerun_host.await_args.kwargs["with_power"] is True
    for m in (
        svc.clean_existing_containers,
        svc.clean_stale_vloopback_volumes,
        svc.sweep_stale_cache_volumes,
        svc.select_affordable_cache_volumes,
        svc.reclaim_dphn_cache_for_rental,
        svc._image_has_encrypted_volume_label,
    ):
        assert _probe_kwarg(m) is probe, m
    from services import docker_service as ds

    assert _probe_kwarg(ds.build_gpu_docker_config_for_executor) is probe
    assert _probe_kwarg(ds.restore_tracked_gpu_power_limits) is probe
    assert _probe_kwarg(ds.raise_low_power_limits_to_default) is probe


@pytest.mark.asyncio
async def test_create_container_withdraws_docker_listings_after_a_removal(svc_fixture, monkeypatch):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    probe = _probe()
    _wire(svc, monkeypatch, ssh_client, probe_result=probe)
    svc.clean_existing_containers = AsyncMock(return_value=["pod_old"])
    result = await _run_create_container(svc, _deploy_payload())
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _probe_kwarg(svc.clean_existing_containers) is probe
    for m in (
        svc.clean_stale_vloopback_volumes,
        svc.sweep_stale_cache_volumes,
        svc.select_affordable_cache_volumes,
        svc.reclaim_dphn_cache_for_rental,
    ):
        assert _probe_kwarg(m) is None, m
    from services import docker_service as ds

    # a docker removal does not touch the GPU / power sections
    assert _probe_kwarg(ds.build_gpu_docker_config_for_executor) is probe
    assert _probe_kwarg(ds.raise_low_power_limits_to_default) is probe


@pytest.mark.asyncio
async def test_create_container_withdraws_docker_listings_after_a_vloopback_sweep(
    svc_fixture, monkeypatch
):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    probe = _probe()
    _wire(svc, monkeypatch, ssh_client, probe_result=probe)
    svc.clean_stale_vloopback_volumes = AsyncMock(return_value=["volume_b"])
    result = await _run_create_container(svc, _deploy_payload())
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _probe_kwarg(svc.clean_existing_containers) is probe
    assert _probe_kwarg(svc.clean_stale_vloopback_volumes) is probe
    assert _probe_kwarg(svc.sweep_stale_cache_volumes) is None
    assert _probe_kwarg(svc.reclaim_dphn_cache_for_rental) is None


@pytest.mark.asyncio
async def test_create_container_withdraws_power_state_after_a_restore(svc_fixture, monkeypatch):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    probe = _probe()
    _wire(svc, monkeypatch, ssh_client, probe_result=probe)
    from services import docker_service as ds

    monkeypatch.setattr(
        "services.docker_service.restore_tracked_gpu_power_limits", AsyncMock(return_value=1)
    )
    result = await _run_create_container(svc, _deploy_payload())
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _probe_kwarg(ds.restore_tracked_gpu_power_limits) is probe
    assert _probe_kwarg(ds.raise_low_power_limits_to_default) is None


@pytest.mark.asyncio
async def test_create_container_probe_failure_leaves_every_consumer_on_its_own_commands(
    svc_fixture, monkeypatch
):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    _wire(svc, monkeypatch, ssh_client, probe_result=None)
    result = await _run_create_container(
        svc, _deploy_payload(enable_volume_encryption=True, is_sysbox=True)
    )
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    svc.probe_prerun_host.assert_awaited_once()
    from services import docker_service as ds

    for m in (
        svc.clean_existing_containers,
        svc.clean_stale_vloopback_volumes,
        svc._image_has_encrypted_volume_label,
        ds.build_gpu_docker_config_for_executor,
        ds.raise_low_power_limits_to_default,
    ):
        assert _probe_kwarg(m) is None


@pytest.mark.asyncio
async def test_create_container_pearl_filler_does_not_ask_for_power(svc_fixture, monkeypatch):
    from payload_models.payloads import GpuPowerLimit, WorkloadKind

    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    _wire(svc, monkeypatch, ssh_client, probe_result=_probe())
    monkeypatch.setattr(
        "services.docker_service.apply_filler_gpu_power_limits", AsyncMock(return_value=True)
    )
    payload = _deploy_payload(
        workload_kind=WorkloadKind.FILLER,
        gpu_power_limits=[GpuPowerLimit(gpu_uuid="GPU-test", watts=300)],
    )
    result = await _run_create_container(svc, payload)
    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert svc.probe_prerun_host.await_args.kwargs["with_power"] is False


@pytest.fixture
def svc_fixture():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )
