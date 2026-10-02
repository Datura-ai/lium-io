"""DAH-3257 — rental pre-run host probe (RENTAL_PRERUN_HOST_PROBE_ENABLED).

Before `docker run`, a rent read eight host listings over eight serial SSH commands (containers,
volumes, mounted volumes, volume names again, GPU minor map, shared device nodes, nvidia-smi power
state, the image's encryption label). With the flag on, one probe command returns them all; every
consumer reads its section and keeps its removals / writes as they are. Covered here:

- the probe command carries every section and the exact per-command texts (`PROC_GPU_INFO_CMD`,
  `POWER_STATE_CMD`, the shared-node loop, the label inspect); `with_power=False` has no nvidia-smi;
- the parser: every section, a failed section → None, a missing `_RC` / an untagged line / an
  unknown tag → raise, the label's whole-stdout semantics;
- the command and the parser together through `sh -c` with a docker stub (containers, volumes,
  mounts, label; nvidia-smi absent → power None);
- `probe_prerun_host` is never fatal (SSH error or garbage → None);
- each consumer with a probe runs no listing command and produces the same result / the same
  removal commands as the per-command path for the same host facts; a failed section falls back;
- `create_container`: flag off → never probes; flag on → one probe handed to every consumer; the
  docker listings are withdrawn after a removal; the last-resort power raise never reads the probe;
- a customer create that removed only a filler, cleanly, keeps every probe and relists
  nothing; any other removal, a survivor or a failed rm relists as before;
- that removal and its confirmation are the create's one host command; a removal that
  times out fails the create at the cleanup step;
- a customer create removes its fillers as soon as the SSH session is up, beside the image
  inspect and the probes; the cleanup step awaits that removal, never removes those fillers again,
  and everything after it (power restore, cache reclaim, docker run) still waits for it.
"""

from __future__ import annotations

import inspect
import os
import stat
import subprocess
import asyncio
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest
from core.config import settings
from services import nvidia_devices as nd
from services.docker_service import (
    DockerService,
    _ENCRYPTED_VOLUME_IMAGE_LABEL,
    VolumeHostProbe,
    _remove_and_list_containers_command,
)
import services.docker_service as ds_module
from services.gpu_power_limit import (
    POWER_LIMIT_SET_CONCURRENCY,
    POWER_STATE_CMD,
    raise_low_power_limits_to_default,
    restore_tracked_gpu_power_limits,
)
from services.nvidia_devices import (
    GPU_DEVICE_NODES_CMD,
    PROC_GPU_INFO_CMD,
    build_gpu_docker_config_for_executor,
    shared_device_nodes_command,
)
from services.prerun_host_probe import (
    DOCKER_MOUNTED_VOLUME_NAMES_CMD,
    DOCKER_PS_ALL_NAMES_IDS_CMD,
    DOCKER_VOLUME_LS_NAME_DRIVER_CMD,
    PREFIX_FAILED_MARKER,
    PrerunHostProbe,
    PrerunHostProbeParseError,
    ProbedVolume,
    image_label_command,
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
        docker_image=_IMAGE,
        image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL,
        miner_hotkey=_HOTKEY,
        with_power=True
    )
    assert "\n" not in cmd
    for tag in (
        "PS", "VOL", "MNT", "GPUMINORMAP", "GPUDEV", "SHARED", "SHAREDW", "POWER", "LABEL", "PORTCHECK"
    ):
        assert f"t {tag} " in cmd
    # shlex.quote wraps each section command; the quoted forms of the shared texts are inside
    import shlex

    assert shlex.quote(DOCKER_PS_ALL_NAMES_IDS_CMD) in cmd
    assert shlex.quote(DOCKER_VOLUME_LS_NAME_DRIVER_CMD) in cmd
    assert shlex.quote(DOCKER_MOUNTED_VOLUME_NAMES_CMD) in cmd
    assert f"|| echo {PREFIX_FAILED_MARKER}" in cmd

    assert shlex.quote(PROC_GPU_INFO_CMD) in cmd
    assert shlex.quote(GPU_DEVICE_NODES_CMD) in cmd
    assert shlex.quote(POWER_STATE_CMD) in cmd
    assert shlex.quote(shared_device_nodes_command(is_whole_host_rental=False)) in cmd
    assert (
        shlex.quote(shared_device_nodes_command(is_whole_host_rental=True, whole_host_only=True))
        in cmd
    )
    assert shlex.quote(image_label_command(_IMAGE, _ENCRYPTED_VOLUME_IMAGE_LABEL)) in cmd
    assert shlex.quote(port_check_containers_command(_HOTKEY)) in cmd
    # the probe reads; it never removes, installs or writes
    for verb in (" rm ", "volume rm", "plugin install", "nvidia-smi -pl", "-pm 1"):
        assert verb not in cmd


def test_probe_command_without_power_has_no_nvidia_smi():
    cmd = prerun_host_probe_command(
        docker_image=_IMAGE,
        image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL,
        miner_hotkey=_HOTKEY,
        with_power=False
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
    assert probe.volumes == (
        ProbedVolume("volume_a", "vloopback:latest"),
        ProbedVolume("dphn_cache_x", "local"),
    )
    assert probe.volume_names == ("volume_a", "dphn_cache_x")
    assert probe.mounted_volume_names == ("volume_a",)
    assert probe.gpu_minor_map_stdout == "GPU-1, 0\nGPU-2, 1"
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
        ({"gpu_minor_map_rc": 2}, "gpu_minor_map_stdout"),
        ({"power_rc": 127}, "power_state_stdout"),
        ({"label_rc": 1}, "image_label_value"),
        ({"port_check_rc": 1}, "port_check_container_names"),
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


def test_probe_through_sh_lists_docker_sections_and_marks_absent_nvidia_smi(tmp_path):
    out = _run_probe_in_sh(tmp_path)
    probe = parse_prerun_host_probe(out, with_power=True)
    assert probe.container_names == ("pod_a", "filler_b", "other")
    assert probe.volumes == (
        ProbedVolume("volume_a", "vloopback:latest"),
        ProbedVolume("volume_b", "local"),
    )
    assert probe.mounted_volume_names == ("volume_a", "volume_a")  # blank lines dropped as before
    assert probe.image_label_value == "1"
    assert probe.port_check_container_names == ("health_check_1",)
    # the GPU / device-node sections list whatever the test host has (a GPU workstation has
    # /dev/nvidia*, a CI runner may have /dev/infiniband/*) — they are listings, never failures
    assert probe.gpu_minor_map_stdout is not None and probe.gpu_device_nodes is not None
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
        docker_image=_IMAGE,
        image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL,
        miner_hotkey=_HOTKEY,
        with_power=False
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
        ssh, docker_image=_IMAGE, with_power=True, miner_hotkey=_HOTKEY, log_extra={}
    )
    assert probe is not None and probe.container_names == ("pod_a", "other")
    assert _cmds(ssh) == [
        prerun_host_probe_command(
            docker_image=_IMAGE,
        image_label=_ENCRYPTED_VOLUME_IMAGE_LABEL,
        miner_hotkey=_HOTKEY,
        with_power=True
        )
    ]


@pytest.mark.asyncio
async def test_probe_prerun_host_ssh_error_is_none(docker_service):
    ssh = _ssh(ConnectionError("channel closed"))
    assert (
        await docker_service.probe_prerun_host(
            ssh, docker_image=_IMAGE, with_power=True, miner_hotkey=_HOTKEY, log_extra={}
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
            ssh, docker_image=_IMAGE, with_power=True, miner_hotkey=_HOTKEY, log_extra={}
        )
        is None
    )
    assert ssh.run.await_args.kwargs["timeout"] == _PRERUN_HOST_PROBE_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_probe_prerun_host_garbage_is_none(docker_service):
    ssh = _ssh(_ssh_result(stdout="sh: t: not found\n"))
    assert (
        await docker_service.probe_prerun_host(
            ssh, docker_image=_IMAGE, with_power=True, miner_hotkey=_HOTKEY, log_extra={}
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
    assert _cmds(live) == [DOCKER_PS_ALL_NAMES_IDS_CMD]
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
    assert _cmds(ssh) == [DOCKER_PS_ALL_NAMES_IDS_CMD]


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
            ProbedVolume("volume_a", "vloopback:latest"),
            ProbedVolume("volume_b", "vloopback"),
            ProbedVolume("volume_c", "local"),
            ProbedVolume("volume_d", "vloopback:latest"),
            ProbedVolume("other", "vloopback"),
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
            host_probe=_probe(volumes=(ProbedVolume("volume_c", "local"),)),
        )
        == []
    )
    assert _cmds(ssh) == []


@pytest.mark.asyncio
async def test_clean_stale_vloopback_failed_mount_section_inspects_itself(
    docker_service, monkeypatch
):
    monkeypatch.setattr("services.docker_service.retry_ssh_command", AsyncMock())
    probe = _probe(volumes=(ProbedVolume("volume_a", "vloopback"),), mounted_volume_names=None)
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
            ProbedVolume("dphn_cache_old", "local"),
            ProbedVolume("dphn_cache_new", "local"),
            ProbedVolume("volume_a", "vloopback"),
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


@pytest.mark.asyncio
async def test_raise_low_power_limits_takes_no_probe_and_queries_live():
    # The last-resort raise runs minutes after the probe (volume creation, bootstrap restore), so
    # it has no probe parameter at all: the state it acts on is always a live query.
    assert "host_probe" not in inspect.signature(raise_low_power_limits_to_default).parameters
    ssh = _ssh(_ssh_result(stdout="GPU-1, 450.00, 450.00, 100.00, 450.00\n"))
    assert await raise_low_power_limits_to_default(ssh, "exec-1", ["GPU-1"]) == 0
    assert _cmds(ssh) == [POWER_STATE_CMD]


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
    assert POWER_STATE_CMD not in _cmds(ssh)


@pytest.mark.asyncio
async def test_port_check_with_probe_nothing_lingering_runs_nothing(docker_service):
    ssh = _ssh()
    removal = await docker_service.wait_for_port_check_containers(
        executor_info=Mock(), miner_hotkey=_HOTKEY, keypair=Mock(), private_key="",
        ssh_client=ssh, probed_container_names=(),
    )
    assert removal == (False, "No port check containers found")
    assert _cmds(ssh) == []


@pytest.mark.asyncio
async def test_port_check_with_probe_lingering_removes_the_same_as_the_live_listing(docker_service):
    live_ssh = _ssh(_ssh_result(stdout="health_check_1\n"), _ssh_result(stdout="3f2a9c1d7e4b\n"))
    live = await docker_service.wait_for_port_check_containers(
        executor_info=Mock(), miner_hotkey=_HOTKEY, keypair=Mock(), private_key="",
        ssh_client=live_ssh,
    )
    probed_ssh = _ssh(_ssh_result(stdout="3f2a9c1d7e4b\n"))
    probed = await docker_service.wait_for_port_check_containers(
        executor_info=Mock(), miner_hotkey=_HOTKEY, keypair=Mock(), private_key="",
        ssh_client=probed_ssh, probed_container_names=("health_check_1",),
    )
    assert live == probed == (True, "Port check containers forcefully removed")
    assert _cmds(live_ssh) == [port_check_containers_command(_HOTKEY), _cmds(probed_ssh)[0]]
    assert _cmds(probed_ssh)[0].endswith("| xargs -r /usr/bin/docker rm -fv")


@pytest.mark.asyncio
async def test_a_probed_port_check_gone_by_now_is_not_reported_removed(docker_service):
    # the probe at SSH connect listed health_check_1; it exited before the pre-run wait, so xargs -r ran no rm
    ssh = _ssh(_ssh_result(stdout=""))

    removal = await docker_service.wait_for_port_check_containers(
        executor_info=Mock(), miner_hotkey=_HOTKEY, keypair=Mock(), private_key="",
        ssh_client=ssh, probed_container_names=("health_check_1",),
    )

    assert removal == (False, "Port check containers listed but none removed")


@pytest.mark.asyncio
async def test_a_failed_docker_rm_is_not_reported_removed(docker_service):
    # the live listing names a port check; its removal fails on the host
    ssh = _ssh(
        _ssh_result(stdout="health_check_1\n"),
        _ssh_result(exit_status=1, stderr="Error response from daemon: removal already in progress"),
    )

    removal = await docker_service.wait_for_port_check_containers(
        executor_info=Mock(), miner_hotkey=_HOTKEY, keypair=Mock(), private_key="", ssh_client=ssh,
    )

    assert removal.removed is False


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
    probe = _probe(port_check_container_names=("health_check_1",))
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
    port_check = svc.wait_for_port_check_containers.await_args.kwargs
    assert port_check["probed_container_names"] is probe.port_check_container_names
    # the last-resort raise runs minutes after the probe and never takes it
    assert "host_probe" not in ds.raise_low_power_limits_to_default.await_args.kwargs


@pytest.mark.asyncio
async def test_the_early_gpu_power_restore_leaves_two_sessions_for_the_volume_steps(svc_fixture, monkeypatch):
    svc = svc_fixture
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    _wire(svc, monkeypatch, _deploy_ssh_client(), probe_result=_probe())

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    from services import docker_service as ds

    for power_step in (ds.restore_tracked_gpu_power_limits, ds.raise_low_power_limits_to_default):
        assert power_step.await_args_list[0].kwargs["concurrency"] == POWER_LIMIT_SET_CONCURRENCY - 2, power_step


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
    port_check = svc.wait_for_port_check_containers.await_args.kwargs
    assert port_check["probed_container_names"] is None
    from services import docker_service as ds

    # a docker removal does not touch the GPU / power sections
    assert _probe_kwarg(ds.build_gpu_docker_config_for_executor) is probe
    assert _probe_kwarg(ds.restore_tracked_gpu_power_limits) is probe


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


@pytest.mark.asyncio
async def test_cached_create_runs_each_host_probe_once_beside_the_image_inspect(svc_fixture, monkeypatch):
    """DAH-3980: on the cached path the probes cost no round trip of their own."""
    svc = svc_fixture
    host_probes_started_at_inspect = _wire_early_probes(svc, monkeypatch, image_present=True)

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert host_probes_started_at_inspect == [1]
    svc.probe_prerun_host.assert_awaited_once()
    svc.probe_volume_host.assert_awaited_once()


@pytest.mark.asyncio
async def test_pulled_image_probes_the_host_again_after_the_pull(svc_fixture, monkeypatch):
    """The early label section read no image, and a pull can take minutes: both probes run again."""
    svc = svc_fixture
    _wire_early_probes(svc, monkeypatch, image_present=False)

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _docker_client(svc).pulled_images == [_IMAGE]
    assert svc.probe_prerun_host.await_count == 2
    assert svc.probe_volume_host.await_count == 2


@pytest.mark.asyncio
async def test_cleanup_that_removed_a_volume_measures_the_volume_facts_again(svc_fixture, monkeypatch):
    """df and the volume list read before the removal would size the new volume on stale facts."""
    svc = svc_fixture
    _wire_early_probes(svc, monkeypatch, image_present=True)
    svc.clean_stale_vloopback_volumes = AsyncMock(return_value=["volume_old"])

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    svc.probe_prerun_host.assert_awaited_once()
    assert svc.probe_volume_host.await_count == 2


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
) -> AsyncMock:
    """Both early probes on; the cleanup, the sweeps and the port-check wait are the real
    ones over a stub SSH client, so every listing they run is a command on it. The host lists the
    probe's containers until the first rm, then ``ps_after_rm``."""
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    monkeypatch.setattr(settings, "RENTAL_VOLUME_FAST_PATH_ENABLED", True)
    ssh_client = _deploy_ssh_client()
    names_on_host = list(probe.container_names or () if containers_on_host is None else containers_on_host)

    async def answer(cmd, *args, **kwargs):
        nonlocal names_on_host
        if "/usr/bin/docker rm -fv" in cmd:
            await asyncio.sleep(docker_rm_seconds)
            if docker_rm_raises is not None:
                raise docker_rm_raises
            names_on_host = ps_after_rm.split()
            # the removal command reports the rm's status and the names left after it
            names_after = "".join(f"NAME\t{name}\n" for name in names_on_host)
            return _ssh_result(
                stdout=f"RM\t{docker_rm_exit}\n{names_after}PS\t{listing_after_rm_exit}\n"
            )
        if cmd == DOCKER_PS_ALL_NAMES_IDS_CMD:
            ids = container_ids or {}
            return _ssh_result(stdout="".join(f"{name} {ids.get(name, '')}\n" for name in names_on_host))
        return _ssh_result()

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


@pytest.mark.asyncio
async def test_customer_create_keeps_the_probes_after_removing_only_a_filler(svc_fixture, monkeypatch):
    """The filler's removal is the only host change; nothing is listed or probed again."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=_probe_with_containers("filler_x")
    )
    # the backend protects a preempted filler's volume until its own filler delete
    payload = _deploy_payload(active_volume_names=["volume_x"])

    result = await _run_create_container(svc, payload)

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    # the rm and its confirming listing are one host command; the listing that finds the filler
    # at SSH connect is the other
    assert _cmds(ssh_client) == [
        DOCKER_PS_ALL_NAMES_IDS_CMD,
        _remove_and_list_containers_command(["filler_x"], []),
    ]
    assert _relisting_commands(ssh_client) == []
    svc.probe_prerun_host.assert_awaited_once()
    svc.probe_volume_host.assert_awaited_once()


@pytest.mark.asyncio
async def test_customer_create_removes_the_filler_at_ssh_connect_by_the_id_it_was_listed_under(
    svc_fixture, monkeypatch
):
    """A same-name filler created after the listing is not the instance the removal targets."""
    svc = svc_fixture
    filler_id = "f" * 64
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=_probe_with_containers("filler_x"), container_ids={"filler_x": filler_id}
    )

    result = await _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _removal_commands(ssh_client) == [_remove_and_list_containers_command([filler_id], [])]


_FILLER_X_REMOVAL = _remove_and_list_containers_command(["filler_x"], [])
_POD_OLD_REMOVAL = _remove_and_list_containers_command(["pod_old"], ["volume_old"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("containers", "active_volume_names", "docker_rm_exit", "ps_after_rm", "listing_after_rm_exit", "removals"),
    [
        # a pod removed beside the filler: the filler at SSH connect, the pod at the cleanup step
        (("filler_x", "pod_old"), ["volume_x"], 0, "", 0, [_FILLER_X_REMOVAL, _POD_OLD_REMOVAL]),
        # the filler's volume removed by the create
        (("filler_x",), [], 0, "", 0, [_remove_and_list_containers_command(["filler_x"], ["volume_x"])]),
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
async def test_customer_create_without_a_filler_runs_the_same_commands_as_before(
    svc_fixture, monkeypatch
):
    """Keeping the probes changes nothing when the cleanup removes nothing (the list is the one before it).
    Removing fillers at SSH connect adds one read-only listing, run beside the probes."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=_probe_with_containers("pod_keep")
    )

    result = await _run_create_container(
        svc, _deploy_payload(active_container_names=["pod_keep"], active_volume_names=["volume_keep"])
    )

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _cmds(ssh_client) == [DOCKER_PS_ALL_NAMES_IDS_CMD]
    svc.probe_prerun_host.assert_awaited_once()
    svc.probe_volume_host.assert_awaited_once()


def _df_record(avail_bytes: int) -> str:
    # `df -P -B1` output with "\r" in place of "\n", as the removal command's DF line carries it
    return f"0\tFilesystem 1-blocks Used Available Capacity Mounted on\r/dev/vda1 0 0 {avail_bytes} 50% /hostfs\r"


def _fresh_sizing_payload(**over):
    # the DAH-2183 contract: the volume is sized on the host's df
    return _deploy_payload(disk_share=0.5, volume_limit_gb=1000, min_volume_gb=2, **over)


@pytest.mark.asyncio
async def test_a_filler_removal_sizes_the_volume_on_the_df_read_after_its_rm(svc_fixture, monkeypatch):
    """Both dfs clear the minimum but size different volumes; the early one predates the disk the rm freed."""
    svc = svc_fixture
    gb = ds_module._FRESH_SIZING_GB_BYTES
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=_probe_with_containers("filler_x"), df_after_rm=_df_record(140 * gb)
    )
    svc.probe_volume_host = AsyncMock(
        return_value=VolumeHostProbe(
            docker_root_dir="/var/lib/docker",
            df_avail_bytes=100 * gb,
            vloopback_volume_names=[],
            loopback_plugin_enabled=True,
        )
    )
    monkeypatch.delattr(svc, "resolve_volume_sizing")

    result = await _run_create_container(svc, _fresh_sizing_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _cmds(ssh_client) == [_remove_and_list_containers_command(["filler_x"], [], with_df=True)]
    svc.probe_volume_host.assert_awaited_once()
    # 0.5 x (df - 20 GB overhead), two thirds of it the volume: 100 GB -> 26 GB, 140 GB -> 40 GB
    assert svc.create_local_volume.await_args.kwargs["limit"] == 40


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "df_after_rm", [None, "137\tFilesystem\r/dev/vda1 9 2 42"], ids=["not_read", "died_mid_output"]
)
async def test_a_filler_removal_without_its_df_measures_the_volume_facts_again(svc_fixture, monkeypatch, df_after_rm):
    """The removal's df was not read, or not to its end: the early df is not used in its place."""
    svc = svc_fixture
    gb = ds_module._FRESH_SIZING_GB_BYTES
    _wire_customer_create_over_the_host(svc, monkeypatch, probe=_probe_with_containers("filler_x"), df_after_rm=df_after_rm)
    svc.probe_volume_host = AsyncMock(
        side_effect=[VolumeHostProbe("/var/lib/docker", size * gb, [], True) for size in (100, 140)]
    )
    monkeypatch.delattr(svc, "resolve_volume_sizing")

    result = await _run_create_container(svc, _fresh_sizing_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    # sized on the live probe's 140 GB, not on the early 100 GB
    assert svc.create_local_volume.await_args.kwargs["limit"] == 40


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


def _record_removal(ssh_client, events: list[str]) -> None:
    host = ssh_client.run.side_effect

    async def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker rm -fv"):
            events.append("removal issued")
            answer = await host(cmd, *args, **kwargs)
            events.append("removal confirmed")
            return answer
        return await host(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run


@pytest.mark.asyncio
async def test_customer_create_removes_the_filler_before_the_image_inspect_and_the_probes_finish(
    svc_fixture, monkeypatch
):
    """The kill is issued at SSH connect and the cleanup step does not issue it again."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=_probe_with_containers("filler_x")
    )
    events: list[str] = []
    _record_removal(ssh_client, events)
    docker_client = _docker_client(svc)
    inspect_image = docker_client.local_image_repo_digests
    probe = svc.probe_prerun_host.return_value

    async def slow_inspect(*, image):
        await asyncio.sleep(0.05)
        events.append("image inspected")
        return await inspect_image(image=image)

    async def slow_probe(*args, **kwargs):
        await asyncio.sleep(0.05)
        events.append("host probed")
        return probe

    docker_client.local_image_repo_digests = slow_inspect
    svc.probe_prerun_host = AsyncMock(side_effect=slow_probe)

    result = await _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert events[:2] == ["removal issued", "removal confirmed"]
    assert sorted(events[2:]) == ["host probed", "image inspected"]
    assert _removal_commands(ssh_client) == [_remove_and_list_containers_command(["filler_x"], [])]


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_saw_the_filler", [True, False])
async def test_customer_create_hands_on_a_listing_without_the_filler_whichever_side_won_the_race(
    svc_fixture, monkeypatch, probe_saw_the_filler
):
    """The probe lists the host before or after the removal; both end without the filler, and the
    early df stays (read beside the removal it can only read less free space)."""
    svc = svc_fixture
    probe = (
        _probe_with_containers("filler_x", "pod_keep")
        if probe_saw_the_filler
        else _probe_with_containers("pod_keep")
    )
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=probe, containers_on_host=("filler_x", "pod_keep")
    )

    result = await _run_create_container(
        svc,
        _deploy_payload(
            active_container_names=["pod_keep"], active_volume_names=["volume_keep", "volume_x"]
        ),
    )

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _removal_commands(ssh_client) == [_remove_and_list_containers_command(["filler_x"], [])]
    assert _relisting_commands(ssh_client) == []
    handed_on = _probe_kwarg(svc.select_affordable_cache_volumes)
    assert handed_on.container_names == ("pod_keep",)
    assert handed_on.mounted_volume_names == ("volume_keep",)
    svc.probe_volume_host.assert_awaited_once()


@pytest.mark.asyncio
async def test_power_restore_cache_reclaim_and_docker_run_wait_for_the_early_removal(
    svc_fixture, monkeypatch
):
    """Safety order: a PEARL filler is gone before its cap is lifted, and the Dolphin cache
    is reclaimed only once the filler that mounts it is gone."""
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc, monkeypatch, probe=_probe_with_containers("filler_x"), docker_rm_seconds=0.05
    )
    events: list[str] = []
    _record_removal(ssh_client, events)

    def recorder(event: str, returns=None) -> AsyncMock:
        return AsyncMock(side_effect=lambda *args, **kwargs: events.append(event) or returns)

    svc._restore_gpu_power_for_uncapped_pod = recorder("power restore")
    svc.reclaim_dphn_cache_for_rental = recorder("cache reclaim", returns=False)
    svc._run_rental_docker_create_with_port_retry = recorder("docker run")

    result = await _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"]))

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert events[:2] == ["removal issued", "removal confirmed"]
    assert sorted(events[2:]) == ["cache reclaim", "docker run", "power restore"]


def _removals_at_ssh_connect_left_running() -> list[asyncio.Task]:
    return [
        task
        for task in asyncio.all_tasks()
        if "remove_fillers_at_ssh_connect" in repr(task.get_coro()) and not task.done()
    ]


_HOSTILE_HOST_TEXT = "HOSTILE-BANNER-6f1c\nforged log line"


def _relisting_fails_after_the_first(ssh_client) -> None:
    # the listing that finds the filler works; the one after the failed rm does not, so the rm's error is raised
    host = ssh_client.run.side_effect
    listings = 0

    async def run(cmd, *args, **kwargs):
        nonlocal listings
        if cmd == DOCKER_PS_ALL_NAMES_IDS_CMD:
            listings += 1
            if listings > 1:
                return _ssh_result(exit_status=1)
        return await host(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("docker_rm_seconds", "docker_rm_raises", "failed_before_the_create", "logged_error_type"),
    [
        # the create fails while the removal is in flight: the removal is cancelled, nothing logged
        (5, None, False, None),
        # the removal failed first on a host's crafted text: only the error's type reaches the log
        (0, asyncssh.ConnectionLost(_HOSTILE_HOST_TEXT), True, "ConnectionLost"),
    ],
    ids=["removal_in_flight", "removal_failed"],
)
async def test_create_failing_before_the_cleanup_settles_the_removal_at_ssh_connect(
    svc_fixture, monkeypatch, caplog, docker_rm_seconds, docker_rm_raises, failed_before_the_create, logged_error_type
):
    svc = svc_fixture
    ssh_client = _wire_customer_create_over_the_host(
        svc,
        monkeypatch,
        probe=_probe_with_containers("filler_x"),
        docker_rm_seconds=docker_rm_seconds,
        docker_rm_raises=docker_rm_raises,
    )
    _relisting_fails_after_the_first(ssh_client)

    async def deleted(*args, **kwargs):
        if failed_before_the_create:
            await asyncio.sleep(0.05)
        raise RuntimeError("deleted")

    monkeypatch.setattr(svc, "_abort_if_cancelled_by_delete", AsyncMock(side_effect=deleted))

    result = await asyncio.wait_for(
        _run_create_container(svc, _deploy_payload(active_volume_names=["volume_x"])), timeout=2
    )

    assert type(result).__name__ == "FailedContainerRequest"
    assert result.failure_step == "ssh_connect"
    assert _removals_at_ssh_connect_left_running() == []
    settled = [
        record.msg.extra
        for record in caplog.records
        if getattr(record.msg, "message", None) == "Filler removal at SSH connect failed before the cleanup step"
    ]
    assert [extra["error_type"] for extra in settled] == ([logged_error_type] if logged_error_type else [])
    logged = "".join(
        record.msg.to_full_string() if hasattr(record.msg, "to_full_string") else record.getMessage()
        for record in caplog.records
    )
    assert "HOSTILE-BANNER-6f1c" not in logged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filler_create", "commands"),
    [
        # a filler create keeps protecting its sibling bundle (DAH-2465) and lists nothing at SSH connect
        (True, []),
        # a bootstrap restore removes the filler at the cleanup step
        (False, [_FILLER_X_REMOVAL]),
    ],
    ids=["filler_create", "bootstrap_restore"],
)
async def test_only_a_customer_create_removes_fillers_at_ssh_connect(
    svc_fixture, monkeypatch, filler_create, commands
):
    from payload_models.payloads import WorkloadKind

    svc = svc_fixture
    filler = "filler_sibling" if filler_create else "filler_x"
    ssh_client = _wire_customer_create_over_the_host(svc, monkeypatch, probe=_probe_with_containers(filler))
    monkeypatch.setattr(svc, "_run_bootstrap_restore", AsyncMock())
    if filler_create:
        payload = _deploy_payload(
            workload_kind=WorkloadKind.FILLER,
            active_container_names=["filler_sibling"],
            active_volume_names=["volume_sibling"],
        )
    else:
        payload = _deploy_payload(active_volume_names=["volume_x"])
        payload.bootstrap_restore = Mock(restore_log_id="restore-log")

    result = await _run_create_container(svc, payload)

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert _cmds(ssh_client) == commands


def test_probe_without_containers_drops_them_and_their_mounts_only():
    probe = _probe(
        container_names=("filler_x", "pod_keep"),
        volumes=(ProbedVolume("volume_x", "vloopback:latest"), ProbedVolume("volume_keep", "local")),
        mounted_volume_names=("volume_x", "volume_keep", "dphn_cache_a"),
        port_check_container_names=("health_check_1",),
    )

    adjusted = probe.without_containers(["filler_x"], ["volume_x"])

    assert adjusted.container_names == ("pod_keep",)
    assert adjusted.mounted_volume_names == ("volume_keep", "dphn_cache_a")
    assert adjusted.volumes == probe.volumes
    assert adjusted.port_check_container_names == ("health_check_1",)
    failed_sections = _probe(container_names=None, mounted_volume_names=None)
    assert failed_sections.without_containers(["filler_x"], ["volume_x"]) == failed_sections


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


_OVERLAPPED_ROWS = (
    "Prerun host probe (parallel)",
    "Volume host probe (parallel)",
    "GPU power restore (parallel)",
)


def _slow_overlapped_operations(svc, monkeypatch) -> None:
    async def slow_probe(*args, **kwargs):
        await asyncio.sleep(0.03)
        return None

    async def slow_raise(*args, **kwargs):
        await asyncio.sleep(0.03)
        return 0

    svc.probe_prerun_host = AsyncMock(side_effect=slow_probe)
    svc.probe_volume_host = AsyncMock(side_effect=slow_probe)
    monkeypatch.setattr("services.docker_service.raise_low_power_limits_to_default", slow_raise)


def _overlapped_rows_ms(result) -> dict[str, int]:
    return {p.name.value: p.duration for p in result.profilers if p.name.value in _OVERLAPPED_ROWS}


@pytest.mark.asyncio
async def test_cached_create_profiles_each_overlapped_operation_with_its_own_duration(
    svc_fixture, monkeypatch
):
    """DAH-3980: the step rows show only the residual wait; zero there must not read as free."""
    svc = svc_fixture
    _wire_early_probes(svc, monkeypatch, image_present=True)
    _slow_overlapped_operations(svc, monkeypatch)

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    rows_ms = _overlapped_rows_ms(result)
    assert sorted(rows_ms) == sorted(_OVERLAPPED_ROWS)
    assert all(ms >= 25 for ms in rows_ms.values()), rows_ms


@pytest.mark.asyncio
async def test_discarded_early_probe_gets_no_overlapped_row(svc_fixture, monkeypatch):
    """A probe rerun at its step is in that step's row; the discarded early run is not counted again."""
    svc = svc_fixture
    _wire_early_probes(svc, monkeypatch, image_present=True)
    _slow_overlapped_operations(svc, monkeypatch)
    svc.clean_stale_vloopback_volumes = AsyncMock(return_value=["volume_old"])

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert svc.probe_volume_host.await_count == 2
    assert sorted(_overlapped_rows_ms(result)) == [
        "GPU power restore (parallel)",
        "Prerun host probe (parallel)",
    ]


@pytest.fixture
def svc_fixture():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


_PORT_ALLOCATED_REFUSAL = RuntimeError(
    "Docker SDK run container failed: 500 Server Error: driver failed programming external "
    "connectivity on endpoint pod_x: Bind for 0.0.0.0:20001 failed: port is already allocated"
)


def _wire_real_docker_run(svc, monkeypatch, *, refusals: int) -> list[str]:
    """Probe on, the real `docker run` retry; returns the order of port-check waits and runs."""
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    monkeypatch.setattr(settings, "PORT_COLLISION_RETRY_ENABLED", False)
    _wire(svc, monkeypatch, _deploy_ssh_client(), probe_result=_probe())
    monkeypatch.delattr(svc, "_run_rental_docker_create_with_port_retry")  # the real retry loop
    monkeypatch.setattr(svc, "_remove_failed_rental_container_for_retry", AsyncMock())
    monkeypatch.setattr("services.docker_service._PORT_ALLOCATED_RETRY_SLEEP_SEC", 0)
    calls: list[str] = []

    async def port_check_wait(**kwargs) -> tuple[bool, str]:
        listing = "early" if kwargs.get("probed_container_names") is not None else "live"
        calls.append(f"port_check_wait:{listing}")
        return False, "No port check containers found"

    async def run_container(spec) -> None:
        calls.append("docker_run")
        if calls.count("docker_run") <= refusals:
            raise _PORT_ALLOCATED_REFUSAL

    svc.wait_for_port_check_containers = port_check_wait
    _docker_client(svc).run_container = run_container
    return calls


@pytest.mark.asyncio
async def test_a_docker_run_that_succeeds_reads_only_the_early_port_check_listing(svc_fixture, monkeypatch):
    svc = svc_fixture
    calls = _wire_real_docker_run(svc, monkeypatch, refusals=0)

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert calls == ["port_check_wait:early", "docker_run"]


@pytest.mark.asyncio
async def test_the_first_port_refusal_reads_the_port_check_listing_live_before_the_retry(
    svc_fixture, monkeypatch
):
    """A port check started after the probe's listing holds a rent port; only a live listing sees it."""
    svc = svc_fixture
    calls = _wire_real_docker_run(svc, monkeypatch, refusals=2)

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert calls == ["port_check_wait:early", "docker_run", "port_check_wait:live", "docker_run", "docker_run"]


@pytest.mark.asyncio
async def test_a_port_check_removed_live_is_not_followed_by_the_five_second_wait(svc_fixture, monkeypatch):
    """The live listing removed the port check that held the port: the retry runs at once."""
    svc = svc_fixture
    calls = _wire_real_docker_run(svc, monkeypatch, refusals=1)
    monkeypatch.setattr("services.docker_service._PORT_ALLOCATED_RETRY_SLEEP_SEC", 5)

    async def port_check_wait(**kwargs) -> tuple[bool, str]:
        if kwargs.get("probed_container_names") is not None:
            calls.append("port_check_wait:early")
            return False, "No port check containers found"
        calls.append("port_check_wait:live")
        return True, "Port check containers forcefully removed"

    svc.wait_for_port_check_containers = port_check_wait
    real_sleep = asyncio.sleep
    retry_sleeps: list[float] = []

    async def recording_sleep(delay, *args, **kwargs):
        if delay == 5:
            retry_sleeps.append(delay)
            delay = 0
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", recording_sleep)

    result = await _run_create_container(svc, _deploy_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    assert calls == ["port_check_wait:early", "docker_run", "port_check_wait:live", "docker_run"]
    assert retry_sleeps == []
