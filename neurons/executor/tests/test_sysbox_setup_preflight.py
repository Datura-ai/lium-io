"""DAH-3319: the preflight in nvidia_docker_sysbox_setup.sh — one PASS / FIX line per host
requirement with the command that fixes it, run before anything is installed and standalone as
`--check` (ticket-0309: a provider reinstalled sysbox three times; the cause was a host setting
nobody had checked).

Same harness as test_sysbox_setup_apt_repo.py: the script is run under bash with stub commands
on PATH and the files it reads (/proc/modules, /etc/os-release, /etc/docker/daemon.json, ...)
under a fixture SYSBOX_SETUP_HOST_ROOT. The check functions are sourced with SYSBOX_SETUP_LIB=1.
"""

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

ANSI = re.compile(r"\x1b\[[0-9;]*m")

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nvidia_docker_sysbox_setup.sh"))

# the real tools the check functions call, so PATH can be built without the host's jq
REAL_TOOLS = ["bash", "grep", "sed", "awk", "head", "tail", "cut", "ls", "dirname", "cat", "seq", "rm", "mkdir"]

STUBS = {
    "id": '#!/bin/bash\necho "${STUB_UID:-0}"\n',
    "uname": textwrap.dedent(
        """\
        #!/bin/bash
        case "$1" in -r) echo "${STUB_KERNEL:-6.8.0-45-generic}" ;; -m) echo "${STUB_ARCH:-x86_64}" ;; esac
        """
    ),
    # The vloopback plugin and its volumes are state in $SYSBOX_SETUP_HOST_ROOT/.stub-docker: `plugin` holds
    # "<enabled> <DATA_DIR>" (seeded once from STUB_VLOOPBACK: ok / absent / relative / disabled), `calls` every
    # docker command line, `volumes` / `removed` the volumes created and removed. Mountpoints are <MOUNT_DIR>/<name>
    # as upstream builds them; STUB_RELATIVE_MOUNTPOINT forces a malformed one.
    "docker": textwrap.dedent(
        """\
        #!/bin/bash
        [ -z "${STUB_NO_DOCKER_DAEMON:-}" ] || { echo "Cannot connect to the Docker daemon" >&2; exit 1; }
        st="$SYSBOX_SETUP_HOST_ROOT/.stub-docker"
        echo "$*" >> "$st/calls"
        root="${STUB_DOCKER_ROOT:-$SYSBOX_SETUP_HOST_ROOT/var/lib/docker}"
        if [ ! -e "$st/seeded" ]; then
            : > "$st/seeded"
            case "${STUB_VLOOPBACK:-ok}" in
                ok) echo "true $root/loopback" > "$st/plugin" ;;
                relative) echo "true loopback" > "$st/plugin" ;;
                disabled) echo "false $root/loopback" > "$st/plugin" ;;
            esac
        fi
        [ -f "$st/plugin" ] && read -r enabled data_dir < "$st/plugin"
        case "$1 $2" in
            "version --format") echo "${STUB_DOCKER_VERSION:-28.5.2}" ;;
            "ps ") exit 0 ;;
            "ps --format") [ -z "${STUB_RUNNING:-}" ] || echo "$STUB_RUNNING" ;;   # install mode's running-container sort
            "ps -a")   # every container, stopped ones too
                [ -z "${STUB_PS_A_HANGS:-}" ] || exit 124
                echo "${STUB_STOPPED_POD:-}"
                [ "$4" != name=pod_ ] || [ -z "${STUB_PODS:-}" ] || echo "$STUB_PODS" ;;
            "ps --filter")
                case "$3" in
                    name=pod_) [ -z "${STUB_PODS:-}" ] || echo "$STUB_PODS" ;;
                    *) echo "${STUB_PORT_CONTAINER:-executor-1}" ;;
                esac ;;
            "info ") echo " Runtimes: io.containerd.runc.v2 nvidia runc sysbox-runc" ;;   # the "already working?" probe of install mode
            "info --format")
                case "$3" in
                    *DockerRootDir*) echo "$root" ;;
                    *Runtimes*) echo "${STUB_DOCKER_RUNTIMES:-\\"runc\\", \\"sysbox-runc\\"}" ;;
                esac ;;
            "plugin inspect")
                # the real one prints a blank line on stdout before it fails on a missing plugin
                [ -f "$st/plugin" ] || { echo; echo "Error: No such plugin: $5" >&2; exit 1; }
                case "$4" in
                    *Enabled*) echo "$enabled" ;;
                    *Env*) echo "DATA_DIR=$data_dir"; echo "LOG_LEVEL=2" ;;
                esac ;;
            "plugin install")
                [ -z "${STUB_PLUGIN_INSTALL_FAILS:-}" ] || { echo "Error response from daemon: Head https://registry-1.docker.io/v2/ashald/docker-volume-loopback/manifests/latest: net/http: TLS handshake timeout" >&2; exit 1; }
                for arg in "$@"; do case "$arg" in DATA_DIR=*) echo "true ${arg#DATA_DIR=}" > "$st/plugin" ;; esac; done ;;
            "plugin disable")
                [ -z "${STUB_PLUGIN_IN_USE:-}" ] || { echo "Error response from daemon: plugin vloopback:latest is in use" >&2; exit 1; }
                echo "false $data_dir" > "$st/plugin" ;;
            "plugin set") echo "$enabled ${4#DATA_DIR=}" > "$st/plugin" ;;
            "plugin enable") echo "true $data_dir" > "$st/plugin" ;;
            "image inspect") [ -z "${STUB_NO_VERIFY_IMAGE:-}${STUB_NO_EXECUTOR_IMAGE:-}" ] || { echo "Error: No such image: $3" >&2; exit 1; } ;;
            "volume create")
                [ -z "${STUB_VOLUME_CREATE_FAILS:-}" ] || { echo "Error response from daemon: create ${!#}: VolumeDriver.Create: no space left on device" >&2; exit 1; }
                echo "${!#}" >> "$st/volumes"
                # the daemon finished creating it after `timeout` killed the CLI
                [ -z "${STUB_VOLUME_CREATE_HANGS:-}" ] || exit 124
                echo "${!#}" ;;
            "volume inspect") if [ -n "${STUB_RELATIVE_MOUNTPOINT:-}" ]; then echo "${!#}"; else echo "/mnt/${!#}"; fi ;;
            "volume rm")
                [ -z "${STUB_VOLUME_RM_FAILS:-}" ] || { echo "Error response from daemon: remove ${!#}: VolumeDriver.Unmount: context deadline exceeded" >&2; exit 1; }
                if [ -s "$st/containers" ]; then echo "Error response from daemon: remove ${!#}: volume is in use - [$(head -1 "$st/containers")]" >&2; exit 1; fi
                echo "${!#}" >> "$st/removed" ;;
            "rm -f") grep -vx "$3" "$st/containers" > "$st/containers.new" 2>/dev/null; cat "$st/containers.new" > "$st/containers" ;;
            "run --rm")
                case " $* " in
                    *" -v "*)
                        # `timeout` killed the CLI; the container keeps running with the volume attached
                        [ -z "${STUB_VLOOPBACK_RUN_HANGS:-}" ] || { echo "$4" >> "$st/containers"; exit 124; }
                        # the probe printed its token, then docker run failed (exit) or hung (124) before completing
                        [ -z "${STUB_VLOOPBACK_RUN_EXIT:-}" ] || { echo lium-vloopback-ok; [ "$STUB_VLOOPBACK_RUN_EXIT" != 124 ] || echo "$4" >> "$st/containers"; exit "$STUB_VLOOPBACK_RUN_EXIT"; }
                        [ -z "${STUB_VLOOPBACK_MOUNT_FAILS:-}" ] && echo lium-vloopback-ok || {
                            echo "docker: Error response from daemon: failed to create task for container: OCI runtime create failed: error during container init: error setting up ID-mapped mount on path 206/fs (likely means idmapped mounts are not supported on the filesystem at this path): lstat 206: no such file or directory: unknown." >&2
                            exit 125; } ;;
                    *" --storage-opt "*)
                        [ -z "${STUB_STORAGE_OPT_UNSUPPORTED:-}" ] || { echo "docker: Error response from daemon: --storage-opt is supported only for overlay over xfs with 'pquota' mount option." >&2; exit 125; }
                        [ "${STUB_STORAGE_OPT_PROBE:-}" != hangs ] || exit 124
                        [ "${STUB_STORAGE_OPT_PROBE:-}" != errors ] || { echo "docker: Error response from daemon: failed to create task for container: context deadline exceeded" >&2; exit 125; } ;;
                    *) [ -z "${STUB_SYSBOX_RUN_FAILS:-}" ] && echo ok || { echo "OCI runtime create failed" >&2; exit 125; } ;;
                esac ;;
        esac
        """
    ),
    "nvidia-smi": textwrap.dedent(
        """\
        #!/bin/bash
        case "$1" in
            --query-gpu=driver_version) echo "${STUB_NV_DRIVER:-580.65.06}" ;;
            --query-gpu=name) for _ in $(seq "${STUB_NV_GPUS:-8}"); do echo "${STUB_NV_NAME:-NVIDIA H100 80GB HBM3}"; done ;;
            --query-gpu=memory.total) for _ in $(seq "${STUB_NV_GPUS:-8}"); do echo "${STUB_NV_MEM_MIB:-81559}"; done ;;
            --list-gpus) for i in $(seq "${STUB_NV_GPUS:-8}"); do echo "GPU $((i - 1)): NVIDIA H100 80GB HBM3"; done ;;
        esac
        """
    ),
    # records "<seconds> <command>" in .stub-docker/timeouts, then runs the command
    "timeout": '#!/bin/bash\necho "$*" >> "$SYSBOX_SETUP_HOST_ROOT/.stub-docker/timeouts"\nshift\nexec "$@"\n',
    "nvidia-container-cli": '#!/bin/bash\nprintf "cli-version: 1.17.8\\nlib-version: 1.17.8\\n"\n',
    # the real `sysbox-runc --version`: the name alone on line 1, the version on line 2
    "sysbox-runc": '#!/bin/bash\nprintf "sysbox-runc\\n\\tversion:\\t${STUB_SYSBOX_VERSION:-0.7.1}\\n\\tcommit:\\tabc123\\n"\n',
    "fusermount3": "#!/bin/bash\nexit 0\n",
    "lspci": '#!/bin/bash\necho "${STUB_LSPCI:-17:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)}"\n',
    # loads the legacy iptables modules into the fixture /proc/modules, or fails with STUB_MODPROBE_FAILS
    "modprobe": textwrap.dedent(
        """\
        #!/bin/bash
        [ -z "${STUB_MODPROBE_FAILS:-}" ] || exit 1
        printf "ip_tables 1 0\\niptable_nat 1 0\\niptable_filter 1 0\\n" >> "$SYSBOX_SETUP_HOST_ROOT/proc/modules"
        """
    ),
    "ss": textwrap.dedent(
        """\
        #!/bin/bash
        # ss -Hltnp "sport = :PORT": one line when STUB_BUSY_PORT is that port
        port=${!#}; port=${port##*:}
        [ "${STUB_BUSY_PORT:-}" = "$port" ] && echo "LISTEN 0 128 0.0.0.0:$port 0.0.0.0:* users:((\\"${STUB_BUSY_PROC:-sshd}\\",pid=812,fd=3))"
        exit 0
        """
    ),
    "ufw": '#!/bin/bash\nprintf "%b\\n" "${STUB_UFW_STATUS:-Status: inactive}"\n',
    "df": textwrap.dedent(
        """\
        #!/bin/bash
        echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
        echo "/dev/nvme0n1p1 ${STUB_DF_TOTAL_KB:-1048576000} 1 ${STUB_DF_AVAIL_KB:-943718400} 1% /"
        """
    ),
}

GOOD_FILES = {
    "etc/os-release": 'NAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n',
    "proc/modules": "ip_tables 32768 3 iptable_nat,iptable_filter\niptable_nat 16384 1\niptable_filter 16384 1\n",
    "proc/driver/nvidia/version": "NVRM version: 580.65.06\n",
    "var/lib/docker/.keep": "",
}


def _bin_dir(tmp_path: Path, *, with_jq: bool, without: tuple[str, ...] = ()) -> Path:
    """Stubs first, then the real tools the checks need — never the host's jq unless asked."""
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        if name in without:
            (stubs / name).unlink(missing_ok=True)
            continue
        path = stubs / name
        path.write_text(body)
        path.chmod(0o755)
    tools = REAL_TOOLS + (["jq"] if with_jq else [])
    for tool in tools:
        real = shutil.which(tool)
        if real is None:
            if tool == "jq":
                pytest.skip("jq not installed")
            raise RuntimeError(f"{tool} not on PATH")
        link = stubs / tool
        if not link.exists():
            link.symlink_to(real)
    return stubs


def _host_root(tmp_path: Path, files: dict[str, str | None] | None = None) -> Path:
    """The fixture tree the script reads through SYSBOX_SETUP_HOST_ROOT; a None value leaves that file out."""
    root = tmp_path / "root"
    (root / ".stub-docker").mkdir(parents=True, exist_ok=True)
    for rel, content in {**GOOD_FILES, **(files or {})}.items():
        if content is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def _script_copy(tmp_path: Path) -> Path:
    """The script in its own directory, so the `.env` it may read next to itself is the test's, never the checkout's."""
    copy_dir = tmp_path / "executor"
    copy_dir.mkdir(exist_ok=True)
    copy = copy_dir / "nvidia_docker_sysbox_setup.sh"
    shutil.copy(SCRIPT, copy)
    return copy


class CheckResult(NamedTuple):
    rc: int
    output: str
    fix_count: int


def run_check(
    tmp_path: Path,
    function: str,
    *,
    env: dict[str, str] | None = None,
    files: dict[str, str | None] | None = None,
    with_jq: bool = False,
    without: tuple[str, ...] = (),
) -> CheckResult:
    """Source the installer's functions and run one check."""
    stubs = _bin_dir(tmp_path, with_jq=with_jq, without=without)
    root = _host_root(tmp_path, files)
    bash_snippet = (
        f"SYSBOX_SETUP_LIB=1 . {SCRIPT}\nset +e\n{function}\nrc=$?\necho \"RC=$rc FIX=$PREFLIGHT_FIX PASS=$PREFLIGHT_PASS SKIP=$PREFLIGHT_SKIP\"\n"
    )
    proc = subprocess.run(
        ["bash", "-c", bash_snippet],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": str(stubs), "SYSBOX_SETUP_HOST_ROOT": str(root), **(env or {})},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = ANSI.sub("", proc.stdout)
    summary_counters = dict(kv.split("=") for kv in out.strip().splitlines()[-1].split())
    return CheckResult(int(summary_counters["RC"]), out, int(summary_counters["FIX"]))


def run_script(
    tmp_path: Path,
    *args: str,
    env: dict[str, str] | None = None,
    files: dict[str, str | None] | None = None,
    without: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """The installer itself, as a provider runs it, with the same stubs and fixture tree; stdout without colour."""
    stubs = _bin_dir(tmp_path, with_jq=False, without=without)
    root = _host_root(tmp_path, files)
    proc = subprocess.run(
        ["bash", str(_script_copy(tmp_path)), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
        env={"PATH": str(stubs), "SYSBOX_SETUP_HOST_ROOT": str(root), **(env or {})},
    )
    proc.stdout = ANSI.sub("", proc.stdout)
    return proc


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash installer")


# ── kernel ───────────────────────────────────────────────────────────────────


# ── docker ───────────────────────────────────────────────────────────────────


def test_docker_29_0_passes_but_says_untested(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker", env={"STUB_DOCKER_VERSION": "29.0.1"})
    assert rc == 0
    assert "PASS Docker 29.0.1 (29.0 is untested with sysbox" in out


def test_docker_29_1_passes_plainly(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker", env={"STUB_DOCKER_VERSION": "29.1.3"})
    assert rc == 0
    assert "PASS Docker 29.1.3.\n" in out


# ── nvidia ───────────────────────────────────────────────────────────────────


# ── iptables modules (ticket-0309) ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("env", "files", "package"),
    [
        ({"STUB_NV_DRIVER": "570.1.1"}, {}, "nvidia-driver-580-server &&"),
        ({"STUB_NV_DRIVER": "570.1.1"}, {"proc/driver/nvidia/version": "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.1.1\n"}, "nvidia-driver-580-server-open"),
        ({"STUB_NV_DRIVER": "570.1.1", "STUB_NV_NAME": "NVIDIA GeForce RTX 5090"}, {}, "nvidia-driver-580-server-open"),
    ],
)
def test_driver_fix_keeps_the_open_module_where_it_is_needed(tmp_path, env, files, package):
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver", env=env, files=files)
    assert rc == 1
    assert f"sudo apt-get install -y {package}" in out


def test_blackwell_without_a_driver_gets_the_open_module_from_pci(tmp_path):
    env = {"STUB_LSPCI": "17:00.0 3D controller: NVIDIA Corporation GB100 [B200] (rev a1)"}
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver", env=env, without=("nvidia-smi",))
    assert rc == 1
    assert "sudo apt-get install -y nvidia-driver-580-server-open" in out


# ── disk >= 1.5x VRAM (validators' MIN_DISK_TO_VRAM_RATE) ────────────────────


# ── free disk for the node's images and VerifyX's storage test ───────────────


def test_free_disk_needs_room_for_the_images_before_the_first_pull(tmp_path):
    # a 100 GB disk passes the 1.5x VRAM rule for one 48 GB card and then fails VerifyX
    env = {"STUB_DF_AVAIL_KB": str(97 * 1024 * 1024), "STUB_NO_EXECUTOR_IMAGE": "1"}
    rc, out, _ = run_check(tmp_path, "check_free_disk_for_node", env=env)
    assert rc == 1
    assert "FIX  Free disk 97.0 GB on " in out and "is below 140 GB" in out
    assert "Use a disk of at least 150 GB" in out


def test_free_disk_needs_only_the_storage_test_once_the_images_are_pulled(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_free_disk_for_node", env={"STUB_DF_AVAIL_KB": str(120 * 1024 * 1024)})
    assert rc == 0, out
    assert "PASS Free disk 120.0 GB on " in out and ">= 100 GB" in out


def test_free_disk_just_below_100_gib_is_not_rounded_up(tmp_path):
    env = {"STUB_DF_AVAIL_KB": str(100 * 1024 * 1024 - 40000)}
    rc, out, _ = run_check(tmp_path, "check_free_disk_for_node", env=env)
    assert rc == 1, out
    assert "FIX  Free disk 100.0 GB on " in out


# ── ports ────────────────────────────────────────────────────────────────────


# ── sysbox probe ─────────────────────────────────────────────────────────────


# ── vloopback: size-limited volumes (ticket-0331) ────────────────────────────
# ticket-0331, 22 Sep 2026: the fix support sent the provider by hand was "give the plugin an absolute data
# directory, then run `docker volume create -d vloopback -o size=10G t && docker volume inspect t`; the
# Mountpoint must start with /". Install mode now does that itself; these pin the three host shapes.

MUTATIONS = ("plugin install", "plugin disable", "plugin set", "plugin enable")


def _stub_state(tmp_path: Path) -> Path:
    return tmp_path / "root" / ".stub-docker"


def _docker_calls(tmp_path: Path) -> list[str]:
    calls = _stub_state(tmp_path) / "calls"
    return calls.read_text().splitlines() if calls.exists() else []


def _plugin_changes(tmp_path: Path) -> list[str]:
    return [call for call in _docker_calls(tmp_path) if call.startswith(MUTATIONS)]


@pytest.mark.parametrize("seed", ["absent", "relative"])
def test_check_asks_for_the_disk_limit_where_rentals_get_none(tmp_path, seed):
    rc, out, fix = run_check(
        tmp_path, "check_vloopback", env={"STUB_VLOOPBACK": seed, "STUB_STORAGE_OPT_UNSUPPORTED": "1"}
    )
    assert rc == 1 and fix == 1
    assert "FIX  Docker here cannot limit a container's disk" in out
    assert "sudo lium gpu-splitting setup --device" in out
    assert not [call for call in _docker_calls(tmp_path) if call.startswith(("volume ", "plugin "))]


def test_install_mode_does_not_change_the_plugin_under_a_rental(tmp_path):
    proc = run_script(tmp_path, env={"STUB_VLOOPBACK": "relative", "STUB_PODS": "pod_4f1c"})
    assert proc.returncode == 1
    assert "do not work on this host: rentals found (pod_4f1c)" in proc.stdout
    assert _plugin_changes(tmp_path) == []

# ── fuse3 ───────────────────────────────────────────────────────────────────


# ── the script as a provider runs it ─────────────────────────────────────────


def test_install_mode_stops_on_a_stopped_rental(tmp_path):
    # a renter's stopped pod is still a rental; the upgrade path removes every stopped container
    proc = run_script(tmp_path, env={"STUB_SYSBOX_VERSION": "0.6.6", "STUB_STOPPED_POD": "pod_abc123"})
    assert proc.returncode == 1
    assert "Rentals found (pod_* containers, running or stopped). Cannot proceed." in proc.stdout
    assert "docker ps -a --filter name=pod_" in proc.stdout
    assert "pod_abc123" in proc.stdout
    assert "Removing stopped containers" not in proc.stdout


def test_install_mode_loads_the_iptables_modules_itself(tmp_path):
    proc = run_script(tmp_path, files={"proc/modules": "nf_tables 311296 0\n"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Loaded the iptables modules (ip_tables iptable_nat iptable_filter) and made them load at boot." in proc.stdout
    assert "The preflight printed" not in proc.stdout
    conf = tmp_path / "root" / "etc" / "modules-load.d" / "lium-iptables.conf"
    assert conf.read_text() == "ip_tables\niptable_nat\niptable_filter\n"


# ── failure diagnostics (the block printed when the verify container does not start) ─────────




def test_aarch64_host_passes_the_arch_check(tmp_path):
    # GB300 / Grace: nestybox ships an arm64 .deb, so aarch64 is no longer a blocking FIX
    proc = run_script(tmp_path, "--check", env={"STUB_ARCH": "aarch64"})
    assert "PASS Architecture aarch64." in proc.stdout
