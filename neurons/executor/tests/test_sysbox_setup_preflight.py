"""DAH-3319: nvidia_docker_sysbox_setup.sh never touches a host that holds a rental.

Install mode stops on a stopped rental and leaves the vloopback plugin alone under a rental; the
per-requirement PASS / FIX preflight cases were cut as low-signal in DAH-4027.

The script is run under bash with stub commands on PATH and the files it reads under a fixture
SYSBOX_SETUP_HOST_ROOT.
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
REAL_TOOLS = ["bash", "grep", "sed", "awk", "head", "tail", "cut", "ls", "dirname", "cat", "seq", "rm"]

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
            "image inspect") [ -z "${STUB_NO_VERIFY_IMAGE:-}" ] || { echo "Error: No such image: $3" >&2; exit 1; } ;;
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
        echo "/dev/nvme0n1p1 ${STUB_DF_TOTAL_KB:-1048576000} 1 1 1% /"
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


def test_install_mode_does_not_change_the_plugin_under_a_rental(tmp_path):
    proc = run_script(tmp_path, env={"STUB_VLOOPBACK": "relative", "STUB_PODS": "pod_4f1c"})
    assert proc.returncode == 1
    assert "do not work on this host: rentals found (pod_4f1c)" in proc.stdout
    assert _plugin_changes(tmp_path) == []


# ── the script as a provider runs it ─────────────────────────────────────────


def test_install_mode_stops_on_a_stopped_rental(tmp_path):
    # a renter's stopped pod is still a rental; the upgrade path removes every stopped container
    proc = run_script(tmp_path, env={"STUB_SYSBOX_VERSION": "0.6.6", "STUB_STOPPED_POD": "pod_abc123"})
    assert proc.returncode == 1
    assert "Rentals found (pod_* containers, running or stopped). Cannot proceed." in proc.stdout
    assert "docker ps -a --filter name=pod_" in proc.stdout
    assert "pod_abc123" in proc.stdout
    assert "Removing stopped containers" not in proc.stdout
