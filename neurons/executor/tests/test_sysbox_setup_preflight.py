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
REAL_TOOLS = ["bash", "grep", "sed", "awk", "head", "tail", "cut", "ls", "dirname", "cat", "seq", "rm", "mkdir", "readlink"]

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
                    *DriverStatus*) echo "${STUB_DRIVER_STATUS:-[[\\"Backing Filesystem\\",\\"extfs\\"]]}" ;;
                    *Containerd*) echo "${STUB_CONTAINERD_ADDRESS-/run/containerd/containerd.sock}" ;;
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
    "apparmor_parser": '#!/bin/bash\necho "$*" >> "$SYSBOX_SETUP_HOST_ROOT/.stub-docker/apparmor"\n',
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
        case "${!#}" in   # STUB_DATA_DF_TOTAL_KB: a data-root on its own big disk, the rest on the root disk
            */docker) [ -z "${STUB_DATA_DF_TOTAL_KB:-}" ] || { echo "/dev/md0 $STUB_DATA_DF_TOTAL_KB 1 ${STUB_DATA_DF_AVAIL_KB:-$STUB_DATA_DF_TOTAL_KB} 1% /data"; exit 0; } ;;
            *.content) [ -z "${STUB_CONTENT_DF_AVAIL_KB:-}" ] || { echo "/dev/sdb1 $((14 * 1024 ** 3)) 1 $STUB_CONTENT_DF_AVAIL_KB 1% /content"; exit 0; } ;;
        esac
        echo "${STUB_DF_SOURCE:-/dev/nvme0n1p1} ${STUB_DF_TOTAL_KB:-1048576000} 1 ${STUB_DF_AVAIL_KB:-943718400} 1% /"
        """
    ),
    # `ctr --address <socket> plugins ls -d` of the running containerd: its content and snapshotter roots
    "ctr": textwrap.dedent(
        """\
        #!/bin/bash
        [ "$3 $4 $5" = "plugins ls -d" ] && [ -z "${STUB_CTR_FAILS:-}" ] || exit 1
        r=${STUB_CTR_ROOT:-$SYSBOX_SETUP_HOST_ROOT/var/lib/containerd}
        printf '\\tType:\\tio.containerd.content.v1\\n\\tID:\\tcontent\\n\\tExports:\\t\\n\\t\\troot\\t%s\\n\\n' "$r/io.containerd.content.v1.content"
        printf '\\tType:\\tio.containerd.snapshotter.v1\\n\\tID:\\toverlayfs\\n\\tExports:\\t\\n\\t\\troot\\t%s\\n' "$r/io.containerd.snapshotter.v1.overlayfs"
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
        if isinstance(content, Link):
            path.symlink_to(content)
        else:
            path.write_text(content)
    return root


class Link(str):
    """A fixture entry that is a symlink to this target (as /proc/<pid>/ns/mnt reads)."""


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


SNAPSHOTTER = {"STUB_DRIVER_STATUS": '[["driver-type","io.containerd.snapshotter.v1"]]'}
_OVERLAYFS = "io.containerd.snapshotter.v1.overlayfs"
SNAPSHOTS = f"var/lib/containerd/{_OVERLAYFS}"


def test_disk_is_measured_where_the_containerd_image_store_keeps_containers(tmp_path):
    # webg8845, 27 Oct 2026: data-root on a 14 TB disk, but Docker 29's containerd image store kept the
    # containers on the 436 GB root disk; the check read the data-root and passed
    env = {**SNAPSHOTTER, "STUB_DATA_DF_TOTAL_KB": str(14 * 1024**3), "STUB_DF_TOTAL_KB": str(436 * 1024**2)}
    files = {f"{SNAPSHOTS}/.keep": ""}
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=env, files=files)
    assert rc == 1, out
    assert "FIX  Disk 436.0 GB on " in out and f"/{SNAPSHOTS} is below" in out
    assert '"features": {"containerd-snapshotter": false}' in out


def test_disk_follows_the_data_root_without_the_containerd_image_store(tmp_path):
    env = {"STUB_DATA_DF_TOTAL_KB": str(14 * 1024**3), "STUB_DF_TOTAL_KB": str(436 * 1024**2)}
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=env, files={f"{SNAPSHOTS}/.keep": ""})
    assert rc == 0, out
    assert "containerd" not in out


# the validator reports the tighter of the containerd store and data-root (rental volumes stay in data-root):
# preflight must fail whichever filesystem is the limiting one, for the total and for the free space
TIGHTER_STORE_CASES = [
    # (check, store df kB, data-root df kB, expected rc, limiting dir suffix)
    ("check_total_disk_for_vram", 436 * 1024**2, 14 * 1024**3, 1, f"/{SNAPSHOTS}"),
    ("check_total_disk_for_vram", 14 * 1024**3, 436 * 1024**2, 1, "/var/lib/docker"),
    ("check_total_disk_for_vram", 14 * 1024**3, 14 * 1024**3, 0, None),
    ("check_free_disk_for_node", 50 * 1024**2, 14 * 1024**3, 1, f"/{SNAPSHOTS}"),
    ("check_free_disk_for_node", 14 * 1024**3, 50 * 1024**2, 1, "/var/lib/docker"),
    ("check_free_disk_for_node", 14 * 1024**3, 14 * 1024**3, 0, None),
]


@pytest.mark.parametrize("check, store_kb, data_root_kb, expected_rc, limiting", TIGHTER_STORE_CASES)
def test_disk_checks_measure_the_tighter_of_the_store_and_data_root(tmp_path, check, store_kb, data_root_kb, expected_rc, limiting):
    env = {**SNAPSHOTTER, "STUB_DF_TOTAL_KB": str(store_kb), "STUB_DF_AVAIL_KB": str(store_kb),
           "STUB_DATA_DF_TOTAL_KB": str(data_root_kb)}
    rc, out, _ = run_check(tmp_path, check, env=env, files={f"{SNAPSHOTS}/.keep": ""})
    assert rc == expected_rc, out
    if limiting:
        assert f"{limiting} is below" in out, out


@pytest.mark.parametrize("store_avail_kb, data_root_avail_kb, limiting", [
    (0, 900 * 1024**2, f"/{SNAPSHOTS}"),
    (900 * 1024**2, 0, "/var/lib/docker"),
])
def test_a_full_filesystem_fails_the_free_disk_check(tmp_path, store_avail_kb, data_root_avail_kb, limiting):
    env = {**SNAPSHOTTER, "STUB_DF_AVAIL_KB": str(store_avail_kb), "STUB_DATA_DF_TOTAL_KB": str(14 * 1024**3),
           "STUB_DATA_DF_AVAIL_KB": str(data_root_avail_kb)}
    rc, out, _ = run_check(tmp_path, "check_free_disk_for_node", env=env, files={f"{SNAPSHOTS}/.keep": ""})
    assert rc == 1, out
    assert "FIX  Free disk 0.0 GB on " in out and f"{limiting} is below" in out, out


CONTENT_STORE = "var/lib/containerd/io.containerd.content.v1.content/.keep"


# pulled layers land in the content store, which may be its own filesystem: free space counts it, the total does not
@pytest.mark.parametrize("check, full, expected_rc, limiting", [
    ("check_free_disk_for_node", "content", 1, "/var/lib/containerd/io.containerd.content.v1.content"),
    ("check_free_disk_for_node", "snapshots", 1, f"/{SNAPSHOTS}"),
    ("check_free_disk_for_node", "data-root", 1, "/var/lib/docker"),
    ("check_free_disk_for_node", None, 0, None),
    ("check_total_disk_for_vram", "content", 0, None),
])
def test_free_disk_counts_whichever_filesystem_is_full(tmp_path, check, full, expected_rc, limiting):
    big = str(900 * 1024**2)
    env = {**SNAPSHOTTER, "STUB_DF_TOTAL_KB": str(14 * 1024**3), "STUB_DATA_DF_TOTAL_KB": str(14 * 1024**3),
           "STUB_DF_AVAIL_KB": "0" if full == "snapshots" else big,
           "STUB_DATA_DF_AVAIL_KB": "0" if full == "data-root" else big,
           "STUB_CONTENT_DF_AVAIL_KB": "0" if full == "content" else big}
    rc, out, _ = run_check(tmp_path, check, env=env, files={f"{SNAPSHOTS}/.keep": "", CONTENT_STORE: ""})
    assert rc == expected_rc, out
    if limiting:
        assert f"{limiting} is below" in out, out


@pytest.mark.parametrize(
    "config_root, running_root",
    [
        ("var/lib/containerd", "var/lib/containerd"),  # unchanged configuration
        # root moved to the big disk in the config, containerd not restarted: it still writes to the root disk
        ("nvme/containerd", "var/lib/containerd"),
        ("nvme/containerd", "nvme/containerd"),  # restarted onto it
    ],
)
def test_free_disk_is_read_where_the_running_containerd_keeps_containers(tmp_path, config_root, running_root):
    root = tmp_path / "root"
    files = {f"{d}/{_OVERLAYFS}/.keep": "" for d in ("var/lib/containerd", "nvme/containerd")}
    files["etc/containerd/config.toml"] = f'version = 2\nroot = "{root}/{config_root}"\n'
    env = {**SNAPSHOTTER, "STUB_CTR_ROOT": f"{root}/{running_root}", "STUB_DATA_DF_TOTAL_KB": str(14 * 1024**3),
           "STUB_DF_AVAIL_KB": str(50 * 1024**2)}
    rc, out, _ = run_check(tmp_path, "check_free_disk_for_node", env=env, files=files)
    assert rc == 1, out
    assert f"FIX  Free disk 50.0 GB on {root}/{running_root}/{_OVERLAYFS} is below" in out
    assert "containerd-snapshotter" in out


@pytest.mark.parametrize(
    "env",
    [
        {"STUB_CTR_FAILS": "1"},  # the running containerd cannot be asked: never its config files instead
        {"STUB_CONTAINERD_ADDRESS": ""},  # Docker names no containerd
        {"STUB_CTR_ROOT": "/gone/containerd"},  # the reported root is not on the host
    ],
)
def test_an_unknown_containerd_root_is_skipped_not_read_as_the_leftover(tmp_path, env):
    files = {f"{SNAPSHOTS}/.keep": ""}
    rc, out, _ = run_check(tmp_path, "check_free_disk_for_node", env={**SNAPSHOTTER, **env, "STUB_DF_AVAIL_KB": str(50 * 1024**2)}, files=files)
    assert rc == 0, out
    assert "SKIP" in out and "is below" not in out


def _containerd_process(*flags: str, pid: int = 4242, mnt: str = "mnt:[1]") -> dict[str, str]:
    # a running containerd serving Docker's socket, as its /proc/<pid>/cmdline reads, in mount namespace mnt
    return {f"proc/{pid}/cmdline": "\0".join(("/usr/bin/containerd", *flags)) + "\0",
            f"proc/{pid}/ns/mnt": Link(mnt), "proc/1/ns/mnt": Link("mnt:[1]")}


_NO_CTR_DISKS = {**SNAPSHOTTER, "STUB_DATA_DF_TOTAL_KB": str(14 * 1024**3), "STUB_DF_TOTAL_KB": str(436 * 1024**2)}


@pytest.mark.parametrize(
    "files",
    [
        # the root moved: the built-in folder is a leftover and the 14 TB data-root says nothing of the store
        {f"{SNAPSHOTS}/.keep": "", "etc/containerd/config.toml": 'root = "/small/containerd"\n'},
        {f"{SNAPSHOTS}/.keep": "", "etc/containerd/config.toml": '  root_path = "/small/snapshots"\n'},
        {f"{SNAPSHOTS}/.keep": "", "etc/containerd/config.toml": 'imports = ["/opt/containerd/*.toml"]\n'},
        {},  # no store at a built-in root
        # no --root: the config file on disk may not be what the daemon loaded (rewritten or deleted since)
        {f"{SNAPSHOTS}/.keep": "", **_containerd_process()},
        {f"{SNAPSHOTS}/.keep": "", "etc/containerd/config.toml": 'version = 2\nroot = "/var/lib/containerd"\n',
         **_containerd_process()},
        {f"{SNAPSHOTS}/.keep": "", "opt/ctrd.toml": 'root = "/var/lib/containerd"\n', **_containerd_process("-c", "/opt/ctrd.toml")},
    ],
)
def test_a_host_without_ctr_whose_store_cannot_be_told_is_skipped_not_passed(tmp_path, files):
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=_NO_CTR_DISKS, files=files, without=("ctr",))
    assert rc == 0, out
    assert "SKIP" in out and "PASS=1" not in out


def test_a_host_without_ctr_is_measured_at_the_running_containerds_root_flag(tmp_path):
    # --root on the command line moves the store; the stale built-in folder must not be measured instead
    files = {f"{SNAPSHOTS}/.keep": "", f"small/containerd/{_OVERLAYFS}/.keep": "",
             **_containerd_process("--root", "/small/containerd")}
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=_NO_CTR_DISKS, files=files, without=("ctr",))
    assert f"/small/containerd/{_OVERLAYFS}" in out, out


def test_a_rentals_containerd_on_the_same_socket_path_is_not_read_as_dockers(tmp_path):
    # a rental's own containerd names a small --root in its own mount namespace; the host's has no flags
    files = {f"{SNAPSHOTS}/.keep": "", f"small/containerd/{_OVERLAYFS}/.keep": "",
             **_containerd_process("--root", "/small/containerd", pid=300, mnt="mnt:[7]"),
             **_containerd_process("--root", "/var/lib/containerd")}
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=_NO_CTR_DISKS, files=files, without=("ctr",))
    assert f"/small/containerd/{_OVERLAYFS}" not in out, out
    assert SNAPSHOTS in out, out


def test_a_host_containerd_whose_config_serves_another_address_is_not_read_as_dockers(tmp_path):
    # Docker's own containerd sets its socket in its config, not on the command line: it is not the one on Docker's
    files = {f"{SNAPSHOTS}/.keep": "", f"small/containerd/{_OVERLAYFS}/.keep": "",
             "run/docker/containerd/containerd.toml": '[grpc]\n  address = "/run/docker/containerd/containerd.sock"\n',
             **_containerd_process("--config", "/run/docker/containerd/containerd.toml", "--root", "/small/containerd",
                                   pid=300),
             **_containerd_process("--root", "/var/lib/containerd")}
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=_NO_CTR_DISKS, files=files, without=("ctr",))
    assert f"/small/containerd/{_OVERLAYFS}" not in out, out
    assert SNAPSHOTS in out, out


@pytest.mark.parametrize(
    "files",
    [
        {f"{SNAPSHOTS}/.keep": "", **_containerd_process("--root=/gone/containerd")},  # its --root holds no store
        # --root names the big disk while the config's root_path puts the snapshots on a small one
        {f"{SNAPSHOTS}/.keep": "", "etc/containerd/config.toml":
         '[plugins."io.containerd.snapshotter.v1.overlayfs"]\n  root_path = "/small/snapshots"\n',
         **_containerd_process("--root", "/var/lib/containerd")},
        # a containerd on Docker's socket whose mount namespace cannot be read
        {f"{SNAPSHOTS}/.keep": "", "proc/4242/cmdline": "/usr/bin/containerd\0"},
        # its --config moves the root while /etc/containerd/config.toml is at the default
        {f"{SNAPSHOTS}/.keep": "", "opt/ctrd.toml": 'root = "/small/containerd"\n',
         **_containerd_process("-c", "/opt/ctrd.toml")},
        # two host containerds on Docker's socket: which one Docker talks to cannot be told
        {f"{SNAPSHOTS}/.keep": "", f"small/containerd/{_OVERLAYFS}/.keep": "",
         **_containerd_process("--root", "/small/containerd", pid=300), **_containerd_process()},
        # no containerd process on Docker's socket at all: the built-in folder may be a leftover
        {f"{SNAPSHOTS}/.keep": ""},
        # the only one on Docker's socket is a rental's, in its own mount namespace
        {f"{SNAPSHOTS}/.keep": "", **_containerd_process(mnt="mnt:[7]")},
        # started with --config, then the file deleted: the daemon still runs on what it loaded
        {f"{SNAPSHOTS}/.keep": "", **_containerd_process("--config=/etc/containerd/custom.toml")},
    ],
)
def test_a_host_without_ctr_reads_the_running_containerds_flags_or_skips(tmp_path, files):
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=_NO_CTR_DISKS, files=files, without=("ctr",))
    assert rc == 0, out
    assert "SKIP" in out and "PASS=1" not in out


def test_a_containerd_whose_address_may_be_imported_is_skipped(tmp_path):
    # Docker's containerd sets its custom address in an imported file: which daemon serves it cannot be told
    files = {f"{SNAPSHOTS}/.keep": "", "etc/containerd/custom.toml": 'imports = ["/etc/containerd/conf.d/*.toml"]\n',
             **_containerd_process("--config", "/etc/containerd/custom.toml", "--root", "/small/containerd")}
    env = {**_NO_CTR_DISKS, "STUB_CONTAINERD_ADDRESS": "/run/custom/containerd.sock"}
    rc, out, _ = run_check(tmp_path, "check_total_disk_for_vram", env=env, files=files, without=("ctr",))
    assert rc == 0, out
    assert "SKIP" in out and "PASS=1" not in out
    assert rc == 0, out
    assert "SKIP" in out and "PASS=1" not in out


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


# ── data-root on a loop image (DAH-5150) ─────────────────────────────────────


def test_data_root_on_a_buffered_loop_gets_the_direct_io_fix(tmp_path):
    env = {"STUB_DF_SOURCE": "/dev/loop7"}
    rc, out, _ = run_check(tmp_path, "check_data_root_loop_dio", env=env, files={"sys/block/loop7/loop/dio": "0\n"})
    assert rc == 1
    assert "is on the loop device /dev/loop7 without direct I/O" in out
    assert "With no rental running: sudo losetup --direct-io=on /dev/loop7" in out


def test_data_root_on_a_direct_io_loop_passes(tmp_path):
    env = {"STUB_DF_SOURCE": "/dev/loop7"}
    rc, out, _ = run_check(tmp_path, "check_data_root_loop_dio", env=env, files={"sys/block/loop7/loop/dio": "1\n"})
    assert rc == 0
    assert "on /dev/loop7 with direct I/O." in out


def test_data_root_on_a_real_disk_is_not_reported(tmp_path):
    rc, out, fixes = run_check(tmp_path, "check_data_root_loop_dio")
    assert (rc, fixes) == (0, 0)
    assert "loop" not in out


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

APPARMOR_RULE = "mount fstype=@{fuse_types} -> /var/lib/sysboxfs/**,"
APPARMOR_ON = {"sys/module/apparmor/parameters/enabled": "Y\n", "etc/apparmor.d/fusermount3": "profile fusermount3 {}\n"}


def test_ubuntu_2604_fusermount3_profile_without_the_sysboxfs_rule_is_a_fix(tmp_path):
    rc, out, fixes = run_check(
        tmp_path, "check_fusermount3_apparmor", files=APPARMOR_ON
    )
    assert (rc, fixes) == (1, 1)
    assert "FIX  The AppArmor profile /etc/apparmor.d/fusermount3 (Ubuntu 26.04)" in out
    assert f"printf '%s\\n' '{APPARMOR_RULE}'" in out


def test_install_reloads_the_profile_even_when_an_earlier_run_already_wrote_the_rule(tmp_path):
    # an earlier run may have appended the rule and then failed to reload: a rerun must retry the reload
    files = {**APPARMOR_ON, "etc/apparmor.d/local/fusermount3": APPARMOR_RULE + "\n"}
    rc, _, _ = run_check(tmp_path, "allow_sysboxfs_in_fusermount3_apparmor", files=files)
    assert rc == 0
    assert (tmp_path / "root/.stub-docker/apparmor").read_text() == "-r /etc/apparmor.d/fusermount3\n"
    assert (tmp_path / "root/etc/apparmor.d/local/fusermount3").read_text().count(APPARMOR_RULE) == 1


def test_install_with_apparmor_off_skips_the_reload(tmp_path):
    # apparmor_parser fails with AppArmor off (apparmor=0); the install must not stop on it
    files = {**APPARMOR_ON, "sys/module/apparmor/parameters/enabled": "N\n"}
    rc, _, _ = run_check(tmp_path, "allow_sysboxfs_in_fusermount3_apparmor", files=files)
    assert rc == 0
    assert not (tmp_path / "root/.stub-docker/apparmor").exists()


@pytest.mark.parametrize(
    "files",
    [
        {},
        {**APPARMOR_ON, "etc/apparmor.d/local/fusermount3": APPARMOR_RULE + "\n"},
        {**APPARMOR_ON, "sys/module/apparmor/parameters/enabled": "N\n"},
    ],
    ids=["no-profile", "rule-in-local", "apparmor-off"],
)
def test_fusermount3_without_a_profile_or_with_the_rule_passes(tmp_path, files):
    rc, out, _ = run_check(tmp_path, "check_fusermount3_apparmor", files=files)
    assert rc == 0
    assert "PASS fusermount3 may mount sysbox-fs" in out


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


@pytest.mark.parametrize(
    ("error", "hinted"),
    [
        ('docker: Error response from daemon: OCI runtime create failed: namespace {"time" ""} does not exist', True),
        ("nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open", False),
    ],
)
def test_the_docker_28_hint_shows_only_for_the_time_namespace_error(tmp_path, error, hinted):
    _, out, _ = run_check(
        tmp_path, 'time_namespace_hint "$VERIFY_OUT"', env={"STUB_DOCKER_VERSION": "29.9.2", "VERIFY_OUT": error}
    )
    assert ("downgrade the daemon to Docker 28.x" in out) is hinted


# ── failure diagnostics (the block printed when the verify container does not start) ─────────




def test_aarch64_host_passes_the_arch_check(tmp_path):
    # GB300 / Grace: nestybox ships an arm64 .deb, so aarch64 is no longer a blocking FIX
    proc = run_script(tmp_path, "--check", env={"STUB_ARCH": "aarch64"})
    assert "PASS Architecture aarch64." in proc.stdout
