"""DAH-3319: the preflight in nvidia_docker_sysbox_setup.sh — one PASS / FIX line per host
requirement with the command that fixes it, run before anything is installed and standalone as
`--check` (ticket-0309: a provider reinstalled sysbox three times; the cause was a host setting
nobody had checked).

Same harness as test_sysbox_setup_apt_repo.py: the script is run under bash with stub commands
on PATH and the files it reads (/proc/modules, /etc/os-release, /etc/docker/daemon.json, ...)
under a fixture HOST_ROOT. The check functions are sourced with SYSBOX_SETUP_LIB=1.
"""

import os
import re
import shutil
import subprocess
import sys
import textwrap

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
    "docker": textwrap.dedent(
        """\
        #!/bin/bash
        [ -z "${STUB_NO_DOCKER_DAEMON:-}" ] || { echo "Cannot connect to the Docker daemon" >&2; exit 1; }
        case "$1 $2" in
            "version --format") echo "${STUB_DOCKER_VERSION:-28.5.2}" ;;
            "ps ") exit 0 ;;
            "ps --filter") echo "${STUB_PORT_CONTAINER:-executor-1}" ;;
            "info --format")
                case "$3" in
                    *DockerRootDir*) echo "${STUB_DOCKER_ROOT:-$HOST_ROOT/var/lib/docker}" ;;
                    *Runtimes*) echo "${STUB_DOCKER_RUNTIMES:-\\"runc\\", \\"sysbox-runc\\"}" ;;
                esac ;;
            "run --rm") [ -z "${STUB_SYSBOX_RUN_FAILS:-}" ] && echo ok || { echo "OCI runtime create failed" >&2; exit 125; } ;;
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
    "nvidia-container-cli": '#!/bin/bash\nprintf "cli-version: 1.17.8\\nlib-version: 1.17.8\\n"\n',
    "sysbox-runc": '#!/bin/bash\necho "sysbox-runc\\n\\tversion:\\t0.6.6"\n',
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


def _bin_dir(tmp_path, *, with_jq: bool, without: tuple[str, ...] = ()):
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


def _host_root(tmp_path, files=None):
    root = tmp_path / "root"
    for rel, content in {**GOOD_FILES, **(files or {})}.items():
        if content is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def _script_copy(tmp_path):
    """The script in its own directory, so the `.env` it may read next to itself is the test's, never the checkout's."""
    copy_dir = tmp_path / "executor"
    copy_dir.mkdir(exist_ok=True)
    copy = copy_dir / "nvidia_docker_sysbox_setup.sh"
    shutil.copy(SCRIPT, copy)
    return copy


def run_check(tmp_path, function, *, env=None, files=None, with_jq=False, without=()):
    """Source the installer's functions and run one check; returns (rc, output, fix_count)."""
    stubs = _bin_dir(tmp_path, with_jq=with_jq, without=without)
    root = _host_root(tmp_path, files)
    program = (
        f"SYSBOX_SETUP_LIB=1 . {SCRIPT}\nset +e\n{function}\nrc=$?\necho \"RC=$rc FIX=$PREFLIGHT_FIX PASS=$PREFLIGHT_PASS SKIP=$PREFLIGHT_SKIP\"\n"
    )
    proc = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": str(stubs), "HOST_ROOT": str(root), **(env or {})},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = ANSI.sub("", proc.stdout)
    counts = dict(kv.split("=") for kv in out.strip().splitlines()[-1].split())
    return int(counts["RC"]), out, int(counts["FIX"])


def run_script(tmp_path, *args, env=None, files=None, without=()):
    """The installer itself, as a provider runs it, with the same stubs and fixture tree; stdout without colour."""
    stubs = _bin_dir(tmp_path, with_jq=False, without=without)
    root = _host_root(tmp_path, files)
    proc = subprocess.run(
        ["bash", str(_script_copy(tmp_path)), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
        env={"PATH": str(stubs), "HOST_ROOT": str(root), **(env or {})},
    )
    proc.stdout = ANSI.sub("", proc.stdout)
    return proc


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash installer")


# ── kernel ───────────────────────────────────────────────────────────────────


def test_kernel_6_8_passes(tmp_path):
    rc, out, fix = run_check(tmp_path, "check_kernel")
    assert rc == 0 and fix == 0
    assert "PASS Kernel 6.8.0-45-generic" in out


def test_kernel_5_15_on_22_04_names_the_hwe_kernel(tmp_path):
    rc, out, fix = run_check(tmp_path, "check_kernel", env={"STUB_KERNEL": "5.15.0-91-generic"})
    assert rc == 1 and fix == 1
    assert "FIX  Kernel 5.15.0-91-generic is below 5.19" in out
    assert "sudo apt-get install -y linux-generic-hwe-22.04 && sudo reboot" in out
    # the override goes after sudo: sudo's env_reset drops a variable set in front of it (sourced here, so the curl form)
    assert "backport: curl -fsSL https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo SYSBOX_SKIP_KERNEL_CHECK=1 bash" in out


def test_kernel_5_15_on_20_04_says_upgrade_the_distro(tmp_path):
    rc, out, _ = run_check(
        tmp_path, "check_kernel", env={"STUB_KERNEL": "5.15.0-91-generic"}, files={"etc/os-release": 'VERSION_ID="20.04"\n'}
    )
    assert rc == 1
    assert "Ubuntu 20.04 tops out at 5.15" in out
    assert "do-release-upgrade" in out
    assert "hwe-22.04" not in out


def test_kernel_override_is_honoured(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_kernel", env={"STUB_KERNEL": "5.15.0-91-generic", "SYSBOX_SKIP_KERNEL_CHECK": "1"})
    assert rc == 0
    assert "PASS Kernel 5.15.0-91-generic accepted because SYSBOX_SKIP_KERNEL_CHECK=1" in out


# ── docker ───────────────────────────────────────────────────────────────────


def test_docker_28_passes_with_its_version(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker")
    assert rc == 0
    assert "PASS Docker 28.5.2." in out


def test_docker_missing_names_the_install_command(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker", without=("docker",))
    assert rc == 1
    assert "FIX  Docker is not installed." in out
    assert "curl -fsSL https://get.docker.com | sudo sh" in out


def test_docker_daemon_down_names_systemctl(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker", env={"STUB_NO_DOCKER_DAEMON": "1"})
    assert rc == 1
    assert "FIX  Docker daemon is not running" in out
    assert "sudo systemctl enable --now docker" in out


def test_docker_29_1_passes_but_says_untested(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker", env={"STUB_DOCKER_VERSION": "29.1.0"})
    assert rc == 0
    assert "PASS Docker 29.1.0 (29.0–29.1 is untested with sysbox" in out


@pytest.mark.parametrize("with_jq", [False, True])
def test_docker_29_7_without_the_two_settings_names_both(tmp_path, with_jq):
    # a daemon.json that exists but carries neither key, so the jq path is the one that decides
    rc, out, _ = run_check(
        tmp_path,
        "check_docker_features",
        env={"STUB_DOCKER_VERSION": "29.7.0"},
        files={"etc/docker/daemon.json": '{"runtimes": {"sysbox-runc": {"path": "/usr/bin/sysbox-runc"}}}\n'},
        with_jq=with_jq,
    )
    assert rc == 1
    assert "FIX  Docker 29.7.0 without features.cdi, time-namespaces = false" in out
    assert 'add {"features":{"cdi":false,"time-namespaces":false}} to /etc/docker/daemon.json' in out


@pytest.mark.parametrize("with_jq", [False, True])
def test_docker_29_7_with_both_settings_passes(tmp_path, with_jq):
    rc, out, _ = run_check(
        tmp_path,
        "check_docker_features",
        env={"STUB_DOCKER_VERSION": "29.7.0"},
        files={"etc/docker/daemon.json": '{"runtimes": {}, "features": {"cdi": false, "time-namespaces": false}}\n'},
        with_jq=with_jq,
    )
    assert rc == 0, out
    assert "PASS Docker 29.7.0 has the sysbox settings" in out
    assert "features.cdi and features.time-namespaces = false" in out


def test_docker_29_3_needs_only_cdi(tmp_path):
    rc, out, _ = run_check(
        tmp_path,
        "check_docker_features",
        env={"STUB_DOCKER_VERSION": "29.3.0"},
        files={"etc/docker/daemon.json": '{"features": {"cdi": false}}\n'},
    )
    assert rc == 0, out
    assert "time-namespaces" not in out.split("PASS")[1]


@pytest.mark.parametrize("with_jq", [False, True])
def test_docker_29_3_with_cdi_still_on_is_a_fix(tmp_path, with_jq):
    # a daemon.json that mentions cdi but leaves it true is the negative control for both read paths
    rc, out, _ = run_check(
        tmp_path,
        "check_docker_features",
        env={"STUB_DOCKER_VERSION": "29.3.0"},
        files={"etc/docker/daemon.json": '{"features": {"cdi": true}}\n'},
        with_jq=with_jq,
    )
    assert rc == 1
    assert "FIX  Docker 29.3.0 without features.cdi = false" in out
    # below 29.5 the by-hand block names cdi only, like the install step
    assert 'add {"features":{"cdi":false}} to /etc/docker/daemon.json' in out


def test_docker_28_needs_no_features(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_docker_features")
    assert rc == 0
    assert "PASS Docker 28.5.2 needs no daemon.json features" in out


# ── nvidia ───────────────────────────────────────────────────────────────────


def test_nvidia_driver_580_with_8_gpus_passes(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver")
    assert rc == 0
    assert "PASS NVIDIA driver 580.65.06 (8 GPU(s))." in out


def test_nvidia_smi_missing_is_a_fix(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver", without=("nvidia-smi",))
    assert rc == 1
    assert "FIX  nvidia-smi not found" in out
    assert "sudo apt-get install -y nvidia-driver-580-server && sudo reboot" in out


def test_nvidia_driver_not_loaded_is_a_fix(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver", files={"proc/driver/nvidia/version": None})
    assert rc == 1
    assert "FIX  NVIDIA driver is installed but not loaded" in out and "/proc/driver/nvidia missing" in out


def test_nvidia_smi_error_text_is_not_read_as_a_version(tmp_path):
    # nvidia-smi prints this on stdout; it must land in the "not loaded" line with the reboot fix, not in "below 580.65.06"
    rc, out, _ = run_check(
        tmp_path, "check_nvidia_driver", env={"STUB_NV_DRIVER": "Failed to initialize NVML: Driver/library version mismatch"}
    )
    assert rc == 1
    assert "FIX  NVIDIA driver is installed but not loaded (nvidia-smi: Failed to initialize NVML" in out
    assert "sudo reboot" in out and "below 580.65.06" not in out


@pytest.mark.parametrize("driver", ["575.57.08", "580.65.05", "580.64.99"])
def test_nvidia_driver_below_validator_minimum_is_a_fix(tmp_path, driver):
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver", env={"STUB_NV_DRIVER": driver})
    assert rc == 1
    assert f"FIX  NVIDIA driver {driver} is below 580.65.06" in out


@pytest.mark.parametrize("driver", ["580.65.06", "580.65.10", "581.0.0"])
def test_nvidia_driver_at_or_above_the_minimum_passes(tmp_path, driver):
    rc, out, _ = run_check(tmp_path, "check_nvidia_driver", env={"STUB_NV_DRIVER": driver})
    assert rc == 0, out


def test_nvidia_toolkit_present_and_missing(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_nvidia_toolkit")
    assert rc == 0 and "PASS NVIDIA container toolkit (nvidia-container-cli 1.17.8)." in out
    rc, out, _ = run_check(tmp_path, "check_nvidia_toolkit", without=("nvidia-container-cli",))
    assert rc == 1 and "FIX  NVIDIA container toolkit is not installed" in out


# ── iptables modules (ticket-0309) ───────────────────────────────────────────


def test_iptables_modules_loaded_pass(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_iptables_modules")
    assert rc == 0
    assert "PASS Legacy iptables modules loaded" in out


def test_iptables_modules_missing_name_modprobe_and_the_persistent_file(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_iptables_modules", files={"proc/modules": "nf_tables 311296 0\n"})
    assert rc == 1
    assert "FIX  Kernel modules ip_tables iptable_nat iptable_filter are not loaded" in out
    assert "sudo modprobe -a ip_tables iptable_nat iptable_filter" in out
    assert "/etc/modules-load.d/lium-iptables.conf" in out


def test_iptables_built_in_modules_count_as_loaded(tmp_path):
    files = {"proc/modules": "nf_tables 311296 0\n"}
    for mod in ("ip_tables", "iptable_nat", "iptable_filter"):
        files[f"sys/module/{mod}/.keep"] = ""
    rc, out, _ = run_check(tmp_path, "check_iptables_modules", files=files)
    assert rc == 0, out


# ── disk >= 1.5x VRAM (validators' MIN_DISK_TO_VRAM_RATE) ────────────────────


def test_disk_rule_uses_total_size_of_dockers_filesystem(tmp_path):
    # 8 x 81559 MiB = 637.2 GB VRAM -> needs 955.8 GB; the fixture filesystem is 1000.0 GB
    rc, out, _ = run_check(tmp_path, "check_disk_for_vram")
    assert rc == 0
    assert "PASS Disk 1000.0 GB on " in out
    assert ">= 955.8 GB (1.5x of 637.2 GB VRAM)" in out


def test_disk_below_the_rule_is_a_fix_with_the_numbers(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_disk_for_vram", env={"STUB_DF_TOTAL_KB": str(500 * 1024 * 1024)})
    assert rc == 1
    assert "FIX  Disk 500.0 GB on " in out
    assert "is below 955.8 GB (1.5x of 637.2 GB VRAM)" in out
    assert "earns nothing while idle" in out


def test_disk_rule_compares_like_the_validator_at_the_boundary(tmp_path):
    # 1x 81613 MiB = 79.7 GB VRAM -> 119.55 GB needed. The validator compares 79.7 * 1.5 unrounded with the rounded
    # disk, so 119.5 GB fails; a compare against the displayed (rounded) 119.5 would have passed it.
    env = {"STUB_NV_GPUS": "1", "STUB_NV_MEM_MIB": "81613"}
    rc, out, _ = run_check(tmp_path, "check_disk_for_vram", env={**env, "STUB_DF_TOTAL_KB": str(int(119.5 * 1024 * 1024))})
    assert rc == 1, out
    assert "FIX  Disk 119.5 GB" in out and "(1.5x of 79.7 GB VRAM)" in out
    rc, out, _ = run_check(tmp_path, "check_disk_for_vram", env={**env, "STUB_DF_TOTAL_KB": str(int(119.6 * 1024 * 1024))})
    assert rc == 0, out


def test_disk_rule_is_skipped_without_a_gpu(tmp_path):
    rc, out, fix = run_check(tmp_path, "check_disk_for_vram", without=("nvidia-smi",))
    assert rc == 0 and fix == 0
    assert "SKIP Disk >= 1.5x VRAM — no NVIDIA driver" in out


# ── ports ────────────────────────────────────────────────────────────────────


def test_ports_free_and_no_ufw_pass_with_the_outside_probe_hint(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_ports")
    assert rc == 0
    assert "PASS TCP 8080 (executor port) is free" in out
    assert "PASS TCP 2200 (SSH port) is free" in out
    assert "nc -vz <this host's public IP> 8080 2200" in out


def test_port_held_by_another_process_is_a_fix(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_ports", env={"STUB_BUSY_PORT": "8080", "STUB_BUSY_PROC": "nginx"})
    assert rc == 1
    assert "FIX  TCP 8080 (executor port) is already in use by " in out and "nginx" in out
    assert "PASS TCP 2200 (SSH port)" in out


def test_port_served_by_the_executor_container_passes(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_ports", env={"STUB_BUSY_PORT": "2200", "STUB_BUSY_PROC": "docker-proxy"})
    assert rc == 0
    assert "PASS TCP 2200 (SSH port) is served by the executor (executor-1)" in out


def test_port_published_by_another_container_is_a_fix(tmp_path):
    # a docker-proxy listener is not the executor's by definition: ask Docker whose it is
    rc, out, _ = run_check(
        tmp_path,
        "check_ports",
        env={"STUB_BUSY_PORT": "8080", "STUB_BUSY_PROC": "docker-proxy", "STUB_PORT_CONTAINER": "nginx-proxy"},
    )
    assert rc == 1
    assert "FIX  TCP 8080 (executor port) is published by container nginx-proxy" in out
    assert "docker stop nginx-proxy" in out


def test_ports_are_skipped_without_ss(tmp_path):
    rc, out, fix = run_check(tmp_path, "check_ports", without=("ss",))
    assert rc == 0 and fix == 0
    assert "SKIP Ports — 'ss' (iproute2) is missing" in out


def test_ufw_active_without_a_rule_names_ufw_allow(tmp_path):
    ufw = "Status: active\\n\\nTo   Action   From\\n--   ------   ----\\n22/tcp   ALLOW   Anywhere\\n8080/tcp   ALLOW   Anywhere"
    rc, out, _ = run_check(tmp_path, "check_ports", env={"STUB_UFW_STATUS": ufw})
    assert rc == 1
    assert "PASS TCP 8080 (executor port)" in out
    assert "FIX  TCP 2200 (SSH port) is blocked by ufw" in out
    assert "sudo ufw allow 2200/tcp" in out


def test_ufw_range_rule_covers_the_port(tmp_path):
    ufw = "Status: active\\n2000:9000/tcp   ALLOW   Anywhere"
    rc, out, _ = run_check(tmp_path, "check_ports", env={"STUB_UFW_STATUS": ufw})
    assert rc == 0, out


def test_ufw_verbose_and_interface_rows_are_read(tmp_path):
    # `ufw status verbose` prints "ALLOW IN"; an interface rule prints "8080/tcp on eth0"; v6 rows carry "(v6)"
    ufw = "Status: active\\n8080/tcp on eth0   ALLOW IN   Anywhere\\n2200/tcp (v6)   ALLOW IN   Anywhere (v6)\\n2200   ALLOW IN   Anywhere"
    rc, out, _ = run_check(tmp_path, "check_ports", env={"STUB_UFW_STATUS": ufw})
    assert rc == 0, out


def test_ports_come_from_env_then_the_env_file(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_ports", env={"EXECUTOR_PORT": "8001", "SSH_PORT": "2201"})
    assert rc == 0
    assert "TCP 8001 (executor port)" in out and "TCP 2201 (SSH port)" in out


def test_env_file_next_to_the_script_is_read(tmp_path):
    copy = _script_copy(tmp_path)
    (copy.parent / ".env").write_text("INTERNAL_PORT=8001\nEXTERNAL_PORT=8123 # external\nSSH_PORT=2299\n")
    proc = run_script(tmp_path, "--check")
    assert "TCP 8123 (executor port)" in proc.stdout and "TCP 2299 (SSH port)" in proc.stdout


def test_env_file_in_the_working_directory_is_ignored_when_piped_from_curl(tmp_path):
    # piped from curl $0 is `bash`; a `.env` in the provider's cwd must not steer the port check
    (tmp_path / ".env").write_text("EXTERNAL_PORT=8123\nSSH_PORT=22\n")
    stubs = _bin_dir(tmp_path, with_jq=False)
    root = _host_root(tmp_path)
    with open(SCRIPT) as fh:
        script = fh.read()
    proc = subprocess.run(
        ["bash", "-s", "--", "--check"],
        input=script,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": str(stubs), "HOST_ROOT": str(root)},
    )
    out = ANSI.sub("", proc.stdout)
    assert "TCP 8080 (executor port)" in out and "TCP 2200 (SSH port)" in out
    assert "TCP 22 " not in out


# ── sysbox probe ─────────────────────────────────────────────────────────────


def test_sysbox_installed_registered_and_running_passes(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_sysbox")
    assert rc == 0
    assert "PASS sysbox-runc 0.6.6 runs a container." in out


def test_sysbox_missing_points_at_the_installer(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_sysbox", without=("sysbox-runc",))
    assert rc == 1
    assert "FIX  sysbox-runc is not installed" in out


def test_sysbox_not_registered_in_docker_is_a_fix(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_sysbox", env={"STUB_DOCKER_RUNTIMES": '{"runc":{}}'})
    assert rc == 1
    assert "FIX  sysbox-runc is installed but not registered" in out


def test_sysbox_container_start_failure_is_a_fix(tmp_path):
    rc, out, _ = run_check(tmp_path, "check_sysbox", env={"STUB_SYSBOX_RUN_FAILS": "1"})
    assert rc == 1
    assert "docker run --rm --runtime=sysbox-runc alpine echo ok' fails" in out


# ── the script as a provider runs it ─────────────────────────────────────────


def test_check_mode_on_a_good_host_exits_zero_with_a_summary(tmp_path):
    proc = run_script(tmp_path, "--check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FIX  " not in proc.stdout
    assert "Preflight: 12 PASS, 0 FIX, 0 SKIP." in proc.stdout


def test_check_mode_without_a_gpu_reports_the_nvidia_fixes_and_exits_one(tmp_path):
    # the EC2 shape: kernel, docker, disk and ports fine, no NVIDIA driver
    proc = run_script(tmp_path, "--check", without=("nvidia-smi", "nvidia-container-cli"))
    assert proc.returncode == 1
    assert "FIX  nvidia-smi not found" in proc.stdout
    assert "FIX  NVIDIA container toolkit is not installed" in proc.stdout
    assert "SKIP Disk >= 1.5x VRAM" in proc.stdout
    assert "PASS Kernel 6.8.0-45-generic" in proc.stdout
    assert "Preflight: 9 PASS, 2 FIX, 1 SKIP." in proc.stdout
    assert f"Fix the lines above, then run: sudo bash {tmp_path / 'executor' / 'nvidia_docker_sysbox_setup.sh'}" in proc.stdout


def test_install_mode_stops_before_installing_on_a_fix(tmp_path):
    proc = run_script(tmp_path, env={"STUB_KERNEL": "5.15.0-91-generic"})
    assert proc.returncode == 1
    assert "FIX  Kernel 5.15.0-91-generic is below 5.19" in proc.stdout
    assert "Nothing was installed." in proc.stdout
    # step 2 is the first thing after the preflight; on a good host the same stubs reach it (next test)
    assert "Checking running containers" not in proc.stdout


def test_install_mode_on_a_good_host_reaches_the_install_steps(tmp_path):
    # the negative control for the test above: same stubs, kernel fine -> the preflight passes and the script goes on
    proc = run_script(tmp_path)
    assert "Preflight: 9 PASS, 0 FIX, 0 SKIP." in proc.stdout
    assert "Nothing was installed." not in proc.stdout
    assert "Checking running containers" in proc.stdout or "Sysbox is already working" in proc.stdout


def test_not_root_is_a_fix_that_names_sudo(tmp_path):
    proc = run_script(tmp_path, "--check", env={"STUB_UID": "1000"})
    assert proc.returncode == 1
    assert "FIX  Not running as root" in proc.stdout
    assert f"sudo bash {tmp_path / 'executor' / 'nvidia_docker_sysbox_setup.sh'} --check" in proc.stdout
    # nothing else runs without root: the other checks would read the process table and Docker's socket as a user
    assert "Preflight: 0 PASS, 1 FIX, 0 SKIP." in proc.stdout


def test_unknown_option_is_refused(tmp_path):
    proc = run_script(tmp_path, "--bogus")
    assert proc.returncode == 2
    assert "Unknown option: --bogus" in proc.stdout


def test_fix_lines_name_the_curl_one_liner_when_piped_from_curl(tmp_path):
    # `curl … | sudo bash -s -- --check` has no script file behind $0
    stubs = _bin_dir(tmp_path, with_jq=False, without=("sysbox-runc",))
    root = _host_root(tmp_path)
    with open(SCRIPT) as fh:
        script = fh.read()
    proc = subprocess.run(
        ["bash", "-s", "--", "--check"],
        input=script,
        capture_output=True,
        text=True,
        env={"PATH": str(stubs), "HOST_ROOT": str(root)},
    )
    out = ANSI.sub("", proc.stdout)
    assert proc.returncode == 1
    assert "FIX  sysbox-runc is not installed" in out
    assert "curl -fsSL https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo bash" in out
    assert "sudo bash bash" not in out


def test_help_lists_check_and_exits_zero(tmp_path):
    # exit 0 is also the regression test for the EXIT trap that used to turn every clean exit into status 1
    proc = run_script(tmp_path, "--help")
    assert proc.returncode == 0
    assert "--check" in proc.stdout
