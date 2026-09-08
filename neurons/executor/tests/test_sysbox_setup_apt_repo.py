"""DAH-2768: nvidia_docker_sysbox_setup.sh apt-installed nvidia-container-toolkit without
adding NVIDIA's apt repository and sent apt's output to /dev/null, so on a host whose drivers
did not come from that repository the installer stopped after "Installing packages" with
nothing on screen (ticket-0286: a provider read the script and added the repo himself).

The installer's package step is run here under bash with stub apt-get/curl/gpg on PATH.
"""

import os
import re
import subprocess
import sys
import textwrap

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "nvidia_docker_sysbox_setup.sh")

# A host without the NVIDIA repository: apt only knows nvidia-container-toolkit once the list
# file the installer is supposed to write exists.
APT_GET_STUB = textwrap.dedent(
    """\
    #!/bin/bash
    echo "apt-get $*" >> "$APT_ROOT/apt.log"
    if [ "$1" = install ] && [[ " $* " == *" nvidia-container-toolkit "* ]] \\
       && ! grep -rqs "nvidia.github.io/libnvidia-container" "$APT_ROOT/etc/apt/sources.list.d/"; then
        echo "E: Unable to locate package nvidia-container-toolkit" >&2
        exit 100
    fi
    exit 0
    """
)
CURL_STUB = textwrap.dedent(
    """\
    #!/bin/bash
    [ -z "${CURL_FAIL:-}" ] || { echo "curl: (6) Could not resolve host" >&2; exit 6; }
    out=/dev/stdout; url=""
    while [ $# -gt 0 ]; do
        case "$1" in -o) out="$2"; shift ;; http*) url="$1" ;; esac
        shift
    done
    case "$url" in
        *gpgkey) echo "FAKE-PGP-KEY" > "$out" ;;
        *nvidia-container-toolkit.list) echo "deb https://nvidia.github.io/libnvidia-container/stable/deb/\\$(ARCH) /" > "$out" ;;
        *) exit 22 ;;
    esac
    """
)
GPG_STUB = textwrap.dedent(
    """\
    #!/bin/bash
    out=""; src=""
    while [ $# -gt 0 ]; do
        case "$1" in -o) out="$2"; shift ;; --dearmor|--yes) ;; *) src="$1" ;; esac
        shift
    done
    [ -n "$out" ] && [ -s "$src" ] && cat "$src" > "$out"
    """
)


def _functions(*names: str) -> str:
    """The named shell functions, verbatim, from the installer."""
    with open(SCRIPT) as fh:
        text = fh.read()
    out = []
    for name in names:
        # one-line helpers (`ok() { ...; }`) or a block closed by a `}` line of its own
        match = re.search(rf"^{name}\(\)\s*\{{[^\n]*\}}\s*$", text, re.M) or re.search(
            rf"^{name}\(\)\s*\{{\n.*?^\}}", text, re.S | re.M
        )
        assert match, f"{name}() is not defined in nvidia_docker_sysbox_setup.sh"
        out.append(match.group(0))
    return "\n".join(out)


def _run_package_step(tmp_path, env=None):
    root = tmp_path / "root"
    (root / "etc/apt/sources.list.d").mkdir(parents=True, exist_ok=True)
    (root / "usr/share/keyrings").mkdir(parents=True, exist_ok=True)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in ("apt-get", APT_GET_STUB), ("curl", CURL_STUB), ("gpg", GPG_STUB):
        path = stubs / name
        path.write_text(body)
        path.chmod(0o755)
    program = (
        "set -e\n"
        + _functions("ok", "warn", "fail", "apt_install", "ensure_nvidia_container_toolkit_repo")
        + "\nensure_nvidia_container_toolkit_repo || exit 1\n"
        "apt_install update -qq || exit 1\n"
        "apt_install install -y -qq nvidia-container-toolkit jq || exit 1\n"
        'echo INSTALLED\n'
    )
    proc = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{stubs}:{os.environ['PATH']}",
            "APT_ROOT": str(root),
            **(env or {}),
        },
    )
    return proc, root


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_repo_is_added_before_the_toolkit_is_installed(tmp_path):
    proc, root = _run_package_step(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "INSTALLED" in proc.stdout
    list_file = root / "etc/apt/sources.list.d/nvidia-container-toolkit.list"
    keyring = root / "usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    assert keyring.read_text().strip() == "FAKE-PGP-KEY"
    assert list_file.read_text().startswith(f"deb [signed-by={keyring}] https://nvidia.github.io/libnvidia-container/")
    apt_log = (root / "apt.log").read_text().splitlines()
    assert apt_log == ["apt-get update -qq", "apt-get install -y -qq nvidia-container-toolkit jq"]


def test_rerun_on_a_host_that_has_the_repo_is_a_no_op(tmp_path):
    root = tmp_path / "root"
    (root / "etc/apt/sources.list.d").mkdir(parents=True)
    # the provider added it under his own file name (ticket-0286 did exactly this)
    (root / "etc/apt/sources.list.d/libnvidia-container.list").write_text(
        "deb https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /\n"
    )
    proc, root = _run_package_step(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "already configured" in proc.stdout
    assert not (root / "etc/apt/sources.list.d/nvidia-container-toolkit.list").exists()


def test_key_download_failure_is_reported_not_silent(tmp_path):
    proc, root = _run_package_step(tmp_path, env={"CURL_FAIL": "1"})
    assert proc.returncode == 1
    assert "Could not fetch the NVIDIA container toolkit signing key" in proc.stdout
    assert "INSTALLED" not in proc.stdout
    assert not (root / "apt.log").exists(), "apt must not run without the repository"


def test_apt_failure_shows_apts_own_error(tmp_path):
    # the pre-fix behaviour: apt fails, and the operator must see why
    root = tmp_path / "root"
    (root / "etc/apt/sources.list.d").mkdir(parents=True)
    (root / "etc/apt/sources.list.d/broken.list").write_text(
        "deb https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /\n"
    )
    stubs = tmp_path / "bin"
    stubs.mkdir()
    apt = stubs / "apt-get"
    apt.write_text("#!/bin/bash\necho 'E: The repository is not signed.' >&2\nexit 100\n")
    apt.chmod(0o755)
    program = "set -e\n" + _functions("ok", "fail", "apt_install") + "\napt_install update -qq || exit 1\necho INSTALLED\n"
    proc = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        env={"PATH": f"{stubs}:{os.environ['PATH']}", "APT_ROOT": str(root)},
    )
    assert proc.returncode == 1
    assert "apt-get update -qq failed" in proc.stdout
    assert "E: The repository is not signed." in proc.stdout
    assert "INSTALLED" not in proc.stdout
