"""Exercise bootstrap functions with isolated PATH doubles, without system installs."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "run_jupyter.sh"


@pytest.mark.asyncio
@pytest.mark.parametrize("manager", ["apt", "apk", "dnf", "yum", "pacman", "zypper"])
@pytest.mark.parametrize("support", ["no-python", "no-venv", "working", "install-fails"])
async def test_bootstrap_installs_only_missing_python_support(tmp_path: Path, manager: str, support: str) -> None:
    # Arrange: only our doubles are on PATH; installation supplies Python and venv support.
    def command(name: str, body: str) -> None:
        path = tmp_path / name
        path.write_text("#!/bin/sh\nset -e\n" + body)
        path.chmod(0o755)

    command("python-stub", """
case "$1" in
    --version) exit 0 ;;
    -c) [ -f "$READY" ] ;;
    -m) echo venv >> "$LOG"; [ -f "$READY" ] ;;
esac
""")
    package_command = "apt-get" if manager == "apt" else manager
    command(package_command, """
echo "packages $*" >> "$LOG"
[ "$FAIL" = 0 ] || exit 42
case "$1" in
    install|add|-Syu)
        case " $* " in *" $REQUIRED "*) : > "$READY" ;; *) exit 43 ;; esac
        /bin/cp "$BIN/python-stub" "$BIN/python3"
        ;;
esac
""")
    if manager == "apt":
        command("apt", "exit 0\n")
    if support != "no-python":
        command("python3", (tmp_path / "python-stub").read_text())
    if support == "working":
        (tmp_path / "ready").touch()
    functions = SCRIPT.read_text().split('# Parse command line arguments\nparse_arguments "$@"')[0]
    # Stop at the pip/Jupyter boundary, but require the same venv-creation command to succeed.
    program = functions + '\ninstall_python_packages() { python3 -m venv "$BIN/env"; echo SETUP; }\ninstall_python_jupyter\n'

    # Act
    result = subprocess.run(
        ["/bin/sh", "-c", program], capture_output=True, text=True, timeout=5,
        env={**os.environ, "PATH": str(tmp_path), "BIN": str(tmp_path),
             "READY": str(tmp_path / "ready"), "LOG": str(tmp_path / "calls"),
             "FAIL": "1" if support == "install-fails" else "0",
             "REQUIRED": "python3-venv" if manager == "apt" else "python-pip" if manager == "pacman" else "python3-pip"},
    )

    # Assert: a failed install must stop before setup, not be swallowed by the probe.
    assert result.returncode == (42 if support == "install-fails" else 0), result.stdout + result.stderr
    assert ("SETUP" in result.stdout) == (support != "install-fails")
    calls = (tmp_path / "calls").read_text().splitlines()
    # Healthy Python must not update/install; missing support must install before creating the venv.
    assert any(call.startswith("packages ") for call in calls) == (support != "working")
    if support != "install-fails":
        assert calls[-1] == "venv"
