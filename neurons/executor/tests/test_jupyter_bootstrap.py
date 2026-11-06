import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("manager, support, expected", [
    ("apt", "no-python", (0, True)), ("apt", "no-venv", (0, True)),
    ("apt", "working", (0, True)), ("apt", "install-fails", (42, False)),
    ("apk", "no-python", (0, True)),
])
def test_bootstrap_installs_only_missing_python_support(tmp_path: Path, manager: str, support: str, expected: tuple[int, bool]) -> None:
    # Arrange: package installation supplies Python and venv; failed installation starts without Python.
    commands: dict[str, str] = {
        "python": 'case "$1" in --version) exit 0;; -c) [ -f "$READY" ];; -m) echo venv >> "$LOG"; [ -f "$READY" ];; esac',
        "apt": "exit 0",
        "apt-get" if manager == "apt" else "apk": 'echo "packages $*" >> "$LOG"; [ "$FAIL" = 0 ] || exit 42\n'
                   'case "$1" in install|add) case " $* " in *" $REQUIRED "*) : > "$READY";; *) exit 43;; esac;'
                   'case "$0:$*" in */apk:*python3-pip*) exit 44;; esac; /bin/ln -sf "$BIN/python" "$BIN/python3";; esac',
    }
    for name, body in commands.items():
        path = tmp_path / name
        path.write_text("#!/bin/sh\nset -e\n" + body + "\n")
        path.chmod(0o755)
    if support in ("no-venv", "working"):
        (tmp_path / "python3").symlink_to(tmp_path / "python")
    if support == "working":
        (tmp_path / "ready").touch()
    functions = (Path(__file__).parents[1] / "run_jupyter.sh").read_text().split('# Parse command line arguments\nparse_arguments "$@"')[0]
    program = functions + '\ninstall_python_packages() { python3 -m venv "$BIN/env"; echo SETUP; }\ninstall_python_jupyter\n'
    # Act: run the real detection and system-package functions, stopping at the pip boundary.
    result = subprocess.run(["/bin/sh", "-c", program], capture_output=True, text=True, timeout=5,
                            env={"PATH": str(tmp_path), "BIN": str(tmp_path),
                                 "REQUIRED": "python3-venv" if manager == "apt" else "py3-pip", "READY": str(tmp_path / "ready"), "LOG": str(tmp_path / "calls"), "FAIL": "1" if support == "install-fails" else "0"})
    # Assert: healthy Python skips packages; failures stop before setup; success creates the venv.
    assert (result.returncode, "SETUP" in result.stdout) == expected, result.stdout + result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert any(call.startswith("packages ") for call in calls) == (support != "working")
    assert (calls[-1] == "venv") == expected[1]
