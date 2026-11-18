"""Code the validator runs on an executor comes from the validator, never from paths the executor reports.

`python_path` and `root_dir` arrive in the executor's answer, so they must not pick what runs as root on the node.
The scripts ship with the validator: the image copies them from the executor sources at build (Dockerfile,
`executor_src` build context) into src/executor_payloads; in the repository they are read from neurons/executor/src.
Each command writes them into a fresh root-only directory before running them. The interpreter is the one the executor
image installs, at the path the image fixes; the validator names it, the executor does not.
"""

from __future__ import annotations

import base64
import shlex
from pathlib import Path

EXECUTOR_PYTHON = "/root/app/.venv/bin/python"
# the image's own directory (a fixed path, not a reported one): the executor's disk reserve backs it
# (setup_disk_reserve.sh), so the scripts are written there even when a renter has filled the host disk
SERVICE_DIR_PARENT = "/root/app"
_IMAGE_PAYLOADS = Path(__file__).resolve().parents[1] / "executor_payloads"
PAYLOAD_DIR = _IMAGE_PAYLOADS if _IMAGE_PAYLOADS.is_dir() else Path(__file__).resolve().parents[3] / "executor" / "src"

# entry script first; the rest are modules it imports from its own directory
GPU_MONITOR = ("gpus_utility.py", "services/pull_lock.py")
INSPECTOR = ("inspector_executor.py",)
VERIFYX = ("verifyx_executor.py",)
MATRIX = ("decrypt_challenge.py",)


def service_dir_glob(purpose: str, parent: str | None = None) -> str:
    return f"{(parent or SERVICE_DIR_PARENT).rstrip('/')}/.lium-validator-{purpose}."


def install(purpose: str, files: tuple[str, ...], parent: str | None = None) -> str:
    """Shell that makes a fresh 0700 directory under `parent`, sets `$d` to it and writes `files` into it; exits 1 on
    any failure."""
    steps = [f"d=$(mktemp -d {shlex.quote(service_dir_glob(purpose, parent) + 'XXXXXXXX')})"]
    for name in files:
        if "/" in name:
            steps.append(f'mkdir -p "$d"/{shlex.quote(str(Path(name).parent))}')
        data = base64.b64encode((PAYLOAD_DIR / name).read_bytes()).decode()
        steps.append(f'printf %s {data} | base64 -d > "$d"/{shlex.quote(name)}')
    return "{ " + " && ".join(steps) + "; } || exit 1"


def run(files: tuple[str, ...]) -> str:
    """The trusted interpreter on the entry script in `$d`; -E -s: no PYTHON* variables, no user site."""
    return f'{shlex.quote(EXECUTOR_PYTHON)} -E -s "$d"/{shlex.quote(files[0])}'


def run_once(purpose: str, files: tuple[str, ...], args: str, env: str = "") -> str:
    """Install, run with `args` (and `env` assignments before the interpreter), remove; the run's exit status."""
    return f'{install(purpose, files)}; {env}{run(files)} {args}; rc=$?; rm -rf "$d"; exit $rc'


def process_pattern(purpose: str, files: tuple[str, ...]) -> str:
    """An extended regex matching the command line of any running entry script of `purpose` (for pkill -f)."""
    return (service_dir_glob(purpose) + "*/" + files[0]).replace(".", r"\.").replace("*", "[^/]*")
