"""Code the validator runs on an executor comes from the validator, never from paths the executor reports.

`python_path` and `root_dir` arrive in the executor's answer, so they must not pick what runs as root on the node.
The scripts ship with the validator: the image copies them from the executor sources at build (Dockerfile,
`executor_src` build context) into src/executor_payloads; in the repository they are read from neurons/executor/src.
Each command writes them into a fresh root-only directory before running them. The interpreter is the one the executor
image installs, at the path the image fixes; the validator names it, the executor does not.
"""

from __future__ import annotations

import base64
import hashlib
import io
import secrets
import shlex
import tarfile
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
# the storage runner and the package it imports, too large for one command line: sent as an archive on stdin (unpack)
STORAGE = ("storage_runner.py",) + tuple(sorted(f"storage/{p.name}" for p in (PAYLOAD_DIR / "storage").glob("*.py")))

# the executor runs with pid: host, so a bare pkill/pgrep also sees other containers' and the host's processes;
# this selector keeps them to the shell's mount namespace, the executor's own
SAME_MOUNT_NS = '--ns "$$" --nslist mnt'


def service_dir_glob(purpose: str, parent: str | None = None) -> str:
    return f"{(parent or SERVICE_DIR_PARENT).rstrip('/')}/.lium-validator-{purpose}."


def new_run_tag() -> str:
    """A tag naming one validation's runs: their directories carry it, so a kill reaches only that validation's."""
    return secrets.token_hex(8)


def digests(files: tuple[str, ...]) -> str:
    """`sha256sum -c` input for the payload `files` as the validator ships them."""
    return "".join(f"{hashlib.sha256((PAYLOAD_DIR / name).read_bytes()).hexdigest()}  {name}\n" for name in files)


def install(
    purpose: str, files: tuple[str, ...], parent: str | None = None, cleanup: bool = False, tag: str = ""
) -> str:
    """Shell that makes a fresh 0700 directory under `parent`, sets `$d` to it and writes `files` into it; exits 1 on
    any failure. `cleanup`: the directory goes when the shell ends (also on HUP/INT/TERM), set before the first write.
    `tag` (from new_run_tag) goes into the directory name, for process_pattern."""
    template = service_dir_glob(purpose, parent) + (f"{tag}." if tag else "") + "XXXXXXXX"
    steps = [f"d=$(mktemp -d {shlex.quote(template)})"]
    if cleanup:
        steps += ["trap 'rm -rf \"$d\"' EXIT", "trap 'exit 1' HUP INT TERM"]
    for name in files:
        if "/" in name:
            steps.append(f'mkdir -p "$d"/{shlex.quote(str(Path(name).parent))}')
        data = base64.b64encode((PAYLOAD_DIR / name).read_bytes()).decode()
        steps.append(f'printf %s {data} | base64 -d > "$d"/{shlex.quote(name)}')
    return "{ " + " && ".join(steps) + "; } || exit 1"


def archive(files: tuple[str, ...]) -> bytes:
    """`files` as the stdin of an `unpack` command: a gzipped tar, base64-encoded."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in files:
            data = (PAYLOAD_DIR / name).read_bytes()
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o600
            tar.addfile(entry, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue())


def unpack(purpose: str, cleanup: bool = False) -> str:
    """As `install`, but the files come from stdin, an `archive` of them."""
    steps = [f"d=$(mktemp -d {shlex.quote(service_dir_glob(purpose) + 'XXXXXXXX')})"]
    if cleanup:
        steps += ["trap 'rm -rf \"$d\"' EXIT", "trap 'exit 1' HUP INT TERM"]
    steps.append('base64 -d | tar -xzf - -C "$d"')
    return "{ " + " && ".join(steps) + "; } || exit 1"


def run(files: tuple[str, ...]) -> str:
    """The trusted interpreter on the entry script in `$d`; -E -s: no PYTHON* variables, no user site."""
    return f'{shlex.quote(EXECUTOR_PYTHON)} -E -s "$d"/{shlex.quote(files[0])}'


def run_once(purpose: str, files: tuple[str, ...], args: str, env: str = "", tag: str = "") -> str:
    """Install, run with `args` (`env` assignments before the interpreter); the directory goes when the shell ends,
    also on a hangup or kill signal of the session. The exit status is the run's."""
    return f"{install(purpose, files, cleanup=True, tag=tag)}; {env}{run(files)} {args}"


def label(files: tuple[str, ...], args: str) -> str:
    """What a log or event names instead of the command, which carries the script itself."""
    return f"{EXECUTOR_PYTHON} <validator>/{files[0]} {args}"


def process_pattern(purpose: str, files: tuple[str, ...], tag: str) -> str:
    """An extended regex matching the command line of a running entry script of `purpose` started with `tag` (for
    pkill -f), and not the shell that installed it (its mktemp template and `"$d"/` never form the path), so that
    shell's cleanup still runs. Other validations' runs carry other tags and are not matched."""
    return (service_dir_glob(purpose) + f"{tag}.@/" + files[0]).replace(".", r"\.").replace("@", "[[:alnum:]]{8}")
