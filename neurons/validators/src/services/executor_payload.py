"""Code the validator runs on an executor comes from the validator, never from paths the executor reports.

`python_path` and `root_dir` arrive in the executor's answer, so they must not pick what runs as root on the node.
The scripts ship with the validator: the image copies them from the executor sources at build (Dockerfile,
`executor_setup` build context, its src/) into src/executor_payloads; in the repository they are read from neurons/executor/src.
Each command writes them into a fresh root-only directory before running them. The interpreter is the one the executor
image installs, at the path the image fixes; the validator names it, the executor does not.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
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

# the executor runs with pid: host, so a bare pkill/pgrep also sees other containers' and the host's processes;
# this selector keeps them to the shell's mount namespace, the executor's own
SAME_MOUNT_NS = '--ns "$$" --nslist mnt'


# a failed staging leaves nothing behind: its partial directory goes before the shell exits
_FAIL = '{ [ -z "${d:-}" ] || rm -rf "$d"; exit 1; }'


def service_dir_glob(purpose: str, parent: str | None = None) -> str:
    return f"{(parent or SERVICE_DIR_PARENT).rstrip('/')}/.lium-validator-{purpose}."


def new_run_tag() -> str:
    """A tag naming one validation's runs: their directories carry it, so a kill reaches only that validation's."""
    return secrets.token_hex(8)


def digests(files: tuple[str, ...]) -> str:
    """`sha256sum -c` input for the payload `files` as the validator ships them."""
    return "".join(f"{hashlib.sha256((PAYLOAD_DIR / name).read_bytes()).hexdigest()}  {name}\n" for name in files)


def install(
    purpose: str,
    files: tuple[str, ...],
    parent: str | None = None,
    cleanup: bool = False,
    tag: str = "",
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
    return "{ " + " && ".join(steps) + "; } || " + _FAIL


def verify(files: tuple[str, ...]) -> str:
    """Shell that checks the files in `$d` are exactly the validator's `files` (sha256), else removes `$d` and exits 1."""
    return f"{{ printf %s {shlex.quote(digests(files))} | (cd \"$d\" && sha256sum -c --quiet --strict -) >/dev/null; }} || {_FAIL}"


def run(files: tuple[str, ...]) -> str:
    """The trusted interpreter on the entry script in `$d`; -E -s: no PYTHON* variables, no user site."""
    return f'{shlex.quote(EXECUTOR_PYTHON)} -E -s "$d"/{shlex.quote(files[0])}'


def run_once(purpose: str, files: tuple[str, ...], args: str, env: str = "", tag: str = "") -> str:
    """Install, run with `args` (`env` assignments before the interpreter); the directory goes when the shell ends,
    also on a hangup or kill signal of the session. The exit status is the run's."""
    # a tagged run cancelled before its shell existed (kill_runs left the marker) never starts
    gate = f"[ ! -e {shlex.quote(cancel_marker(purpose, tag))} ] || exit 1; " if tag else ""
    return f"{gate}{install(purpose, files, cleanup=True, tag=tag)}; {env}{run(files)} {args}"


def label(files: tuple[str, ...], args: str) -> str:
    """What a log or event names instead of the command, which carries the script itself."""
    return f"{EXECUTOR_PYTHON} <validator>/{files[0]} {args}"


def process_pattern(purpose: str, files: tuple[str, ...], tag: str) -> str:
    """An extended regex matching the command line of a running entry script of `purpose` started with `tag` (for
    pkill -f), and not the shell that installed it (its mktemp template and `"$d"/` never form the path), so that
    shell's cleanup still runs. Other validations' runs carry other tags and are not matched."""
    return (service_dir_glob(purpose) + f"{tag}.@/" + files[0]).replace(".", r"\.").replace("@", "[[:alnum:]]{8}")


def launcher_pattern(purpose: str, tag: str) -> str:
    """An extended regex matching the command line of the shell that installs and starts a `purpose` run tagged `tag`
    (its mktemp template carries the tag), and nothing else."""
    return (service_dir_glob(purpose) + f"{tag}.XXXXXXXX").replace(".", r"\.")


def cancel_marker(purpose: str, tag: str) -> str:
    """The file kill_runs leaves for `tag`: outside the run directories' glob, so their removal keeps it."""
    return f"{service_dir_glob(purpose)}{tag}-cancelled"


def kill_runs(purpose: str, files: tuple[str, ...], tag: str) -> str:
    """Shell that ends every run of `purpose` tagged `tag`: SIGKILL to the whole process group of each tagged launcher,
    so a script it is still installing or has not started yet never starts, then to any tagged entry script left, then
    removes the tagged directories. Runs with other tags, and the shell running this, are not touched. Exits 1 when the
    marker could not be written (the kills still run): a launcher not up yet may then still start."""
    me = "$(ps -o pgid= -p $$ | tr -d ' ')"
    marker = shlex.quote(cancel_marker(purpose, tag))
    return (
        # the marker first: a launcher whose shell is not up yet (its exec still in flight) sees it and never starts;
        # one already up is in the process groups killed next. Markers of long-finished runs are swept.
        f"find {shlex.quote(SERVICE_DIR_PARENT)} -maxdepth 1 -name {shlex.quote(Path(cancel_marker(purpose, '*')).name)} "
        # in a subshell: a marker dash cannot create would end this shell before the kills; the status is kept instead
        f"-mmin +60 -delete 2>/dev/null; m=0; ( : > {marker} ) 2>/dev/null || m=1; "
        # no `--` before the group: dash's kill builtin takes it as a number and kills nothing
        f"for p in $(pgrep -f {SAME_MOUNT_NS} -- {shlex.quote(launcher_pattern(purpose, tag))}); do "
        "g=$(ps -o pgid= -p \"$p\" | tr -d ' '); "
        f'[ -n "$g" ] && [ "$g" != "{me}" ] && kill -9 "-$g" 2>/dev/null; done; '
        f"pkill -9 -f {SAME_MOUNT_NS} -- {shlex.quote(process_pattern(purpose, files, tag))}; "
        # without the marker a launcher not up yet may still start: the cancellation is incomplete (exit 1)
        f'rm -rf -- "{service_dir_glob(purpose)}"{tag}.*; [ "$m" = 0 ]'
    )
