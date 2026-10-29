"""executor-v1.139 (12 Oct 2026): the runner's entrypoint.sh force-recreated the executor on hosts whose
NVIDIA driver had been updated without a reboot; the NVIDIA hook refused the new container and 24 rented
nodes went offline. The entrypoint now probes the NVIDIA runtime with the running executor's own image
before it recreates anything.

The script is run under sh with a stub `docker` on PATH that records every call and answers
`compose ps`, `inspect` and `run` from STUB_* variables.
"""

import os
import shutil
import subprocess
import textwrap

import pytest

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "entrypoint.sh"))

DOCKER_STUB = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$*" >> "$STUB_CALLS"
    case "$1 $2" in
        "compose ps") echo "${STUB_RUNNING_EXECUTOR:-}" ;;
        "compose logs") exit 0 ;;
        "compose up") exit 0 ;;
        "inspect --format") echo "sha256:executorimage" ;;
        "ps --all") echo "${STUB_PROBE_LEFTOVERS:-}" ;;
        "rm --force") exit "${STUB_RM_EXIT:-0}" ;;
        "run --rm")
            [ -n "${STUB_PROBE_HANG:-}" ] && exec "$STUB_PROBE_HANG" 30
            [ -z "${STUB_PROBE_ERROR:-}" ] && exit 0
            echo "$STUB_PROBE_ERROR" >&2; exit 125 ;;
    esac
    exit 0
    """
)

MISMATCH = (
    "docker: Error response from daemon: failed to create task for container: OCI runtime create failed: "
    "nvidia-container-cli: initialization error: nvml error: driver/library version mismatch: unknown."
)


def run_entrypoint(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(DOCKER_STUB)
    (bin_dir / "docker").chmod(0o755)
    # the log-follow loop never ends; a `sleep` that fails ends the script after one round
    (bin_dir / "sleep").write_text('#!/bin/sh\necho "sleep $*" >> "$STUB_CALLS"\nexit 7\n')
    (bin_dir / "sleep").chmod(0o755)
    for tool in ("sh", "tail", "cat", "timeout"):
        os.symlink(shutil.which(tool), bin_dir / tool)
    calls = tmp_path / "calls"
    calls.touch()
    result = subprocess.run(
        ["sh", SCRIPT],
        env={"PATH": str(bin_dir), "STUB_CALLS": str(calls), **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, calls.read_text().splitlines()


def recreated(calls):
    return any(c.startswith("compose up") and "--force-recreate" in c for c in calls)


def test_probe_fails_with_driver_mismatch_keeps_running_executor(tmp_path):
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_PROBE_ERROR=MISMATCH)

    assert "run --rm --gpus all --label io.lium.executor-runner.probe=nvidia-runtime --entrypoint true sha256:executorimage" in calls
    assert not recreated(calls)
    assert not any(c.startswith("image prune") for c in calls)
    assert result.returncode != 0
    held = [line for line in result.stdout.splitlines() if line.startswith("Update held")]
    assert len(held) == 1
    assert "driver updated without a reboot" in held[0]
    assert "reboot the host or reload the NVIDIA driver once no renter is on it" in held[0]
    assert "sleep 300" in calls


def test_probe_fails_for_another_reason_also_holds(tmp_path):
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_PROBE_ERROR="could not select device driver \"\" with capabilities: [[gpu]]"
    )

    assert not recreated(calls)
    assert "Update held" in result.stdout
    assert "NVIDIA runtime cannot start a GPU container" in result.stdout


def test_probe_ok_recreates(tmp_path):
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123")

    probe = calls.index("run --rm --gpus all --label io.lium.executor-runner.probe=nvidia-runtime --entrypoint true sha256:executorimage")
    up = calls.index("compose up --pull always --detach --wait --force-recreate")
    assert probe < up
    assert "Update held" not in result.stdout


@pytest.mark.parametrize("probe_error", ["", MISMATCH])
def test_first_install_recreates_without_probe(tmp_path, probe_error):
    result, calls = run_entrypoint(tmp_path, STUB_PROBE_ERROR=probe_error)

    assert not any(c.startswith("run ") for c in calls)
    assert recreated(calls)


def test_a_hanging_probe_holds_the_update(tmp_path):
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_PROBE_HANG=shutil.which("sleep"), RUNNER_PROBE_TIMEOUT_SECONDS="1"
    )

    assert "Update held, running executor kept" in result.stdout
    assert not recreated(calls)


def test_leftover_probe_containers_are_removed_before_probing(tmp_path):
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_PROBE_LEFTOVERS="dead1\ndead2")

    removal = calls.index("rm --force dead1 dead2")
    probe = next(i for i, c in enumerate(calls) if c.startswith("run --rm"))
    assert removal < probe
    assert recreated(calls)


def test_a_leftover_probe_that_cannot_be_removed_holds_the_update(tmp_path):
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_PROBE_LEFTOVERS="dead1", STUB_RM_EXIT="1")

    assert not any(c.startswith("run ") for c in calls)
    assert not recreated(calls)
    assert "cannot be removed (dead1)" in result.stdout
    assert "sleep 300" in calls
