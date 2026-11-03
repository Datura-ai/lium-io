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
import time

import pytest

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "entrypoint.sh"))

DOCKER_STUB = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$*" >> "$STUB_CALLS"
    case "$1 $2" in
        "compose ps") echo "${STUB_RUNNING_EXECUTOR:-}" ;;
        "top abc123") echo "${STUB_TOP:-}" ;;
        "compose logs") exit 0 ;;
        "compose up") exit 0 ;;
        "inspect --format") echo "sha256:executorimage" ;;
        "ps --all") echo "${STUB_PROBE_LEFTOVERS:-}" ;;
        "ps --no-trunc") echo "${STUB_HELPERS:-}" ;;
        "rm --force") exit "${STUB_RM_EXIT:-0}" ;;
        "image ls") echo "${STUB_EXECUTOR_IMAGES:-}" ;;
        "image rm") exit "${STUB_IMAGE_RM_EXIT:-0}" ;;
        "tag sha256:executorimage") exit "${STUB_TAG_EXIT:-0}" ;;
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
    for tool in ("sh", "tail", "cat", "timeout", "date", "rm"):
        os.symlink(shutil.which(tool), bin_dir / tool)
    calls = tmp_path / "calls"
    calls.touch()
    result = subprocess.run(
        ["sh", SCRIPT],
        env={"PATH": str(bin_dir), "STUB_CALLS": str(calls), "RUNNER_STORAGE_HOLD_FILE": str(tmp_path / "hold"), **env},
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
    assert not any(c.startswith("image rm") for c in calls)


def test_previous_executor_image_is_removed_after_the_update(tmp_path):
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_EXECUTOR_IMAGES="sha256:previous sha256:executorimage"
    )

    up = calls.index("compose up --pull always --detach --wait --force-recreate")
    assert calls.index("image rm sha256:previous") > up
    assert "image rm sha256:executorimage" not in calls
    assert "Removed old executor image sha256:previous" in result.stdout


def test_every_past_releases_executor_image_goes_but_the_running_one(tmp_path):
    result, calls = run_entrypoint(
        tmp_path,
        STUB_RUNNING_EXECUTOR="abc123",
        STUB_EXECUTOR_IMAGES="sha256:v163 sha256:v164 sha256:executorimage",
    )

    assert "image rm sha256:v163" in calls
    assert "image rm sha256:v164" in calls
    assert "image rm sha256:executorimage" not in calls


def test_previous_image_still_in_use_is_kept_and_the_update_goes_on(tmp_path):
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_EXECUTOR_IMAGES="sha256:previous", STUB_IMAGE_RM_EXIT="1"
    )

    assert "image rm sha256:previous" in calls
    assert "Removed old" not in result.stdout
    assert "image prune -f" in calls


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


def test_the_running_image_is_tagged_latest_before_any_old_image_is_removed(tmp_path):
    # the validator's probes and Jupyter copies run daturaai/compute-subnet-executor:latest; removing the previous
    # image drops its :latest, so the running image must hold the tag before any removal
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_EXECUTOR_IMAGES="sha256:v163 sha256:previous sha256:executorimage"
    )

    tag = calls.index("tag sha256:executorimage daturaai/compute-subnet-executor:latest")
    removals = [i for i, c in enumerate(calls) if c.startswith("image rm")]
    assert len(removals) == 2
    assert all(tag < i for i in removals)


def test_old_images_are_kept_when_the_running_image_cannot_be_tagged(tmp_path):
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_EXECUTOR_IMAGES="sha256:previous sha256:executorimage", STUB_TAG_EXIT="1"
    )

    assert "tag sha256:executorimage daturaai/compute-subnet-executor:latest" in calls
    assert not any(c.startswith("image rm") for c in calls)
    assert recreated(calls)
    assert "image prune -f" in calls
    assert any(c.startswith("compose logs") for c in calls)


STORAGE_TOP = "COMMAND\n/usr/bin/python3 /root/app/src/storage_runner.py --spec -"


def test_running_backup_holds_the_update(tmp_path):
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_TOP=STORAGE_TOP)

    assert not recreated(calls)
    assert not any(c.startswith("run ") for c in calls)
    assert "a backup is running in it (held 0 min of 30)" in result.stdout
    assert "sleep 120" in calls
    assert (tmp_path / "hold").read_text().strip().isdigit()


def test_hold_expires_and_updates(tmp_path):
    (tmp_path / "hold").write_text("1\n")  # held since 1970: far past the 30 min cap
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_TOP=STORAGE_TOP)

    assert "Update hold expired" in result.stdout
    assert recreated(calls)
    assert not (tmp_path / "hold").exists()


def test_no_storage_operation_updates_and_clears_hold(tmp_path):
    (tmp_path / "hold").write_text("1\n")
    result, calls = run_entrypoint(tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_TOP="COMMAND\npython3 /root/app/src/executor.py")

    assert "top abc123 -o args" in calls
    assert recreated(calls)
    assert not (tmp_path / "hold").exists()


RESTORE_HELPER = '"nsenter -t 4242 -m -U -- restic restore 8a42d9ed --target /root/restored --json"'


def test_running_restore_holds_up_to_two_hours(tmp_path):
    (tmp_path / "hold").write_text(f"{int(time.time()) - 3600}\n")  # held 1 h: past the backup cap
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_TOP=STORAGE_TOP, STUB_HELPERS=RESTORE_HELPER
    )

    assert not recreated(calls)
    assert "a restore is running in it (held 60 min of 120)" in result.stdout


def test_restore_hold_expires_after_two_hours(tmp_path):
    (tmp_path / "hold").write_text(f"{int(time.time()) - 7300}\n")
    result, calls = run_entrypoint(
        tmp_path, STUB_RUNNING_EXECUTOR="abc123", STUB_TOP=STORAGE_TOP, STUB_HELPERS=RESTORE_HELPER
    )

    assert "Update hold expired after 121 min: updating with a restore still running." in result.stdout
    assert recreated(calls)
