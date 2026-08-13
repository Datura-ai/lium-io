"""DAH-2679: a renter CVM's surviving directory is booted again, not stranded.

The shape under test is the one a host reboot leaves behind: the instance record and the VM
directory — disk included — are on disk, and the supervisor process is not. Before this task
that was a hard FAILED with the renter's data stranded on a disk nothing would ever boot.

The claims pinned here:

  * with `cvm_relaunch_renter` on, reconciliation re-spawns the supervisor over the SAME
    directory and lands back on RENTER_RUNNING;
  * the flag defaults OFF, and off keeps the fail-loudly behavior byte for byte;
  * only a RENTER CVM relaunches — a validation CVM's disk is deliberately not durable state;
  * the attempt count is spent even when the spawn fails, and the cap turns a dying guest
    into FAILED instead of a crash loop through every cvmd restart;
  * a missing disk, a stray TDX guest, or an unconfigured host all decline and fail as before.

No readiness is asserted because none is probed: a live supervisor is the same evidence the
adopt path accepts, and whether the sealing key actually unlocks the same disk is the
hardware half of DAH-2679 (a TDX host, not this suite).
"""

from pathlib import Path

import pytest
from cvmd.config import LaunchConfig
from cvmd.cvm import supervisor
from cvmd.cvm.instance import Instance, InstanceStore
from cvmd.cvm.manager import RELAUNCH_MAX_ATTEMPTS, CvmManager
from cvmd.cvm.switching import SwitchStore
from cvmd.state.machine import NodeState
from cvmd.state.store import StateStore

STALE_PID = 999_999_999
NEW_PID = 424242


def seed_cvm(
    state_dir: Path,
    launch_config: LaunchConfig,
    *,
    kind: str = "renter",
    attempts: int = 0,
    with_disk: bool = True,
    torn_down: bool = False,
) -> Path:
    """Persist what a host reboot leaves: a record, a directory, a disk — and no process.

    `torn_down=True` seeds the interrupted-teardown shape instead: `destroy` persists
    TEARDOWN before it kills anything, so that is the state a crash mid-teardown leaves.
    """
    vm_dir = launch_config.run_dir / "cvm-under-test"
    vm_dir.mkdir(parents=True, exist_ok=True)
    if with_disk:
        (vm_dir / supervisor.DISK_IMAGE).write_bytes(b"")

    InstanceStore(state_dir).set(
        Instance(
            instance_id="cvm-under-test",
            kind=kind,
            artifact_id="base-test",
            vm_dir=str(vm_dir),
            supervisor_pid=STALE_PID,
            created_at="2026-08-13T00:00:00+00:00",
            qemu="10.1.0",
            os_image_hash="a" * 64,
            compose_hash="b" * 64,
            rental_id="rental-1" if kind == "renter" else None,
            relaunch_attempts=attempts,
        )
    )
    store = StateStore(state_dir)
    store.transition(NodeState.LAUNCHING)
    store.transition(NodeState.RENTER_RUNNING if kind == "renter" else NodeState.VALIDATION_RUNNING)
    if torn_down:
        store.transition(NodeState.TEARDOWN)
    return vm_dir


def make_manager(state_dir: Path, config: LaunchConfig) -> CvmManager:
    # catalog_store=None on purpose: reconciliation reads the host, never the catalog, and a
    # relaunch that suddenly needed one would be a regression this argument turns into a crash.
    return CvmManager(
        config=config,
        catalog_store=None,
        store=StateStore(state_dir),
        instances=InstanceStore(state_dir),
        switches=SwitchStore(state_dir),
    )


@pytest.fixture
def relaunch_config(launch_config: LaunchConfig) -> LaunchConfig:
    from dataclasses import replace

    return replace(launch_config, relaunch_renter=True)


@pytest.fixture
def dead_supervisor(monkeypatch):
    """The recorded pid is gone, and nothing else holds a TDX guest on this host."""
    monkeypatch.setattr(supervisor, "is_supervisor", lambda pid, vm_dir: False)
    monkeypatch.setattr(supervisor, "running_cvms", list)


@pytest.fixture
def respawn(monkeypatch, dead_supervisor) -> list[dict]:
    """Record the re-spawn instead of starting QEMU."""
    calls: list[dict] = []

    def fake_spawn(*, scripts_dir, vm_dir, kp_port):
        calls.append({"scripts_dir": scripts_dir, "vm_dir": vm_dir, "kp_port": kp_port})
        return NEW_PID

    monkeypatch.setattr(supervisor, "spawn", fake_spawn)
    return calls


class TestTheRelaunch:
    def test_a_rebooted_renter_cvm_comes_back_over_its_own_directory(
        self, state_dir, relaunch_config, respawn
    ):
        vm_dir = seed_cvm(state_dir, relaunch_config)
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.RENTER_RUNNING
        assert respawn == [
            {
                "scripts_dir": relaunch_config.dstack_scripts_dir,
                "vm_dir": vm_dir,
                "kp_port": relaunch_config.key_provider_port,
            }
        ], "the SAME directory must be booted — a fresh one would not hold the renter's disk"
        assert manager._store.document.last_error is None

    def test_the_record_carries_the_new_pid_and_the_spent_attempt(
        self, state_dir, relaunch_config, respawn
    ):
        seed_cvm(state_dir, relaunch_config)
        manager = make_manager(state_dir, relaunch_config)
        manager.reconcile()

        instance = manager._instances.current
        assert instance.supervisor_pid == NEW_PID
        assert instance.relaunch_attempts == 1
        assert instance.report()["relaunch_attempts"] == 1

    def test_a_report_that_never_relaunched_keeps_its_pinned_shape(
        self, state_dir, relaunch_config
    ):
        seed_cvm(state_dir, relaunch_config)
        instance = InstanceStore(state_dir).current
        assert "relaunch_attempts" not in instance.report()

    def test_the_previous_lifes_pid_file_is_removed_before_the_spawn(
        self, state_dir, relaunch_config, dead_supervisor, monkeypatch
    ):
        """`spawn` returns the first pid it can read from `supervisor.pid`, and the surviving
        directory still holds the OLD life's file — without the unlink, the record would name
        a dead (or recycled) pid while the new guest boots."""
        vm_dir = seed_cvm(state_dir, relaunch_config)
        (vm_dir / "supervisor.pid").write_text(f"{STALE_PID}\n")
        seen: list[bool] = []

        def fake_spawn(*, scripts_dir, vm_dir, kp_port):
            seen.append((vm_dir / "supervisor.pid").exists())
            return NEW_PID

        monkeypatch.setattr(supervisor, "spawn", fake_spawn)
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.RENTER_RUNNING
        assert seen == [False], "the stale pid file must be gone before the supervisor starts"
        assert manager._instances.current.supervisor_pid == NEW_PID


class TestWhatDeclines:
    def test_the_flag_defaults_off_and_off_fails_exactly_as_before(
        self, state_dir, launch_config, respawn
    ):
        assert LaunchConfig().relaunch_renter is False
        seed_cvm(state_dir, launch_config)
        manager = make_manager(state_dir, launch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert "is gone but its directory" in manager._store.document.last_error
        assert respawn == []

    def test_a_validation_cvm_never_relaunches(self, state_dir, relaunch_config, respawn):
        seed_cvm(state_dir, relaunch_config, kind="validation")
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert respawn == []

    def test_an_interrupted_teardown_is_never_resurrected(
        self, state_dir, relaunch_config, respawn
    ):
        """`destroy` persists TEARDOWN before it kills anything. A crash mid-teardown must
        stay dead: relaunching it would boot a rental the platform already destroyed, over
        hardware the teardown was in the middle of verifiably releasing."""
        seed_cvm(state_dir, relaunch_config, torn_down=True)
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert respawn == []

    def test_a_directory_whose_disk_is_gone_has_nothing_to_boot(
        self, state_dir, relaunch_config, respawn, monkeypatch
    ):
        vm_dir = seed_cvm(state_dir, relaunch_config, with_disk=False)
        # Keep something in the directory so it "remains" for the failure message's purposes.
        (vm_dir / "console.log").write_text("old boot\n")
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert respawn == []

    def test_a_stray_tdx_guest_blocks_the_relaunch(
        self, state_dir, relaunch_config, respawn, monkeypatch
    ):
        """GPU passthrough is exclusive — booting over a foreign guest double-books it."""
        seed_cvm(state_dir, relaunch_config)
        monkeypatch.setattr(
            supervisor, "running_cvms", lambda: [(4242, "/opt/qemu-dstack/bin/qemu")]
        )
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert respawn == []


class TestTheAttemptCap:
    def test_a_failed_spawn_spends_the_attempt_and_fails_the_node(
        self, state_dir, relaunch_config, dead_supervisor, monkeypatch
    ):
        seed_cvm(state_dir, relaunch_config)

        def refuse(**kwargs):
            raise supervisor.SupervisorError("no QEMU on this host")

        monkeypatch.setattr(supervisor, "spawn", refuse)
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert manager._instances.current.relaunch_attempts == 1, (
            "an attempt that consumed host resources is spent whether or not it produced "
            "a supervisor — otherwise a spawn dying mid-way retries unbounded"
        )

    def test_past_the_cap_the_node_fails_instead_of_looping(
        self, state_dir, relaunch_config, respawn
    ):
        seed_cvm(state_dir, relaunch_config, attempts=RELAUNCH_MAX_ATTEMPTS)
        manager = make_manager(state_dir, relaunch_config)

        assert manager.reconcile() is NodeState.FAILED
        assert respawn == []


class TestOldRecordsStillDecode:
    def test_an_instance_json_written_before_this_task_reads_as_zero_attempts(
        self, state_dir, relaunch_config
    ):
        """The field is new; the fleet's records are not. An old record must not read as
        corrupt — that would fail the node for a schema change."""
        import json

        vm_dir = relaunch_config.run_dir / "old-cvm"
        record = {
            "version": 1,
            "instance_id": "old-cvm",
            "kind": "renter",
            "artifact_id": "base-test",
            "vm_dir": str(vm_dir),
            "supervisor_pid": STALE_PID,
            "created_at": "2026-08-01T00:00:00+00:00",
            "qemu": "10.1.0",
            "os_image_hash": "a" * 64,
            "compose_hash": "b" * 64,
            "ports": [],
            "ssh_fingerprint": None,
            "rental_id": "rental-0",
        }
        (state_dir / "instance.json").write_text(json.dumps(record))

        loaded = InstanceStore(state_dir).current
        assert loaded is not None
        assert loaded.relaunch_attempts == 0
