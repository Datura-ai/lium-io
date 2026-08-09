"""The launch path: what it refuses, in what order, and what it leaves behind when it does.

The launches here stop at the point QEMU would start. `supervisor.spawn` is the seam — a test
host has no TDX, no GPUs and no OS image, so starting a real guest is the hardware run's job
(see the acceptance evidence on the PR). Everything up to and including the measurement gate
is real: the real dstack.py writes the real VM directory and the real hashes are compared.

That split is deliberate. Every refusal below happens *before* a process exists, so these are
exactly the paths where a bug would leave a node in a state someone has to clean up by hand.
"""

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from cvmd.config import Config, LaunchConfig
from cvmd.cvm import measure, supervisor
from cvmd.cvm.instance import InstanceStore
from cvmd.cvm.manager import CvmManager, LaunchFailure, Triple
from cvmd.state.machine import NodeState
from cvmd.state.store import StateStore

QEMU_FALLBACK = "0.0.0-test"
OS_IMAGE_HASH = "d" * 64


@pytest.fixture
def free_ports() -> tuple[int, int]:
    with socket.socket() as a, socket.socket() as b:
        a.bind(("127.0.0.1", 0))
        b.bind(("127.0.0.1", 0))
        return a.getsockname()[1], b.getsockname()[1]


@pytest.fixture
def catalog_file(tmp_path, compose_file, guest_scripts, image_dir, dstack, monkeypatch) -> Path:
    """A catalog pinning the triple this host will actually produce.

    The compose hash is computed by preparing the CVM once and measuring it — the same value
    `setup_instance` will produce during the test. Hardcoding a hash would make the test assert
    that two constants are equal; deriving it makes the test assert that the *gate* works.
    """
    from cvmd.catalog import Artifact
    from cvmd.dstack.plan import setup_namespace

    init, pre_launch = guest_scripts
    probe_dir = tmp_path / "probe" / "instance"
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    artifact = Artifact(
        id="validation-test",
        kind="validation",
        qemu="unused",
        os_image_hash=OS_IMAGE_HASH,
        compose_hash="0" * 64,
        os_image_path=image_dir,
        compose_path=compose_file,
        init_script=init,
        pre_launch_script=pre_launch,
    )
    dstack.DStackManager().setup_instance(
        setup_namespace(
            artifact=artifact,
            launch=LaunchConfig(vcpus=2, memory="2G", disk="20G"),
            vm_dir=probe_dir,
        )
    )
    compose_hash = measure.compose_hash(probe_dir)

    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {
                        "id": "validation-test",
                        "kind": "validation",
                        "qemu": QEMU_FALLBACK,
                        "os_image_hash": OS_IMAGE_HASH,
                        "compose_hash": compose_hash,
                        "os_image_path": str(image_dir),
                        "compose_path": str(compose_file),
                        "init_script": str(init),
                        "pre_launch_script": str(pre_launch),
                    }
                ],
            }
        )
    )
    return path


@pytest.fixture
def approved(catalog_file) -> Triple:
    entry = json.loads(catalog_file.read_text())["artifacts"][0]
    return Triple(
        qemu=entry["qemu"],
        os_image_hash=entry["os_image_hash"],
        compose_hash=entry["compose_hash"],
    )


@pytest.fixture
def launch_config(tmp_path, dstack_scripts, catalog_file, env_file, free_ports) -> LaunchConfig:
    return LaunchConfig(
        dstack_scripts_dir=dstack_scripts,
        catalog_path=catalog_file,
        run_dir=tmp_path / "vms",
        vcpus=2,
        memory="2G",
        disk="20G",
        gpus=(),
        ports=(f"tcp:127.0.0.1:{free_ports[0]}:2200", f"tcp:127.0.0.1:{free_ports[1]}:8001"),
        env_file=env_file,
        ssh_guest_port=2200,
        launch_timeout_seconds=2,
    )


@pytest.fixture
def manager(state_dir, launch_config, monkeypatch) -> CvmManager:
    # This host's QEMU is whatever the developer happens to have, or none. Pinning the reported
    # version makes the catalog's `qemu` field meaningful in a test rather than machine-dependent.
    monkeypatch.setattr(
        "cvmd.cvm.measure.qemu_version", lambda _dstack: QEMU_FALLBACK, raising=True
    )
    return CvmManager(
        config=launch_config, store=StateStore(state_dir), instances=InstanceStore(state_dir)
    )


@pytest.fixture
def spawned(monkeypatch) -> list[dict]:
    """Record spawns instead of starting QEMU, and report the fake pid as alive."""
    calls: list[dict] = []

    def fake_spawn(*, scripts_dir, vm_dir, kp_port):
        calls.append({"scripts_dir": scripts_dir, "vm_dir": vm_dir, "kp_port": kp_port})
        return 424242

    monkeypatch.setattr(supervisor, "spawn", fake_spawn)
    monkeypatch.setattr(
        supervisor, "is_supervisor", lambda pid, vm_dir: pid == 424242 and bool(calls)
    )
    monkeypatch.setattr(supervisor, "running_cvms", list)
    return calls


@pytest.fixture
def guest_is_up(monkeypatch):
    monkeypatch.setattr(
        "cvmd.cvm.sshkey.read_host_key", lambda host, port: "SHA256:testfingerprint"
    )


class TestASuccessfulLaunch:
    def test_it_reports_the_instance_the_ports_and_the_fingerprint(
        self, manager, approved, spawned, guest_is_up
    ):
        report = manager.create(kind="validation", triple=approved)

        assert report["kind"] == "validation"
        assert report["artifact_id"] == "validation-test"
        assert report["instance_id"]
        assert report["ssh_host_key_fingerprint"] == "SHA256:testfingerprint"
        assert [p["guest_port"] for p in report["ports"]] == [2200, 8001]
        assert report["measurements"]["compose_hash"] == approved.compose_hash
        assert report["state"] == "VALIDATION_RUNNING"

    def test_it_drives_the_states_the_task_asks_for(self, manager, approved, spawned, guest_is_up):
        assert manager.reconcile() is NodeState.RECONCILING
        manager.create(kind="validation", triple=approved)
        assert manager._store.state is NodeState.VALIDATION_RUNNING

    def test_the_supervisor_gets_the_scripts_directory_and_the_vm_directory(
        self, manager, approved, spawned, guest_is_up, dstack_scripts
    ):
        manager.create(kind="validation", triple=approved)
        assert spawned[0]["scripts_dir"] == dstack_scripts
        assert (spawned[0]["vm_dir"] / "shared" / "app-compose.json").is_file()

    def test_the_state_endpoint_describes_the_running_cvm(
        self, manager, approved, spawned, guest_is_up
    ):
        report = manager.create(kind="validation", triple=approved)
        described = manager.describe()
        assert described["instance_id"] == report["instance_id"]
        assert described["supervisor_alive"] is True


class TestRefusals:
    def test_a_triple_the_catalog_does_not_carry(self, manager, approved, spawned):
        """The task's second verification case: refused, with an explicit reason."""
        wrong = Triple(
            qemu=approved.qemu, os_image_hash=approved.os_image_hash, compose_hash="e" * 64
        )
        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=wrong)

        assert raised.value.status == 422
        assert "compose_hash" in raised.value.reason
        assert not spawned, "nothing may be started for an unapproved triple"

    def test_a_second_launch_while_one_runs(self, manager, approved, spawned, guest_is_up):
        """The task's third verification case: rejected, node state stays consistent."""
        manager.create(kind="validation", triple=approved)

        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert raised.value.status == 409
        assert manager._store.state is NodeState.VALIDATION_RUNNING
        assert len(spawned) == 1

    def test_a_confidential_guest_started_outside_cvmd(
        self, manager, approved, spawned, monkeypatch
    ):
        """GPU passthrough is exclusive whoever claimed it — including `lium-cvm.sh`."""
        monkeypatch.setattr(
            supervisor, "running_cvms", lambda: [(4242, "/opt/qemu-dstack/bin/qemu-system-x86_64")]
        )
        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert raised.value.status == 409
        assert "4242" in raised.value.reason
        assert not spawned

    def test_a_forwarded_port_already_in_use(self, manager, approved, spawned, free_ports):
        with socket.socket() as held:
            held.bind(("127.0.0.1", free_ports[0]))
            held.listen()

            with pytest.raises(LaunchFailure) as raised:
                manager.create(kind="validation", triple=approved)

        assert raised.value.status == 409
        assert str(free_ports[0]) in raised.value.reason
        assert not spawned

    def test_a_host_with_no_launch_configuration(self, state_dir, approved):
        manager = CvmManager(
            config=LaunchConfig(), store=StateStore(state_dir), instances=InstanceStore(state_dir)
        )
        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert raised.value.status == 503
        assert "run_dir" in raised.value.reason

    def test_a_stray_vm_directory_blocks_a_launch(self, manager, approved, spawned, launch_config):
        """A stopped CVM still owns this node's GPUs and ports until its directory is gone."""
        stray = launch_config.run_dir / "left-behind"
        stray.mkdir(parents=True)
        (stray / supervisor.DISK_IMAGE).write_bytes(b"")

        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert raised.value.status == 409
        assert "left-behind" in raised.value.reason
        assert not spawned

    def test_an_ssh_guest_port_that_nothing_forwards(
        self, state_dir, launch_config, approved, spawned, monkeypatch
    ):
        """A one-character config slip must not become "running, fingerprint null".

        With no mapping to read the host key through there is no readiness check left, so the
        launch would report VALIDATION_RUNNING milliseconds after spawning QEMU — before the
        guest had booted. Refused before anything is prepared instead.
        """
        monkeypatch.setattr(
            "cvmd.cvm.measure.qemu_version", lambda _dstack: QEMU_FALLBACK, raising=True
        )
        mismatched = replace(launch_config, ssh_guest_port=2201)
        manager = CvmManager(
            config=mismatched, store=StateStore(state_dir), instances=InstanceStore(state_dir)
        )

        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert raised.value.status == 503
        assert "cvm_ssh_guest_port" in raised.value.reason
        assert not spawned
        assert not mismatched.run_dir.exists()


class TestTheMeasurementGate:
    def test_a_prepared_cvm_that_would_measure_differently_is_not_launched(
        self, manager, approved, spawned, monkeypatch
    ):
        """The gate's whole reason to exist.

        The catalog approves the triple, so resolution succeeds — but what lands on disk hashes
        to something else. A launch here would attest as a stack nobody whitelisted.
        """
        monkeypatch.setattr(measure, "compose_hash", lambda vm_dir: "f" * 64)

        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert raised.value.status == 409
        assert "would not measure as the requested triple" in raised.value.reason
        assert not spawned

    def test_the_refused_launch_leaves_nothing_behind(
        self, manager, approved, spawned, monkeypatch, launch_config
    ):
        """A refusal has to return the node to where it was, or the next launch is blocked."""
        monkeypatch.setattr(measure, "compose_hash", lambda vm_dir: "f" * 64)
        with pytest.raises(LaunchFailure):
            manager.create(kind="validation", triple=approved)

        assert supervisor.vm_directories(launch_config.run_dir) == []
        assert list(launch_config.run_dir.iterdir()) == []

    def test_every_mismatch_is_named_at_once(self, manager, approved, spawned, monkeypatch):
        monkeypatch.setattr(measure, "compose_hash", lambda vm_dir: "f" * 64)
        monkeypatch.setattr(measure, "os_image_hash", lambda image_path: "9" * 64)

        with pytest.raises(LaunchFailure) as raised:
            manager.create(kind="validation", triple=approved)

        assert "compose_hash" in raised.value.reason
        assert "os_image_hash" in raised.value.reason


class TestReconciliation:
    def test_a_node_with_nothing_on_it_is_idle(self, manager):
        assert manager.reconcile() is NodeState.RECONCILING

    def test_a_running_cvm_is_adopted_after_a_restart(
        self, manager, approved, spawned, guest_is_up, state_dir, launch_config, monkeypatch
    ):
        """A CVM outlives a cvmd restart; the daemon has to come back under it."""
        manager.create(kind="validation", triple=approved)

        restarted = CvmManager(
            config=launch_config,
            store=StateStore(state_dir),
            instances=InstanceStore(state_dir),
        )
        assert restarted.reconcile() is NodeState.VALIDATION_RUNNING

    def test_a_record_whose_supervisor_is_gone_fails_the_node(
        self, manager, approved, spawned, guest_is_up, state_dir, launch_config, monkeypatch
    ):
        """Measured on au11: a stale `runtime.json` outlived its process by hours.

        A recorded pid is a claim, so a record with no live process is a node in an unknown
        state — not an idle one.
        """
        manager.create(kind="validation", triple=approved)
        monkeypatch.setattr(supervisor, "is_supervisor", lambda pid, vm_dir: False)

        restarted = CvmManager(
            config=launch_config,
            store=StateStore(state_dir),
            instances=InstanceStore(state_dir),
        )
        assert restarted.reconcile() is NodeState.FAILED
        assert "is gone but its directory" in restarted._store.document.last_error

    def test_a_vm_directory_cvmd_never_recorded_fails_the_node(self, manager, launch_config):
        stray = launch_config.run_dir / "unknown"
        stray.mkdir(parents=True)
        (stray / supervisor.DISK_IMAGE).write_bytes(b"")

        assert manager.reconcile() is NodeState.FAILED
        assert "no record of" in manager._store.document.last_error

    def test_a_state_claiming_a_cvm_that_is_not_there_fails_the_node(self, manager):
        """The daemon says VALIDATION_RUNNING, the host says nothing is running. That is a
        failure to investigate, not a node quietly declaring itself free."""
        manager._store.transition(NodeState.LAUNCHING)
        manager._store.transition(NodeState.VALIDATION_RUNNING)

        assert manager.reconcile() is NodeState.FAILED
        assert "no CVM is running on it" in manager._store.document.last_error

    def test_a_host_that_manages_no_cvms_is_left_alone(self, state_dir):
        """cvmd installed before its catalog exists must not re-derive a state it cannot see."""
        store = StateStore(state_dir)
        store.transition(NodeState.LAUNCHING)
        manager = CvmManager(config=LaunchConfig(), store=store, instances=InstanceStore(state_dir))

        assert manager.reconcile() is NodeState.LAUNCHING

    def test_an_unreadable_instance_record_fails_closed(self, state_dir, launch_config):
        """Never "no record, therefore idle" — a CVM cvmd forgot may still be running."""
        (state_dir / "instance.json").write_text("{ truncated")
        manager = CvmManager(
            config=launch_config, store=StateStore(state_dir), instances=InstanceStore(state_dir)
        )

        assert manager.reconcile() is NodeState.FAILED
        assert "unreadable or malformed" in manager._store.document.last_error


class TestSettling:
    """`_settle` is the recovery path, so it must not raise the error it exists to avoid."""

    @pytest.mark.parametrize(
        "running",
        [NodeState.LAUNCHING, NodeState.VALIDATION_RUNNING, NodeState.RENTER_RUNNING],
    )
    def test_settling_to_reconciling_from_a_running_state(self, manager, running):
        """RECONCILING is the staging point *and* the destination here.

        Reaching it is the whole move; asking for it again is RECONCILING -> RECONCILING, which
        the machine rejects.
        """
        manager._store.transition(running)

        assert manager._settle(NodeState.RECONCILING) is NodeState.RECONCILING
        assert manager._store.state is NodeState.RECONCILING

    def test_settling_to_a_running_state_still_goes_the_long_way(self, manager):
        manager._store.transition(NodeState.FAILED)

        assert manager._settle(NodeState.VALIDATION_RUNNING) is NodeState.VALIDATION_RUNNING


class TestOverHttp:
    """The same paths through auth, the router and the worker-thread offload."""

    @pytest.fixture
    def launching_client(
        self, clients_file, state_dir, launch_config, spawned, guest_is_up, monkeypatch
    ):
        from conftest import signed_request  # noqa: F401 - imported for the tests below
        from cvmd.app import create_app
        from fastapi.testclient import TestClient

        monkeypatch.setattr(
            "cvmd.cvm.measure.qemu_version", lambda _dstack: QEMU_FALLBACK, raising=True
        )
        config = Config(authorized_clients=clients_file, state_dir=state_dir, launch=launch_config)
        with TestClient(create_app(config), raise_server_exceptions=False) as client:
            yield client

    def _body(self, approved: Triple) -> bytes:
        return json.dumps(
            {
                "kind": "validation",
                "qemu": approved.qemu,
                "os_image_hash": approved.os_image_hash,
                "compose_hash": approved.compose_hash,
            }
        ).encode()

    def test_the_validator_key_launches_and_gets_the_report(
        self, launching_client, validator_key, approved
    ):
        from conftest import signed_request

        response = signed_request(
            launching_client, validator_key, "POST", "/v1/cvm", body=self._body(approved)
        )

        assert response.status_code == 201
        assert response.json()["ssh_host_key_fingerprint"] == "SHA256:testfingerprint"

    def test_the_state_endpoint_then_shows_the_cvm(
        self, launching_client, validator_key, platform_key, approved
    ):
        from conftest import signed_request

        created = signed_request(
            launching_client, validator_key, "POST", "/v1/cvm", body=self._body(approved)
        ).json()
        state = signed_request(launching_client, platform_key, "GET", "/v1/state").json()

        assert state["state"] == "VALIDATION_RUNNING"
        assert state["cvm"]["instance_id"] == created["instance_id"]

    def test_a_triple_missing_from_the_body_is_422(self, launching_client, validator_key):
        from conftest import signed_request

        response = signed_request(
            launching_client,
            validator_key,
            "POST",
            "/v1/cvm",
            body=json.dumps({"kind": "validation"}).encode(),
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "value", ["A" * 64, "sha256:" + "a" * 64, "abc"], ids=["uppercase", "prefixed", "short"]
    )
    def test_a_hash_that_is_not_lowercase_hex_is_422(self, launching_client, validator_key, value):
        """Rejected at the edge, so a malformed pin cannot read as an unapproved stack later."""
        from conftest import signed_request

        body = json.dumps(
            {
                "kind": "validation",
                "qemu": "10.1.0",
                "os_image_hash": value,
                "compose_hash": "b" * 64,
            }
        ).encode()
        assert (
            signed_request(
                launching_client, validator_key, "POST", "/v1/cvm", body=body
            ).status_code
            == 422
        )

    def test_the_platform_key_tears_the_cvm_down(
        self, launching_client, validator_key, platform_key, approved, monkeypatch
    ):
        """DELETE carries no `kind`, so it is renter-scoped: letting the validation key call it
        would let a validator destroy a renter's CVM."""
        from conftest import signed_request

        signed_request(
            launching_client, validator_key, "POST", "/v1/cvm", body=self._body(approved)
        )
        monkeypatch.setattr(supervisor, "shutdown", lambda *a, **k: "guest powered off on request")
        monkeypatch.setattr(supervisor, "is_supervisor", lambda pid, vm_dir: False)

        response = signed_request(launching_client, platform_key, "DELETE", "/v1/cvm")

        assert response.status_code == 200
        assert response.json()["torn_down"] is True

    def test_a_second_operation_while_one_is_in_flight_is_refused(
        self, launching_client, validator_key, approved
    ):
        """The lock is acquired without blocking: the caller gets an answer, not a queue slot."""
        from conftest import signed_request

        launching_client.app.state.cvm_lock.acquire()
        try:
            response = signed_request(
                launching_client, validator_key, "POST", "/v1/cvm", body=self._body(approved)
            )
        finally:
            launching_client.app.state.cvm_lock.release()

        assert response.status_code == 409
        assert "already in progress" in response.json()["detail"]


class TestReadinessWithoutKeyscan:
    """A host with no `ssh-keyscan` still finishes its launch.

    `read_host_key` returns None both when the tool is missing and when the guest is not up
    yet, so a loop that cannot tell them apart polls for the whole launch timeout and then
    fails a node whose CVM booted fine. The tool is looked up once, before the loop.
    """

    @pytest.fixture
    def no_keyscan(self, monkeypatch):
        monkeypatch.setattr("cvmd.cvm.sshkey.keyscan_available", lambda: False)

        def unreachable(host, port):
            raise AssertionError("ssh-keyscan is absent; read_host_key must not be polled")

        monkeypatch.setattr("cvmd.cvm.sshkey.read_host_key", unreachable)
        monkeypatch.setattr("cvmd.cvm.sshkey.accepts_connection", lambda host, port: True)

    def test_it_falls_back_to_a_tcp_accept(self, manager, approved, spawned, no_keyscan):
        report = manager.create(kind="validation", triple=approved)

        assert report["state"] == "VALIDATION_RUNNING"
        assert report["ssh_host_key_fingerprint"] is None
        assert "ssh-keyscan" in report["note"]

    def test_the_launch_does_not_burn_its_whole_timeout(
        self, manager, approved, spawned, no_keyscan, monkeypatch
    ):
        """Sleeping at all here means the loop is waiting for a fingerprint that cannot come."""
        monkeypatch.setattr(
            "cvmd.cvm.manager.time.sleep",
            lambda _s: (_ for _ in ()).throw(AssertionError("polled instead of falling back")),
        )
        assert manager.create(kind="validation", triple=approved)["state"] == "VALIDATION_RUNNING"


class TestTeardown:
    def test_tearing_down_a_running_cvm_frees_the_node(
        self, manager, approved, spawned, guest_is_up, launch_config, monkeypatch
    ):
        report = manager.create(kind="validation", triple=approved)
        monkeypatch.setattr(supervisor, "shutdown", lambda *a, **k: "guest powered off on request")
        monkeypatch.setattr(supervisor, "is_supervisor", lambda pid, vm_dir: False)

        result = manager.destroy()

        assert result["torn_down"] is True
        assert result["instance_id"] == report["instance_id"]
        assert manager._store.state is NodeState.RECONCILING
        assert supervisor.vm_directories(launch_config.run_dir) == []

    def test_a_relaunch_after_teardown_succeeds(
        self, manager, approved, spawned, guest_is_up, monkeypatch
    ):
        """The point of removing the directory: the node is reusable without hand-cleanup."""
        manager.create(kind="validation", triple=approved)
        monkeypatch.setattr(supervisor, "shutdown", lambda *a, **k: "stopped")
        monkeypatch.setattr(supervisor, "is_supervisor", lambda pid, vm_dir: False)
        manager.destroy()

        monkeypatch.setattr(supervisor, "is_supervisor", lambda pid, vm_dir: pid == 424242)
        assert manager.create(kind="validation", triple=approved)["state"] == "VALIDATION_RUNNING"

    def test_tearing_down_an_idle_node_is_not_an_error(self, manager):
        """A platform that timed out mid-teardown has to be able to repeat the call."""
        result = manager.destroy()
        assert result == {"torn_down": False, "reason": "this node is running no CVM"}

    def test_a_stray_directory_with_a_live_guest_is_not_removed(
        self, manager, launch_config, monkeypatch
    ):
        """ "No record" usually means "the record went missing", not "nothing is running".

        Removing the disk from under a live QEMU would report a teardown that did not happen
        and leave the node's GPUs held by a process nothing can name — the record that carried
        its pid is exactly what is gone.
        """
        stray = launch_config.run_dir / "unrecorded"
        stray.mkdir(parents=True)
        (stray / supervisor.DISK_IMAGE).write_bytes(b"")
        monkeypatch.setattr(supervisor, "running_cvms", lambda: [(4242, "qemu-system-x86_64")])

        with pytest.raises(LaunchFailure) as raised:
            manager.destroy()

        assert raised.value.status == 409
        assert "4242" in raised.value.reason
        assert (stray / supervisor.DISK_IMAGE).exists()
        assert manager._store.state is NodeState.FAILED

    def test_a_stray_directory_with_no_guest_is_removed(self, manager, launch_config, monkeypatch):
        stray = launch_config.run_dir / "unrecorded"
        stray.mkdir(parents=True)
        (stray / supervisor.DISK_IMAGE).write_bytes(b"")
        monkeypatch.setattr(supervisor, "running_cvms", list)

        result = manager.destroy()

        assert result["torn_down"] is True
        assert not stray.exists()
        assert manager._store.state is NodeState.RECONCILING

    def test_a_graceful_shutdown_that_raises_still_signals_the_group(self, monkeypatch):
        """dstack reads runtime.json outside its own try block, so it can raise here.

        Letting that propagate would skip the signal ladder entirely — the guest would never be
        signalled at all, which is the one outcome `shutdown` exists to prevent.
        """
        signalled: list[int] = []

        class Raising:
            @staticmethod
            def shutdown_instance(vm_dir, timeout, force):
                raise KeyError("cid")

        monkeypatch.setattr(
            supervisor,
            "shutdown_by_signal",
            lambda pid, *, timeout: signalled.append(pid) or "process group stopped by SIGTERM",
        )

        detail = supervisor.shutdown(Raising, Path("/nonexistent"), 4242, timeout=1)

        assert signalled == [4242]
        assert detail == "process group stopped by SIGTERM"

    def test_a_cvm_that_will_not_stop_fails_the_node(
        self, manager, approved, spawned, guest_is_up, monkeypatch
    ):
        """Never report a teardown that did not happen — that is the DAH-2577 acceptance bar."""
        manager.create(kind="validation", triple=approved)
        monkeypatch.setattr(supervisor, "shutdown", lambda *a, **k: "still present after SIGKILL")

        with pytest.raises(LaunchFailure) as raised:
            manager.destroy()

        assert raised.value.status == 500
        assert manager._store.state is NodeState.FAILED
