"""The renter launch path (DAH-2580): what authorizes it, and what it refuses.

Same seam as `test_launch.py` — `supervisor.spawn` is stubbed, everything up to and including
the measurement gate is real dstack.py writing a real VM directory. The point of this module is
the one thing that differs from a validation launch: the compose is not in the catalog, so it
has to be authorized some other way. Three mechanisms carry that, and each has a test here.

  the catalog still pins the rest       an image or QEMU build it does not carry is 422
  the platform key is the only sender   a validator-signed renter create is 403
  the measurement decides               a compose that does not hash to the requested
                                        `compose_hash` never reaches QEMU

The third is the load-bearing one, because it is the only one that still holds when the host
itself is the adversary. It is checked from both directions: a request whose hash is wrong for
its compose, and a request whose *flags* are wrong for its hash.
"""

import json

import pytest
from conftest import OS_IMAGE_HASH, QEMU_FALLBACK, signed_request
from cvmd.config import Config
from cvmd.cvm import measure
from cvmd.cvm.manager import LaunchFailure
from cvmd.cvm.renter import RenterOrder
from cvmd.state.machine import NodeState

CUSTOMER_COMPOSE = (
    "services:\n"
    "  workload:\n"
    "    image: alpine:3.20\n"
    "    command: sleep infinity\n"
    "  lium-attest-agent:\n"
    "    image: daturaai/lium-attest-agent@sha256:" + ("e" * 64) + "\n"
    "    read_only: true\n"
)


def measured_hash(dstack, tmp_path, image_dir, compose_text, **flags) -> str:
    """The `compose_hash` this host would produce for that compose and those flags.

    Derived by preparing a throwaway VM directory with the real launcher and hashing what it
    wrote, exactly as the catalog fixture does. A hardcoded constant would make these tests
    assert that two literals are equal; deriving it makes them assert that the gate works.
    """
    from cvmd.catalog import Artifact
    from cvmd.config import LaunchConfig
    from cvmd.cvm import renter
    from cvmd.dstack.plan import setup_namespace

    staged = renter.stage(
        RenterOrder(
            qemu=QEMU_FALLBACK,
            os_image_hash=OS_IMAGE_HASH,
            compose_hash="0" * 64,
            compose=compose_text,
            **flags,
        ),
        base=Artifact(
            id="probe",
            kind="validation",
            qemu=QEMU_FALLBACK,
            os_image_hash=OS_IMAGE_HASH,
            compose_hash="0" * 64,
            os_image_path=image_dir,
            compose_path=image_dir / "unused",
        ),
        directory=tmp_path / "probe-stage",
    )
    probe_dir = tmp_path / "probe" / "renter"
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    dstack.DStackManager().setup_instance(
        setup_namespace(
            artifact=staged,
            launch=LaunchConfig(vcpus=2, memory="2G", disk="20G"),
            vm_dir=probe_dir,
        )
    )
    return measure.compose_hash(probe_dir)


@pytest.fixture
def renter_hash(dstack, tmp_path, image_dir) -> str:
    return measured_hash(dstack, tmp_path, image_dir, CUSTOMER_COMPOSE)


@pytest.fixture
def order(renter_hash) -> RenterOrder:
    return RenterOrder(
        qemu=QEMU_FALLBACK,
        os_image_hash=OS_IMAGE_HASH,
        compose_hash=renter_hash,
        compose=CUSTOMER_COMPOSE,
        rental_id="rental-abc123",
    )


class TestASuccessfulRenterLaunch:
    def test_it_reports_the_rental_and_the_derived_measurement(
        self, manager, order, spawned, guest_is_up
    ):
        report = manager.create_renter(order)

        assert report["kind"] == "renter"
        assert report["rental_id"] == "rental-abc123"
        assert report["measurements"]["compose_hash"] == order.compose_hash
        assert report["measurements"]["os_image_hash"] == OS_IMAGE_HASH
        assert report["ssh_host_key_fingerprint"] == "SHA256:testfingerprint"
        assert report["state"] == "RENTER_RUNNING"

    def test_the_report_names_the_catalog_entry_the_order_was_built_on(
        self, manager, order, spawned, guest_is_up
    ):
        """A renter artifact id is derived, not invented: it says which approved image and
        QEMU build the customer's compose ran on.

        That the base entry's own kind is `validation` is the point rather than an oversight —
        the manifest is a cross product, so every approved image appears under every kind and
        filtering the base on kind would narrow nothing. See `catalog.resolve_base`.
        """
        report = manager.create_renter(order)
        assert report["artifact_id"] == "renter:validation-test"

    def test_the_customers_compose_does_not_outlive_the_launch(
        self, manager, order, launch_config, spawned, guest_is_up
    ):
        """Staging is an input to `setup_instance` and nothing else reads it afterwards.

        `run_instance` works from `app-compose.json` inside the VM directory, so once the
        measurement has passed there is no reason for a customer's compose to stay readable in
        a plain directory on a host they do not control.
        """
        manager.create_renter(order)

        staged = list(launch_config.renter_dir.iterdir())
        assert staged == []

    def test_the_state_document_records_a_renter_cvm(self, manager, order, spawned, guest_is_up):
        manager.create_renter(order)
        assert manager._store.state == NodeState.RENTER_RUNNING


class TestTheMeasurementGate:
    """The check that still holds when the host is the adversary."""

    def test_a_compose_that_does_not_hash_to_the_request_never_reaches_qemu(
        self, manager, order, spawned
    ):
        from dataclasses import replace

        tampered = replace(order, compose=CUSTOMER_COMPOSE + "# an extra line\n")

        with pytest.raises(LaunchFailure) as raised:
            manager.create_renter(tampered)

        assert raised.value.status == 409
        assert "compose_hash" in raised.value.reason
        assert spawned == []

    def test_a_flag_the_request_did_not_account_for_is_the_same_refusal(
        self, manager, order, spawned
    ):
        """`enable_logs` is folded into the measured file, so turning it on changes the hash.

        This is why the flags are on the request at all rather than defaulted by the host: a
        host that chose one itself would produce a CVM measuring as something the platform never
        predicted, and the validator would reject a node that had done nothing wrong.
        """
        from dataclasses import replace

        with pytest.raises(LaunchFailure) as raised:
            manager.create_renter(replace(order, enable_logs=True))

        assert raised.value.status == 409
        assert "compose_hash" in raised.value.reason

    def test_a_refused_launch_leaves_no_staging_and_no_vm_directory(
        self, manager, order, launch_config, spawned
    ):
        from dataclasses import replace

        with pytest.raises(LaunchFailure):
            manager.create_renter(replace(order, compose_hash="f" * 64))

        assert list(launch_config.renter_dir.iterdir()) == []
        assert not launch_config.run_dir.exists() or list(launch_config.run_dir.iterdir()) == []


class TestWhatTheCatalogStillDecides:
    def test_an_unapproved_os_image_is_refused_by_name(self, manager, order, spawned):
        from dataclasses import replace

        with pytest.raises(LaunchFailure) as raised:
            manager.create_renter(replace(order, os_image_hash="9" * 64))

        assert raised.value.status == 422
        assert "os_image_hash" in raised.value.reason
        assert spawned == []

    def test_an_unapproved_qemu_build_is_refused_by_name(self, manager, order, spawned):
        from dataclasses import replace

        with pytest.raises(LaunchFailure) as raised:
            manager.create_renter(replace(order, qemu="99.9.9"))

        assert raised.value.status == 422
        assert "qemu" in raised.value.reason

    def test_a_host_with_no_catalog_says_so_rather_than_launching(
        self, launch_config, state_dir, order, monkeypatch
    ):
        from cvmd.catalog import CatalogConfig, CatalogStore
        from cvmd.cvm.instance import InstanceStore
        from cvmd.cvm.manager import CvmManager
        from cvmd.cvm.switching import SwitchStore
        from cvmd.state.store import StateStore

        monkeypatch.setattr(
            "cvmd.cvm.measure.qemu_version", lambda _dstack: QEMU_FALLBACK, raising=True
        )
        manager = CvmManager(
            config=launch_config,
            catalog_store=CatalogStore(CatalogConfig()),
            store=StateStore(state_dir),
            instances=InstanceStore(state_dir),
            switches=SwitchStore(state_dir),
        )
        with pytest.raises(LaunchFailure) as raised:
            manager.create_renter(order)

        # 503, not 422: retrying later may work, because the catalog is missing rather than
        # holding a different answer.
        assert raised.value.status == 503

    def test_a_host_with_no_renter_directory_refuses_before_staging_anything(
        self, catalog_store, launch_config, state_dir, order, monkeypatch
    ):
        from dataclasses import replace as dc_replace

        from cvmd.cvm.instance import InstanceStore
        from cvmd.cvm.manager import CvmManager
        from cvmd.cvm.switching import SwitchStore
        from cvmd.state.store import StateStore

        monkeypatch.setattr(
            "cvmd.cvm.measure.qemu_version", lambda _dstack: QEMU_FALLBACK, raising=True
        )
        manager = CvmManager(
            config=dc_replace(launch_config, renter_dir=None),
            catalog_store=catalog_store,
            store=StateStore(state_dir),
            instances=InstanceStore(state_dir),
            switches=SwitchStore(state_dir),
        )
        with pytest.raises(LaunchFailure) as raised:
            manager.create_renter(order)

        assert raised.value.status == 503
        assert "renter_dir" in raised.value.reason


class TestStaleStagingIsSweptAtStartup:
    def test_reconciliation_removes_a_directory_a_killed_launch_left(self, manager, launch_config):
        """A launch discards its own staging directory; only a hard kill can leave one.

        Sweeping it at reconciliation is safe precisely because nothing reads it after the
        measurement — including the CVM this daemon may have just come back underneath.
        """
        orphan = launch_config.renter_dir / "killed-partway"
        orphan.mkdir(parents=True)
        (orphan / "compose.yml").write_text("services: {}\n")

        manager.reconcile()

        assert not orphan.exists()


class TestOverHttp:
    @pytest.fixture
    def launching_client(self, clients_file, state_dir, launch_config, catalog_store, monkeypatch):
        from cvmd.app import create_app
        from fastapi.testclient import TestClient

        monkeypatch.setattr(
            "cvmd.cvm.measure.qemu_version", lambda _dstack: QEMU_FALLBACK, raising=True
        )
        config = Config(
            authorized_clients=clients_file,
            state_dir=state_dir,
            launch=launch_config,
            catalog=catalog_store.config,
        )
        with TestClient(create_app(config), raise_server_exceptions=False) as client:
            yield client

    def body(self, order: RenterOrder, **overrides) -> bytes:
        payload = {
            "kind": "renter",
            "qemu": order.qemu,
            "os_image_hash": order.os_image_hash,
            "compose_hash": order.compose_hash,
            "compose": order.compose,
            "rental_id": order.rental_id,
        }
        payload.update(overrides)
        return json.dumps(payload).encode()

    def test_the_platform_key_launches_and_gets_the_report(
        self, launching_client, platform_key, order, spawned, guest_is_up
    ):
        response = signed_request(
            launching_client, platform_key, "POST", "/v1/cvm", body=self.body(order)
        )

        assert response.status_code == 201
        report = response.json()
        assert report["kind"] == "renter"
        assert report["rental_id"] == "rental-abc123"
        assert report["measurements"]["compose_hash"] == order.compose_hash

    def test_the_validator_key_cannot_ask_for_one_at_all(
        self, launching_client, validator_key, order, spawned
    ):
        """FR: the validator never triggers renter provisioning.

        403 before the body is validated and before anything is staged — the scope is decided
        from `kind`, which is the same field the handler dispatches on.
        """
        response = signed_request(
            launching_client, validator_key, "POST", "/v1/cvm", body=self.body(order)
        )

        assert response.status_code == 403
        assert spawned == []

    def test_a_body_with_no_compose_is_422(self, launching_client, platform_key, order):
        payload = json.loads(self.body(order))
        del payload["compose"]
        response = signed_request(
            launching_client, platform_key, "POST", "/v1/cvm", body=json.dumps(payload).encode()
        )
        assert response.status_code == 422

    def test_an_unknown_field_is_422_rather_than_quietly_ignored(
        self, launching_client, platform_key, order, spawned
    ):
        """An unknown field means the sender is describing a CVM this host cannot build.

        Accepting it would launch something that differs from what the platform recorded and
        what the validator will expect — and the difference would surface as a failed
        attestation on a node that did exactly what it was told.
        """
        response = signed_request(
            launching_client,
            platform_key,
            "POST",
            "/v1/cvm",
            body=self.body(order, gpu_count=8),
        )

        assert response.status_code == 422
        assert spawned == []

    def test_a_compose_larger_than_the_cap_is_422(self, launching_client, platform_key, order):
        response = signed_request(
            launching_client,
            platform_key,
            "POST",
            "/v1/cvm",
            body=self.body(order, compose="x" * (32 * 1024 + 1)),
        )
        assert response.status_code == 422

    def test_the_platform_key_tears_its_own_renter_cvm_down(
        self, launching_client, platform_key, order, spawned, guest_is_up, host_is_free
    ):
        created = signed_request(
            launching_client, platform_key, "POST", "/v1/cvm", body=self.body(order)
        )
        assert created.status_code == 201

        spawned.clear()  # is_supervisor reports the pid gone once no spawn is on record
        destroyed = signed_request(launching_client, platform_key, "DELETE", "/v1/cvm")

        assert destroyed.status_code == 200
        assert destroyed.json()["torn_down"] is True
        state = signed_request(launching_client, platform_key, "GET", "/v1/state").json()
        assert state["cvm"] is None
        assert state["state"] == "RECONCILING"
