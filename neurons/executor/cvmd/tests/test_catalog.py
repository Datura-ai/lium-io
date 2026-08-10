"""The catalog decides what this host may launch, so every way of misreading it is a refusal.

Split the way the code is: what a manifest has to prove before it is believed
(`TestVerification`), what it has to say before it is understood (`TestDecoding`), what happens
to the host's copy when a new one arrives (`TestInstalling`), and how a request is matched
against the result (`TestResolving`).
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import manifest_entry, manifest_payload, sign_manifest
from cvmd.catalog import (
    CatalogConfig,
    CatalogError,
    CatalogStore,
    TripleNotFound,
    parse_manifest,
    resolve,
)

QEMU = "10.1.0"
OS_IMAGE = "a" * 64
COMPOSE = "b" * 64


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    """Two staged OS images, as day-zero Ansible would leave them."""
    root = tmp_path / "images"
    for name in ("dstack-nvidia-0.5.11", "dstack-nvidia-0.5.12"):
        image = root / name
        image.mkdir(parents=True)
        (image / "digest.txt").write_text(OS_IMAGE + "\n")
    return root


@pytest.fixture
def store(tmp_path: Path, images_dir: Path, catalog_signer) -> CatalogStore:
    return CatalogStore(
        CatalogConfig(
            cache_path=tmp_path / "state" / "catalog" / "manifest.json",
            signer=catalog_signer.ss58_address,
            images_dir=images_dir,
            materialize_dir=tmp_path / "state" / "catalog" / "artifacts",
            manifest_url="https://example.invalid/manifest",
        )
    )


def signed(catalog_signer, *entries, **kwargs) -> bytes:
    return sign_manifest(manifest_payload(*entries, **kwargs), catalog_signer)


class TestVerification:
    """A manifest is bytes from the network until every one of these has passed."""

    def test_a_manifest_signed_by_the_configured_key_is_accepted(self, catalog_signer):
        raw = signed(catalog_signer, manifest_entry())
        manifest = parse_manifest(raw, signer=catalog_signer.ss58_address)

        assert manifest.serial == 1
        assert [entry.id for entry in manifest.entries] == ["validation-v3"]

    def test_one_flipped_byte_is_refused(self, catalog_signer):
        """The tamper test the task names. The payload is signed as text, so any edit shows."""
        raw = signed(catalog_signer, manifest_entry())
        envelope = json.loads(raw)
        envelope["payload"] = envelope["payload"].replace('"serial": 1', '"serial": 2')

        with pytest.raises(CatalogError, match="not signed by"):
            parse_manifest(json.dumps(envelope).encode(), signer=catalog_signer.ss58_address)

    def test_a_manifest_signed_by_another_key_is_refused(self, catalog_signer, other_signer):
        raw = sign_manifest(manifest_payload(manifest_entry()), other_signer)

        with pytest.raises(CatalogError, match="only trusts"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)

    def test_the_signer_field_cannot_choose_who_verifies_it(self, catalog_signer, other_signer):
        """The trap this whole design exists to avoid.

        A document that names its own signer and is checked against that name proves only that
        somebody owns a key — anybody with any key can produce one. The configured ss58 is the
        only thing the signature is ever checked against, so a forged manifest that also
        rewrites `signer` is refused before the signature is even looked at.
        """
        payload = manifest_payload(manifest_entry())
        raw = sign_manifest(payload, other_signer, signer=other_signer.ss58_address)

        with pytest.raises(CatalogError, match="only trusts"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)

    def test_an_unsigned_manifest_is_refused(self, catalog_signer):
        """A bare catalog document — what a host would happily have read before DAH-2578."""
        raw = manifest_payload(manifest_entry()).encode()

        with pytest.raises(CatalogError, match="schema"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)

    def test_an_empty_signature_is_refused(self, catalog_signer):
        raw = json.loads(signed(catalog_signer, manifest_entry()))
        raw["signature"] = ""

        with pytest.raises(CatalogError, match="`signature` must be a non-empty string"):
            parse_manifest(json.dumps(raw).encode(), signer=catalog_signer.ss58_address)

    def test_a_host_with_no_configured_signer_trusts_nothing(self, catalog_signer):
        raw = signed(catalog_signer, manifest_entry())

        with pytest.raises(CatalogError, match="no configured manifest signer"):
            parse_manifest(raw, signer="")

    def test_an_expired_manifest_is_refused(self, catalog_signer):
        """Expiry is what stops a revocation being defeated by never delivering the next one."""
        issued = datetime.now(UTC) - timedelta(hours=2)
        raw = signed(catalog_signer, manifest_entry(), issued_at=issued, ttl_seconds=3600)

        with pytest.raises(CatalogError, match="expired at"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)

    def test_an_expired_manifest_can_still_be_read_to_say_so(self, catalog_signer):
        """`/v1/catalog` has to be able to report *why* a node will not launch."""
        issued = datetime.now(UTC) - timedelta(hours=2)
        raw = signed(catalog_signer, manifest_entry(), issued_at=issued, ttl_seconds=3600)

        manifest = parse_manifest(raw, signer=catalog_signer.ss58_address, require_fresh=False)
        assert manifest.is_expired()

    def test_a_manifest_issued_far_in_the_future_is_refused(self, catalog_signer):
        raw = signed(
            catalog_signer, manifest_entry(), issued_at=datetime.now(UTC) + timedelta(days=1)
        )

        with pytest.raises(CatalogError, match="further ahead of this host's clock"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)

    def test_an_expiry_before_its_issue_is_refused(self, catalog_signer):
        raw = signed(catalog_signer, manifest_entry(), ttl_seconds=-60)

        with pytest.raises(CatalogError, match="was never valid"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)

    def test_a_naive_timestamp_is_refused(self, catalog_signer):
        payload = json.loads(manifest_payload(manifest_entry()))
        payload["expires_at"] = "2099-01-01T00:00:00"
        raw = sign_manifest(json.dumps(payload), catalog_signer)

        with pytest.raises(CatalogError, match="must carry a timezone"):
            parse_manifest(raw, signer=catalog_signer.ss58_address)


class TestDecoding:
    def test_defaults_match_lium_cvm_sh(self, catalog_signer):
        """An entry that says nothing about flags gets what the shell path uses by default."""
        raw = signed(catalog_signer, manifest_entry())
        entry = parse_manifest(raw, signer=catalog_signer.ss58_address).entries[0]

        assert entry.local_key_provider is True
        assert entry.enable_logs is False
        assert entry.enable_sysinfo is False

    def test_an_unknown_schema_is_refused_rather_than_guessed(self, catalog_signer):
        envelope = json.loads(signed(catalog_signer, manifest_entry()))
        envelope["schema"] = "lium-cvm-catalog/99"

        with pytest.raises(CatalogError, match="is not the supported"):
            parse_manifest(json.dumps(envelope).encode(), signer=catalog_signer.ss58_address)

    def test_an_unknown_payload_version_is_refused(self, catalog_signer):
        payload = json.loads(manifest_payload(manifest_entry()))
        payload["version"] = 99

        with pytest.raises(CatalogError, match="is not the supported 1"):
            parse_manifest(
                sign_manifest(json.dumps(payload), catalog_signer),
                signer=catalog_signer.ss58_address,
            )

    def test_a_missing_field_is_refused(self, catalog_signer):
        broken = manifest_entry()
        del broken["compose"]

        with pytest.raises(CatalogError, match="compose"):
            parse_manifest(signed(catalog_signer, broken), signer=catalog_signer.ss58_address)

    @pytest.mark.parametrize(
        "value",
        ["A" * 64, "sha256:" + "a" * 64, "a" * 63, "z" * 64],
        ids=["uppercase", "prefixed", "too-short", "not-hex"],
    )
    def test_a_hash_that_could_never_match_is_refused(self, catalog_signer, value):
        """cvmd compares against lowercase hex it computes itself.

        An entry written any other way would never match, and the launch would fail as "not
        approved" — which sends an operator looking at the wrong thing entirely.
        """
        with pytest.raises(CatalogError, match="64 lowercase hex"):
            parse_manifest(
                signed(catalog_signer, manifest_entry(compose_hash=value)),
                signer=catalog_signer.ss58_address,
            )

    @pytest.mark.parametrize(
        "name", ["../../etc", "a/b", "..", "."], ids=["traversal", "nested", "parent", "self"]
    )
    def test_an_image_name_that_escapes_the_image_directory_is_refused(self, catalog_signer, name):
        """`os_image_name` is joined onto a host path, and the joined directory's digest.txt is
        what the measurement gate compares against. A name that escapes chooses that file."""
        with pytest.raises(CatalogError, match="single directory name"):
            parse_manifest(
                signed(catalog_signer, manifest_entry(os_image_name=name)),
                signer=catalog_signer.ss58_address,
            )

    def test_a_repeated_id_is_refused(self, catalog_signer):
        with pytest.raises(CatalogError, match="repeats the id"):
            parse_manifest(
                signed(catalog_signer, manifest_entry(), manifest_entry()),
                signer=catalog_signer.ss58_address,
            )

    def test_an_entry_below_the_manifests_own_floor_is_refused(self, catalog_signer):
        """The backend excludes these when it builds a manifest. This is the check that says so
        if the ratchet and the list it produced ever disagree."""
        entry = manifest_entry(versions={"os_image": 2, "qemu": 1, "compose": 3})

        with pytest.raises(CatalogError, match="below this manifest's own floor"):
            parse_manifest(
                signed(catalog_signer, entry, floors={"os_image": 2, "qemu": 3, "compose": 3}),
                signer=catalog_signer.ss58_address,
            )

    def test_a_boolean_version_is_not_an_integer(self, catalog_signer):
        """`True` compares as 1 against a floor and ratchets nothing."""
        entry = manifest_entry(versions={"os_image": True, "qemu": 1, "compose": 1})

        with pytest.raises(CatalogError, match="must be an integer"):
            parse_manifest(signed(catalog_signer, entry), signer=catalog_signer.ss58_address)

    def test_a_component_this_cvmd_cannot_floor_is_refused(self, catalog_signer):
        entry = manifest_entry(
            versions={"os_image": 1, "qemu": 1, "compose": 1, "firmware": 1},
        )

        with pytest.raises(CatalogError, match="does not ratchet"):
            parse_manifest(signed(catalog_signer, entry), signer=catalog_signer.ss58_address)


class TestInstalling:
    def test_installing_makes_the_manifest_this_hosts_catalog(self, store, catalog_signer):
        store.install(signed(catalog_signer, manifest_entry()), source="test")

        assert store.current().serial == 1
        assert store.config.cache_path.is_file()

    def test_the_measured_inputs_are_written_from_the_manifest(self, store, catalog_signer):
        """The compose and the scripts travel as content, so the host cannot supply its own."""
        entry = manifest_entry(
            compose="services:\n  a:\n    image: x\n",
            init_script="#!/bin/sh\necho init\n",
            pre_launch_script="#!/bin/sh\necho pre\n",
        )
        store.install(signed(catalog_signer, entry), source="test")

        artifact = store.artifacts()[0]
        assert artifact.compose_path.read_text() == "services:\n  a:\n    image: x\n"
        assert artifact.init_script.read_text() == "#!/bin/sh\necho init\n"
        assert artifact.pre_launch_script.read_text() == "#!/bin/sh\necho pre\n"

    def test_an_edited_compose_is_put_back(self, store, catalog_signer):
        """An operator who adds a service to the materialized compose would be launching a stack
        the platform never approved. The measurement gate would catch it afterwards; this puts
        the signed content back before the launch is even prepared."""
        store.install(signed(catalog_signer, manifest_entry()), source="test")
        compose_path = store.artifacts()[0].compose_path
        compose_path.write_text("services:\n  backdoor:\n    image: evil\n")

        assert store.artifacts()[0].compose_path.read_text() == manifest_entry()["compose"]

    def test_a_script_dropped_from_the_manifest_is_removed(self, store, catalog_signer):
        store.install(
            signed(catalog_signer, manifest_entry(init_script="#!/bin/sh\n")), source="test"
        )
        init_path = store.artifacts()[0].init_script
        assert init_path.exists()

        store.install(signed(catalog_signer, manifest_entry(), serial=2), source="test")

        assert store.artifacts()[0].init_script is None
        assert not init_path.exists()

    def test_an_image_the_host_has_not_staged_is_a_named_refusal(self, store, catalog_signer):
        store.install(
            signed(catalog_signer, manifest_entry(os_image_name="dstack-nvidia-9.9.9")),
            source="test",
        )

        with pytest.raises(CatalogError, match="not staged on this host"):
            store.artifacts()

    def test_an_older_serial_is_refused_as_a_rollback(self, store, catalog_signer):
        """A signed manifest from before a revocation is still validly signed."""
        store.install(signed(catalog_signer, manifest_entry(), serial=7), source="test")

        with pytest.raises(CatalogError, match="refused as a rollback"):
            store.install(signed(catalog_signer, manifest_entry(), serial=6), source="test")

        assert store.current().serial == 7

    def test_a_lowered_floor_is_refused(self, store, catalog_signer):
        entry = manifest_entry(versions={"os_image": 5, "qemu": 5, "compose": 5})
        store.install(
            signed(catalog_signer, entry, floors={"os_image": 4, "qemu": 1, "compose": 1}),
            source="test",
        )

        with pytest.raises(CatalogError, match="floor only goes up"):
            store.install(
                signed(
                    catalog_signer,
                    entry,
                    serial=2,
                    floors={"os_image": 3, "qemu": 1, "compose": 1},
                ),
                source="test",
            )

    def test_a_refused_manifest_leaves_the_previous_one_in_force(self, store, catalog_signer):
        """The task's tamper case, at the level a host experiences it: the node keeps working."""
        store.install(signed(catalog_signer, manifest_entry()), source="test")
        envelope = json.loads(signed(catalog_signer, manifest_entry(), serial=2))
        envelope["payload"] = envelope["payload"].replace("validation-v3", "validation-v4")

        with pytest.raises(CatalogError):
            store.install(json.dumps(envelope).encode(), source="test")

        assert [entry.id for entry in store.current().entries] == ["validation-v3"]

    def test_a_revoked_artifact_stops_being_launchable(self, store, catalog_signer):
        """Revocation as a host sees it: the next manifest simply does not carry the entry."""
        store.install(
            signed(catalog_signer, manifest_entry(), manifest_entry(id="renter-v1", kind="renter")),
            source="test",
        )
        assert len(store.artifacts()) == 2

        store.install(signed(catalog_signer, manifest_entry(), serial=2), source="test")

        assert [a.id for a in store.artifacts()] == ["validation-v3"]

    def test_a_host_with_no_manifest_launches_nothing(self, store):
        with pytest.raises(CatalogError, match="holds no catalog manifest"):
            store.artifacts()

    def test_a_host_with_no_signer_says_which_setting_is_missing(self, tmp_path):
        store = CatalogStore(CatalogConfig(cache_path=tmp_path / "m.json"))

        with pytest.raises(CatalogError, match="signer"):
            store.artifacts()

    def test_the_seed_is_adopted_when_the_host_has_nothing(
        self, tmp_path, images_dir, catalog_signer
    ):
        seed = tmp_path / "seed.json"
        seed.write_bytes(signed(catalog_signer, manifest_entry(), serial=3))
        store = CatalogStore(
            CatalogConfig(
                cache_path=tmp_path / "cache.json",
                seed_path=seed,
                signer=catalog_signer.ss58_address,
                images_dir=images_dir,
                materialize_dir=tmp_path / "artifacts",
            )
        )

        assert store.install_seed_if_newer().serial == 3

    def test_a_stale_seed_is_not_adopted_and_is_not_fatal(
        self, tmp_path, images_dir, catalog_signer
    ):
        """A seed left behind after the backend moved on reads exactly like a rollback."""
        seed = tmp_path / "seed.json"
        seed.write_bytes(signed(catalog_signer, manifest_entry(), serial=1))
        store = CatalogStore(
            CatalogConfig(
                cache_path=tmp_path / "cache.json",
                seed_path=seed,
                signer=catalog_signer.ss58_address,
                images_dir=images_dir,
                materialize_dir=tmp_path / "artifacts",
            )
        )
        store.install(signed(catalog_signer, manifest_entry(), serial=5), source="test")

        assert store.install_seed_if_newer() is None
        assert store.current().serial == 5

    def test_describe_reports_a_broken_catalog_rather_than_raising(self, store):
        described = store.describe()

        assert described["usable"] is False
        assert "holds no catalog manifest" in described["error"]

    def test_describe_reports_the_manifest_in_force(self, store, catalog_signer):
        store.install(signed(catalog_signer, manifest_entry(), serial=4), source="test")

        described = store.describe()

        assert described["usable"] is True
        assert described["manifest"]["serial"] == 4
        assert described["manifest"]["entries"][0]["id"] == "validation-v3"


class TestResolving:
    @pytest.fixture
    def artifacts(self, store, catalog_signer):
        store.install(
            signed(
                catalog_signer,
                manifest_entry(qemu=QEMU, os_image_hash=OS_IMAGE, compose_hash=COMPOSE),
                manifest_entry(
                    id="renter-v1",
                    kind="renter",
                    qemu=QEMU,
                    os_image_hash=OS_IMAGE,
                    compose_hash="c" * 64,
                ),
            ),
            source="test",
        )
        return store.artifacts()

    def test_the_matching_triple_resolves(self, artifacts):
        found = resolve(
            artifacts, kind="validation", qemu=QEMU, os_image_hash=OS_IMAGE, compose_hash=COMPOSE
        )
        assert found.id == "validation-v3"

    def test_kind_is_part_of_the_match(self, artifacts):
        """The renter entry shares a QEMU build and image; only its compose hash differs."""
        found = resolve(
            artifacts, kind="renter", qemu=QEMU, os_image_hash=OS_IMAGE, compose_hash="c" * 64
        )
        assert found.id == "renter-v1"

    @pytest.mark.parametrize(
        ("field", "value"),
        [("qemu", "9.2.1"), ("os_image_hash", "f" * 64), ("compose_hash", "e" * 64)],
    )
    def test_an_unapproved_component_is_named_in_the_refusal(self, artifacts, field, value):
        """ "Not found" is not actionable; "this compose hash is not approved" is."""
        request = {"qemu": QEMU, "os_image_hash": OS_IMAGE, "compose_hash": COMPOSE}
        request[field] = value

        with pytest.raises(TripleNotFound) as raised:
            resolve(artifacts, kind="validation", **request)

        message = str(raised.value)
        assert field in message
        assert value in message
        # The refusal also says what *is* approved, so the fix does not need a second round trip.
        assert getattr(artifacts[0], field) in message

    def test_an_unknown_kind_lists_the_kinds_there_are(self, artifacts):
        with pytest.raises(TripleNotFound, match="validation, renter|renter, validation"):
            resolve(
                artifacts, kind="sandbox", qemu=QEMU, os_image_hash=OS_IMAGE, compose_hash=COMPOSE
            )

    def test_two_entries_with_the_same_triple_are_refused(self, store, catalog_signer):
        """A request that could resolve to either entry has no single answer."""
        store.install(
            signed(catalog_signer, manifest_entry(), manifest_entry(id="twin")), source="test"
        )

        with pytest.raises(CatalogError, match="pin the same kind and triple"):
            store.artifacts()
