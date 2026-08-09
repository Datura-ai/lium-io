"""The catalog decides what this host may launch, so every way of misreading it is a refusal."""

import json
from pathlib import Path

import pytest
from cvmd.catalog import CatalogError, TripleNotFound, load_catalog, resolve

QEMU = "10.1.0"
OS_IMAGE = "a" * 64
COMPOSE = "b" * 64


def entry(**overrides) -> dict:
    base = {
        "id": "validation-v3",
        "kind": "validation",
        "qemu": QEMU,
        "os_image_hash": OS_IMAGE,
        "compose_hash": COMPOSE,
        "os_image_path": "/opt/dstack/images/dstack-nvidia-0.5.11",
        "compose_path": "/etc/cvmd/composes/validation-v3.yml",
    }
    base.update(overrides)
    return base


def write(path: Path, *entries, version: int = 1) -> Path:
    path.write_text(json.dumps({"version": version, "artifacts": list(entries)}))
    return path


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    return tmp_path / "catalog.json"


class TestLoading:
    def test_a_well_formed_catalog_loads(self, catalog_path):
        artifacts = load_catalog(write(catalog_path, entry()))
        assert [a.id for a in artifacts] == ["validation-v3"]
        assert artifacts[0].triple == (QEMU, OS_IMAGE, COMPOSE)

    def test_defaults_match_lium_cvm_sh(self, catalog_path):
        """An entry that says nothing about flags gets what the shell path uses by default."""
        artifact = load_catalog(write(catalog_path, entry()))[0]
        assert artifact.local_key_provider is True
        assert artifact.enable_logs is False
        assert artifact.enable_sysinfo is False

    def test_a_missing_catalog_is_an_error_not_an_empty_list(self, catalog_path):
        """An absent catalog must refuse every launch, never approve nothing quietly."""
        with pytest.raises(CatalogError, match="no catalog at"):
            load_catalog(catalog_path)

    def test_unparseable_json_is_refused(self, catalog_path):
        catalog_path.write_text("{not json")
        with pytest.raises(CatalogError, match="cannot read catalog"):
            load_catalog(catalog_path)

    def test_an_unknown_version_is_refused_rather_than_guessed(self, catalog_path):
        with pytest.raises(CatalogError, match="is not the supported"):
            load_catalog(write(catalog_path, entry(), version=99))

    def test_a_missing_field_is_refused(self, catalog_path):
        broken = entry()
        del broken["compose_path"]
        with pytest.raises(CatalogError, match="compose_path"):
            load_catalog(write(catalog_path, broken))

    @pytest.mark.parametrize(
        "value",
        ["A" * 64, "sha256:" + "a" * 64, "a" * 63, "z" * 64],
        ids=["uppercase", "prefixed", "too-short", "not-hex"],
    )
    def test_a_hash_that_could_never_match_is_refused_at_load(self, catalog_path, value):
        """cvmd compares against lowercase hex it computes itself.

        An entry written any other way would never match, and the launch would fail as "not
        approved" — which sends an operator looking at the wrong thing entirely.
        """
        with pytest.raises(CatalogError, match="64 lowercase hex"):
            load_catalog(write(catalog_path, entry(compose_hash=value)))

    def test_two_entries_with_the_same_triple_are_refused(self, catalog_path):
        """A request that could resolve to either entry has no single answer."""
        with pytest.raises(CatalogError, match="pin the same kind and triple"):
            load_catalog(write(catalog_path, entry(id="a"), entry(id="b")))


class TestResolving:
    @pytest.fixture
    def artifacts(self, catalog_path):
        return load_catalog(
            write(
                catalog_path,
                entry(),
                entry(id="renter-v1", kind="renter", compose_hash="c" * 64),
            )
        )

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
