"""The liumd refresh manifest must describe the binaries vendored beside it."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from core.config import Settings

ROOT = Path(__file__).resolve().parents[3]
LIUMD = ROOT / "neurons/executor/liumd"


def test_manifest_hashes_match_the_vendored_binaries():
    manifest = json.loads((LIUMD / "manifest.json").read_text())
    for key, binary in (("x86_64", "liumd-amd64"), ("aarch64", "liumd-arm64")):
        entry = manifest["binaries"][key]
        assert entry["file"] == binary
        assert entry["sha256"] == hashlib.sha256((LIUMD / binary).read_bytes()).hexdigest()


def test_manifest_build_sha_matches_the_x86_binary():
    manifest = json.loads((LIUMD / "manifest.json").read_text())
    version = json.loads(subprocess.check_output([str(LIUMD / "liumd-amd64"), "version"], text=True))
    assert manifest["build_sha"] == version["build_sha"]


def test_the_default_manifest_url_is_main():
    url = Settings.model_fields["LIUMD_BINARY_MANIFEST_URL"].default
    assert url == "https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/liumd/manifest.json"


def test_the_vendored_manifest_signature_verifies_against_the_pinned_key():
    import pytest
    from services.liumd_binary_refresh import _signature_error

    sig = LIUMD / "manifest.json.sig"
    if not sig.exists():
        pytest.skip("manifest.json.sig is vendored from the first liumd-v* release run")
    assert _signature_error((LIUMD / "manifest.json").read_bytes(), sig.read_bytes()) is None


def test_the_code_floor_is_the_vendored_release():
    """MIN_RELEASE is read from the vendored signed manifest: vendoring a release raises the floor with no other edit."""
    from services import liumd_binary_refresh

    release = json.loads((LIUMD / "manifest.json").read_text()).get("release")
    signed = (LIUMD / "manifest.json.sig").exists()
    expected = liumd_binary_refresh._release(release) if signed else (1, 0, 0)
    assert liumd_binary_refresh.MIN_RELEASE == expected
