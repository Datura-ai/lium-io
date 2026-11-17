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
