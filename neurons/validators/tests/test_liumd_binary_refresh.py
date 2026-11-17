"""The prod liumd binary repair path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from services import liumd_binary_refresh
from services.liumd_binary_refresh import ensure_liumd_binary

MANIFEST = {
    "build_sha": "new-build",
    "binaries": {
        "x86_64": {"file": "liumd-amd64", "sha256": "a" * 64},
        "aarch64": {"file": "liumd-arm64", "sha256": "b" * 64},
    },
}
VERSION = '{"build_sha": "old-build"}'


class Runner:
    def __init__(self, *, arch="x86_64", local="old-build", manifest=None, download_sha="a" * 64):
        self.arch = arch
        self.local = local
        self.manifest = manifest if manifest is not None else MANIFEST
        self.download_sha = download_sha
        self.calls: list[str] = []
        self.installed = False

    async def run(self, command, timeout, retryable):
        self.calls.append(command)
        if command == "uname -m":
            return SimpleNamespace(success=True, stdout=self.arch + "\n", stderr="", exit_status=0)
        if command.endswith(" version"):
            build = "new-build" if self.installed else self.local
            return SimpleNamespace(
                success=build is not None,
                stdout=f'{{"build_sha": "{build}"}}' if build is not None else "",
                stderr="",
                exit_status=0 if build is not None else 127,
            )
        if "manifest.json" in command:
            return SimpleNamespace(success=True, stdout=__import__("json").dumps(self.manifest), stderr="", exit_status=0)
        if "liumd-amd64" in command or "liumd-arm64" in command:
            return SimpleNamespace(success=True, stdout=f"RC:0\n{self.download_sha}  /tmp/liumd.binary.fetch\n", stderr="", exit_status=0)
        if command.startswith("install -m 0755"):
            self.installed = True
            return SimpleNamespace(success=True, stdout="", stderr="", exit_status=0)
        raise AssertionError(command)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_REFRESH_ENABLED", True)
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_MANIFEST_URL", "https://example.test/manifest.json")
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_PATH", "/usr/local/bin/liumd")


@pytest.mark.asyncio
async def test_an_up_to_date_binary_is_not_downloaded():
    runner = Runner(local="new-build")

    result = await ensure_liumd_binary(SimpleNamespace(runner=runner))

    assert result.ok is True
    assert result.changed is False
    assert not any("liumd-amd64" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_a_stale_binary_is_downloaded_only_after_the_hash_matches():
    runner = Runner()

    result = await ensure_liumd_binary(SimpleNamespace(runner=runner))

    assert result.ok is True
    assert result.changed is True
    assert result.build_sha == "new-build"
    assert runner.installed is True
    assert any(c.startswith("install -m 0755") for c in runner.calls)


@pytest.mark.asyncio
async def test_a_wrong_download_hash_does_not_install():
    runner = Runner(download_sha="c" * 64)

    result = await ensure_liumd_binary(SimpleNamespace(runner=runner))

    assert result.ok is False
    assert "does not match" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_an_unsupported_arch_does_not_fetch():
    runner = Runner(arch="riscv64")

    result = await ensure_liumd_binary(SimpleNamespace(runner=runner))

    assert result.ok is False
    assert "architecture" in result.detail
    assert not any("manifest.json" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_the_feature_is_off_by_default(monkeypatch):
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_REFRESH_ENABLED", False)
    runner = Runner()

    assert await ensure_liumd_binary(SimpleNamespace(runner=runner)) is None
    assert runner.calls == []
