"""The prod liumd binary repair path."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from services.liumd_binary_refresh import ensure_liumd_binary

from services import liumd_binary_refresh

PINNED = liumd_binary_refresh.PINNED_MANIFEST
NEW_BUILD = PINNED["build_sha"]
X86_SHA = PINNED["binaries"]["x86_64"]["sha256"]


class Runner:
    def __init__(
        self,
        *,
        arch="x86_64",
        local="old-build",
        target=None,
        local_present=True,
        outcome="ok",
        version_delay=0.0,
        version_error=False,
    ):
        self.arch = arch
        self.local = local
        self.target = target if target is not None else f"{arch}-unknown-linux-gnu"
        self.local_present = local_present
        self.outcome = outcome
        self.version_delay = version_delay
        self.version_error = version_error
        self.installed = False
        self.calls: list[str] = []

    async def run(self, command, timeout, retryable):
        self.calls.append(command)
        if command.endswith(" version"):
            if self.version_delay:
                await asyncio.sleep(self.version_delay)
            if self.version_error:
                return SimpleNamespace(
                    success=False,
                    stdout="",
                    stderr="",
                    exit_status=None,
                    error_type="timeout",
                )
            build = NEW_BUILD if self.installed else (self.local if self.local_present else None)
            if build is None:
                return SimpleNamespace(
                    success=False, stdout="", stderr="No such file", exit_status=127
                )
            return SimpleNamespace(
                success=True,
                stdout=json.dumps({"build_sha": build, "target": self.target}),
                stderr="",
                exit_status=0,
            )
        if command == "uname -m":
            return SimpleNamespace(
                success=True, stdout=self.arch + "\n", stderr="", exit_status=0
            )
        if "mktemp /tmp/liumd." in command:
            if self.outcome == "download_failed":
                return SimpleNamespace(
                    success=False,
                    stdout="CURL_RC:7\n",
                    stderr="curl: (7) Failed to connect",
                    exit_status=10,
                )
            if self.outcome == "wrong_hash":
                return SimpleNamespace(
                    success=False,
                    stdout=f"{X86_SHA}  /tmp/liumd.abc: FAILED\nSHA_RC:1\n",
                    stderr="",
                    exit_status=11,
                )
            if self.outcome == "install_failed":
                return SimpleNamespace(
                    success=False,
                    stdout="CURL_RC:0\nSHA_RC:0\nINSTALL_RC:1\n",
                    stderr="install: cannot create",
                    exit_status=1,
                )
            self.installed = True
            return SimpleNamespace(
                success=True,
                stdout="CURL_RC:0\nSHA_RC:0\nINSTALL_RC:0\n",
                stderr="",
                exit_status=0,
            )
        raise AssertionError(command)


def _ctx(runner):
    return SimpleNamespace(
        runner=runner,
        miner_hotkey="5Miner",
        executor=SimpleNamespace(uuid="exec-1", address="10.0.0.5", ssh_port=2200),
    )


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    liumd_binary_refresh._VERIFIED_BUILDS.clear()
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_REFRESH_ENABLED", True)
    monkeypatch.setattr(
        liumd_binary_refresh.settings, "LIUMD_BINARY_FETCH_BASE_URL", "https://example.test/liumd/"
    )
    monkeypatch.setattr(
        liumd_binary_refresh.settings, "LIUMD_BINARY_PATH", "/usr/local/bin/liumd"
    )
    yield
    liumd_binary_refresh._VERIFIED_BUILDS.clear()


@pytest.mark.asyncio
async def test_an_up_to_date_binary_is_not_downloaded():
    runner = Runner(local=NEW_BUILD)

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is True
    assert result.changed is False
    assert len(runner.calls) == 1
    assert runner.calls[0].endswith(" version")
    assert not any("mktemp" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_a_stale_binary_is_downloaded_and_installed_in_one_command():
    runner = Runner()

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is True
    assert result.changed is True
    assert result.build_sha == NEW_BUILD
    assert runner.installed is True
    [install] = [c for c in runner.calls if "mktemp" in c]
    assert "sha256sum -c -" in install
    assert "install -m 0755" in install
    assert "trap 'rm -f \"$tmp\"' EXIT" in install
    assert "/tmp/liumd.binary.fetch" not in install
    assert "liumd-amd64" in install


@pytest.mark.asyncio
async def test_a_wrong_download_hash_does_not_install():
    runner = Runner(outcome="wrong_hash")

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is False
    assert "does not match the pinned build" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_a_failed_download_does_not_install():
    runner = Runner(outcome="download_failed")

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is False
    assert "download failed (curl exit 7)" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_a_failed_install_does_not_report_success():
    runner = Runner(outcome="install_failed")

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is False
    assert "install failed (exit 1)" in result.detail


@pytest.mark.asyncio
async def test_a_missing_binary_uses_the_host_arch():
    runner = Runner(local_present=False)

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is True
    assert result.changed is True
    assert "uname -m" in runner.calls


@pytest.mark.asyncio
async def test_the_binary_target_arch_avoids_uname():
    runner = Runner()

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is True
    assert "uname -m" not in runner.calls


@pytest.mark.asyncio
async def test_an_unsupported_arch_does_not_install():
    runner = Runner(local_present=False, arch="riscv64")

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.ok is False
    assert "architecture" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_a_verified_build_is_cached_per_executor():
    runner = Runner(local=NEW_BUILD)
    ctx = _ctx(runner)

    assert (await ensure_liumd_binary(ctx)).ok is True
    calls = len(runner.calls)
    second = await ensure_liumd_binary(ctx)

    assert second.ok is True and second.changed is False
    assert len(runner.calls) == calls


@pytest.mark.asyncio
async def test_a_cached_build_expires(monkeypatch):
    runner = Runner(local=NEW_BUILD)
    ctx = _ctx(runner)
    assert (await ensure_liumd_binary(ctx)).ok is True
    calls = len(runner.calls)

    monkeypatch.setattr(liumd_binary_refresh.time, "monotonic", lambda: 10**9)

    assert (await ensure_liumd_binary(ctx)).ok is True
    assert len(runner.calls) == calls + 1


@pytest.mark.asyncio
async def test_a_version_read_timeout_skips_without_fetching():
    runner = Runner(version_error=True)

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.skipped is True
    assert "could not be read" in result.detail
    assert not any("mktemp" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_a_low_task_budget_skips_without_ssh():
    runner = Runner()
    budget_left = (
        liumd_binary_refresh.REFRESH_TASK_RESERVE_SECONDS
        + liumd_binary_refresh.MIN_REFRESH_BUDGET_SECONDS
        - 1
    )

    result = await ensure_liumd_binary(_ctx(runner), seconds_left=budget_left)

    assert result.skipped is True
    assert runner.calls == []


@pytest.mark.asyncio
async def test_a_slow_refresh_is_bounded(monkeypatch):
    monkeypatch.setattr(liumd_binary_refresh, "REFRESH_BUDGET_SECONDS", 0.2)
    monkeypatch.setattr(liumd_binary_refresh, "MIN_REFRESH_BUDGET_SECONDS", 0.1)
    runner = Runner(version_delay=1.0)

    result = await ensure_liumd_binary(_ctx(runner))

    assert result.skipped is True
    assert "budget" in result.detail
    assert not any("mktemp" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_a_failed_refresh_is_not_cached():
    runner = Runner(outcome="wrong_hash")
    ctx = _ctx(runner)

    assert (await ensure_liumd_binary(ctx)).ok is False
    runner.outcome = "ok"

    assert (await ensure_liumd_binary(ctx)).ok is True


@pytest.mark.asyncio
async def test_the_feature_is_off_by_default(monkeypatch):
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_REFRESH_ENABLED", False)
    runner = Runner()

    assert await ensure_liumd_binary(_ctx(runner)) is None
    assert runner.calls == []
