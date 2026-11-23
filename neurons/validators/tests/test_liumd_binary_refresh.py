"""The prod liumd binary repair path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from services.liumd_binary_refresh import ensure_liumd_binary

from services import liumd_binary_refresh

MANIFEST = {
    "build_sha": "new-build",
    "binaries": {
        "x86_64": {"file": "liumd-amd64", "sha256": "a" * 64},
        "aarch64": {"file": "liumd-arm64", "sha256": "b" * 64},
    },
}
NEW_BUILD = MANIFEST["build_sha"]


class Runner:
    def __init__(
        self,
        *,
        arch="x86_64",
        local="old-build",
        local_present=True,
        download_sha="a" * 64,
        download_rc=0,
        install_ok=True,
        install_sha=None,
        version_error=False,
    ):
        self.arch = arch
        self.local = local
        self.local_present = local_present
        self.download_sha = download_sha
        self.download_rc = download_rc
        self.install_ok = install_ok
        self.install_sha = install_sha
        self.version_error = version_error
        self.calls: list[str] = []
        self.installed = False

    async def run(self, command, timeout, retryable):
        self.calls.append(command)
        if command.endswith(" version"):
            if self.version_error:
                return SimpleNamespace(
                    success=False, stdout="", stderr="", exit_status=None, error_type="timeout"
                )
            build = NEW_BUILD if self.installed else (self.local if self.local_present else None)
            return SimpleNamespace(
                success=build is not None,
                stdout=(
                    json.dumps(
                        {"build_sha": build, "target": f"{self.arch}-unknown-linux-gnu"}
                    )
                    if build is not None
                    else ""
                ),
                stderr="",
                exit_status=0 if build is not None else 127,
            )
        if command == "uname -m":
            return SimpleNamespace(
                success=True, stdout=self.arch + "\n", stderr="", exit_status=0
            )
        if command.startswith("curl "):
            if self.download_rc:
                return SimpleNamespace(
                    success=True,
                    stdout=f"RC:{self.download_rc}\n",
                    stderr="curl: (22) 404",
                    exit_status=0,
                )
            return SimpleNamespace(
                success=True,
                stdout=f"RC:0\n{self.download_sha}  /tmp/liumd.binary.fetch\n",
                stderr="",
                exit_status=0,
            )
        if command.startswith("install -m 0755"):
            if not self.install_ok:
                return SimpleNamespace(
                    success=False,
                    stdout="",
                    stderr="install: permission denied",
                    exit_status=1,
                )
            self.installed = True
            return SimpleNamespace(
                success=True,
                stdout=f"{self.install_sha or self.download_sha}  /usr/local/bin/liumd\n",
                stderr="",
                exit_status=0,
            )
        raise AssertionError(command)


def _ctx(runner, batch="batch-1"):
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(job_batch_id=batch),
    )


def _manifest_fetcher(manifest=MANIFEST, error=None):
    calls = []

    async def fetch(url, timeout):
        calls.append((url, timeout))
        return (manifest, None) if error is None else (None, error)

    return fetch, calls


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    liumd_binary_refresh._MANIFEST_CACHE.clear()
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_REFRESH_ENABLED", True)
    monkeypatch.setattr(
        liumd_binary_refresh.settings,
        "LIUMD_BINARY_MANIFEST_URL",
        "https://example.test/manifest.json",
    )
    monkeypatch.setattr(
        liumd_binary_refresh.settings, "LIUMD_BINARY_PATH", "/usr/local/bin/liumd"
    )
    yield
    liumd_binary_refresh._MANIFEST_CACHE.clear()


@pytest.mark.asyncio
async def test_an_up_to_date_binary_is_not_downloaded(monkeypatch):
    runner = Runner(local=NEW_BUILD)
    fetch, fetches = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is True
    assert result.changed is False
    assert len(fetches) == 1
    assert fetches[0][1] <= liumd_binary_refresh.MANIFEST_TIMEOUT_SECONDS
    assert len(runner.calls) == 1
    assert runner.calls[0].endswith(" version")
    assert not any("curl " in c for c in runner.calls)
    assert "uname -m" not in runner.calls


@pytest.mark.asyncio
async def test_a_stale_binary_is_downloaded_only_after_the_hash_matches(monkeypatch):
    runner = Runner()
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is True
    assert result.changed is True
    assert result.build_sha == NEW_BUILD
    assert runner.installed is True
    [download] = [c for c in runner.calls if c.startswith("curl ")]
    max_time = int(download.split("--max-time ")[1].split()[0])
    assert 0 < max_time <= 60
    assert any(c.startswith("install -m 0755") for c in runner.calls)


@pytest.mark.asyncio
async def test_a_wrong_download_hash_does_not_install(monkeypatch):
    runner = Runner(download_sha="c" * 64)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is False
    assert "does not match" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_a_manifest_fetch_failure_keeps_a_usable_local_binary(monkeypatch):
    runner = Runner()
    fetch, _ = _manifest_fetcher(error="manifest timeout")
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is True
    assert result.changed is False
    assert "manifest timeout" in result.detail
    assert runner.calls[0].endswith(" version")
    assert not any(c.startswith("curl ") or c.startswith("install ") for c in runner.calls)


@pytest.mark.asyncio
async def test_a_manifest_fetch_failure_without_a_local_binary_fails(monkeypatch):
    runner = Runner(local_present=False)
    fetch, _ = _manifest_fetcher(error="manifest timeout")
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is False
    assert result.detail == "manifest timeout"
    assert runner.calls[0].endswith(" version")


@pytest.mark.asyncio
async def test_no_task_budget_skips_before_any_ssh(monkeypatch):
    runner = Runner()
    fetch, fetches = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=0)

    assert result.ok is False
    assert result.skipped is True
    assert "budget" in result.detail
    assert runner.calls == []
    assert fetches == []


@pytest.mark.asyncio
async def test_no_room_for_the_download_does_not_start_it(monkeypatch):
    runner = Runner(local="old-build")
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(
        _ctx(runner), budget_seconds=liumd_binary_refresh.MIN_DOWNLOAD_BUDGET_SECONDS
    )

    assert result.ok is False
    assert result.skipped is True
    assert "no task budget" in result.detail
    assert not any(c.startswith("curl ") for c in runner.calls)


@pytest.mark.asyncio
async def test_the_manifest_is_fetched_once_per_cycle(monkeypatch):
    runner = Runner(local=NEW_BUILD)
    fetch, fetches = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    assert (await ensure_liumd_binary(_ctx(runner), budget_seconds=60)).ok is True
    assert (await ensure_liumd_binary(_ctx(runner), budget_seconds=60)).ok is True
    assert len(fetches) == 1

    assert (await ensure_liumd_binary(_ctx(runner, "batch-2"), budget_seconds=60)).ok is True
    assert len(fetches) == 2


@pytest.mark.asyncio
async def test_a_local_version_that_cannot_be_read_does_not_download(monkeypatch):
    runner = Runner(version_error=True)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is False
    assert "could not be read" in result.detail
    assert not any(c.startswith("curl ") for c in runner.calls)


@pytest.mark.asyncio
async def test_an_unsupported_arch_does_not_download(monkeypatch):
    runner = Runner(local_present=False, arch="riscv64")
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is False
    assert "architecture" in result.detail
    assert not any(c.startswith("curl ") for c in runner.calls)


@pytest.mark.asyncio
async def test_a_manifest_file_outside_the_two_names_is_rejected(monkeypatch):
    runner = Runner()
    bad = {
        **MANIFEST,
        "binaries": {"x86_64": {"file": "../evil", "sha256": "a" * 64}},
    }
    fetch, _ = _manifest_fetcher(manifest=bad)
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert result.ok is False
    assert "no binary for this arch" in result.detail
    assert not any(c.startswith("curl ") for c in runner.calls)


@pytest.mark.asyncio
async def test_the_feature_is_off_by_default(monkeypatch):
    monkeypatch.setattr(liumd_binary_refresh.settings, "LIUMD_BINARY_REFRESH_ENABLED", False)
    runner = Runner()

    assert await ensure_liumd_binary(_ctx(runner)) is None
    assert runner.calls == []


@pytest.mark.asyncio
async def test_the_answer_build_refreshes_without_a_version_command(monkeypatch):
    runner = Runner()
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )

    assert result.ok is True
    assert result.changed is True
    assert not any(c.endswith(" version") for c in runner.calls)
    install = [c for c in runner.calls if c.startswith("install -m 0755")]
    assert len(install) == 1
    assert "sha256sum /usr/local/bin/liumd" in install[0]


@pytest.mark.asyncio
async def test_an_answer_on_the_expected_build_is_not_downloaded(monkeypatch):
    runner = Runner()
    fetch, fetches = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), NEW_BUILD, "x86_64-unknown-linux-gnu", budget_seconds=60
    )

    assert result.ok is True
    assert result.changed is False
    assert len(fetches) == 1
    assert runner.calls == []


@pytest.mark.asyncio
async def test_a_failed_answer_download_keeps_the_usable_answer(monkeypatch):
    runner = Runner(download_rc=22)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )

    assert result.ok is False
    assert result.skipped is True
    assert result.build_sha == "old-build"
    assert "404" in result.detail
    assert not any(c.startswith("install -m 0755") for c in runner.calls)


@pytest.mark.asyncio
async def test_a_wrong_answer_download_hash_keeps_the_usable_answer(monkeypatch):
    runner = Runner(download_sha="c" * 64)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )

    assert result.ok is False
    assert result.skipped is True
    assert "does not match" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_a_failed_answer_install_keeps_the_usable_answer(monkeypatch):
    runner = Runner(install_ok=False)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )

    assert result.ok is False
    assert result.skipped is True
    assert "permission denied" in result.detail
    assert runner.installed is False


@pytest.mark.asyncio
async def test_a_wrong_installed_answer_hash_keeps_the_usable_answer(monkeypatch):
    runner = Runner(install_sha="d" * 64)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )

    assert result.ok is False
    assert result.skipped is True
    assert result.build_sha == "old-build"
    assert "installed binary sha256 does not match" in result.detail
    assert runner.installed is True


@pytest.mark.asyncio
async def test_a_failed_answer_repair_without_a_build_is_not_skipped(monkeypatch):
    runner = Runner(local_present=False, download_rc=22)
    fetch, _ = _manifest_fetcher()
    monkeypatch.setattr(liumd_binary_refresh, "_fetch_manifest", fetch)

    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner), "", None, budget_seconds=60
    )

    assert result.ok is False
    assert result.skipped is False
    assert not any(c.endswith(" version") for c in runner.calls)
