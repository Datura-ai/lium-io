"""The prod liumd binary repair path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import redis.exceptions
from fakeredis.aioredis import FakeRedis
from services.liumd_binary_refresh import ensure_liumd_binary

from services import liumd_binary_refresh

MANIFEST = {
    "build_sha": "new-build",
    "release": "1.2.3",
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


def _ctx(runner, batch="batch-1", redis=None, executor="exec-1"):
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(job_batch_id=batch),
        miner_hotkey="miner-hk",
        executor=SimpleNamespace(uuid=executor),
        services=SimpleNamespace(redis=SimpleNamespace(redis=redis if redis is not None else FakeRedis())),
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


# The signed manifest: the real fetch path with a test release key pinned in place of the production one.
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat  # noqa: E402

RELEASE_KEY = Ed25519PrivateKey.generate()
OTHER_KEY = Ed25519PrivateKey.generate()
MANIFEST_BYTES = (json.dumps(MANIFEST, indent=2) + "\n").encode()


def _sig(key, data=MANIFEST_BYTES):
    return (key.sign(data).hex() + "\n").encode()


def _serve(monkeypatch, files):
    """files: URL suffix -> bytes, or None for a 404."""
    fetched = []

    async def fetch_bytes(url, timeout, limit):
        fetched.append(url)
        body = files.get(url.rsplit("/", 1)[-1])
        return (body, None) if body is not None else (None, "fetch status 404")

    monkeypatch.setattr(liumd_binary_refresh, "_fetch_bytes", fetch_bytes)
    pub = RELEASE_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    monkeypatch.setattr(liumd_binary_refresh, "MANIFEST_PUBLIC_KEY_HEX", pub)
    return fetched


def test_the_pinned_key_is_the_liumd_release_key():
    assert liumd_binary_refresh.MANIFEST_PUBLIC_KEY_HEX == (
        "7b187df472a586d61423cf24d72bbccdb98bfd671df5b4b6b9e713676990a41b"
    )


@pytest.mark.asyncio
async def test_a_manifest_signed_by_the_release_key_installs(monkeypatch):
    fetched = _serve(monkeypatch, {"manifest.json": MANIFEST_BYTES, "manifest.json.sig": _sig(RELEASE_KEY)})
    runner = Runner()

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert fetched == ["https://example.test/manifest.json", "https://example.test/manifest.json.sig"]
    assert (result.ok, result.changed, result.build_sha) == (True, True, NEW_BUILD)
    assert runner.installed


@pytest.mark.parametrize(
    ("case", "manifest", "signature", "error"),
    [
        ("missing", MANIFEST_BYTES, None, "manifest signature fetch status 404"),
        ("other key", MANIFEST_BYTES, _sig(OTHER_KEY), "does not verify against the pinned release key"),
        ("manifest changed after signing", MANIFEST_BYTES.replace(b"a" * 64, b"c" * 64), _sig(RELEASE_KEY),
         "does not verify against the pinned release key"),
        ("not hex", MANIFEST_BYTES, b"zz\n", "manifest signature is not hex"),
        ("short", MANIFEST_BYTES, b"00" * 32, "manifest signature is not 64 bytes"),
        ("empty", MANIFEST_BYTES, b"", "manifest signature is not 64 bytes"),
    ],
)
@pytest.mark.parametrize("local_present", [True, False])
@pytest.mark.asyncio
async def test_a_manifest_without_a_valid_signature_installs_nothing(
    monkeypatch, case, manifest, signature, error, local_present
):
    _serve(monkeypatch, {"manifest.json": manifest, "manifest.json.sig": signature})
    runner = Runner(local_present=local_present)

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert error in result.detail
    assert not runner.installed
    assert not any(c.startswith(("curl ", "install ")) for c in runner.calls)
    # fail closed and reported as a refusal, not as an unavailable manifest: the node keeps the binary it has
    assert (result.ok, result.refused, result.build_sha) == (False, True, "old-build" if local_present else None)
    assert "using local" not in result.detail


@pytest.mark.asyncio
async def test_a_signature_fetch_that_fails_on_the_network_is_not_a_refusal(monkeypatch):
    async def fetch_bytes(url, timeout, limit):
        return (MANIFEST_BYTES, None) if url.endswith(".json") else (None, "ClientError: reset")

    monkeypatch.setattr(liumd_binary_refresh, "_fetch_bytes", fetch_bytes)
    result = await ensure_liumd_binary(_ctx(Runner()), budget_seconds=60)

    assert (result.ok, result.refused, result.build_sha) == (True, False, "old-build")
    assert "manifest unavailable, using local liumd" in result.detail


def _signed(release):
    data = (json.dumps({**MANIFEST, "release": release}, indent=2) + "\n").encode()
    return {"manifest.json": data, "manifest.json.sig": _sig(RELEASE_KEY, data)}


@pytest.mark.asyncio
async def test_a_validly_signed_older_release_is_refused_after_a_newer_one_was_accepted(monkeypatch):
    """The replay: main gets an older signed manifest and its binaries back. The signature verifies, but the node
    accepted a higher release before, so nothing is installed and the refusal is reported as one."""
    redis = FakeRedis()
    _serve(monkeypatch, _signed("1.2.3"))
    assert (await ensure_liumd_binary(_ctx(Runner(), redis=redis), budget_seconds=60)).changed

    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, _signed("1.2.2"))
    runner = Runner(local=NEW_BUILD)
    result = await ensure_liumd_binary(_ctx(runner, redis=redis), budget_seconds=60)

    assert (result.ok, result.refused, result.build_sha) == (False, True, NEW_BUILD)
    assert "release 1.2.2 is below 1.2.3" in result.detail
    assert not any(c.startswith(("curl ", "install ")) for c in runner.calls)


@pytest.mark.asyncio
async def test_the_run_answer_refresh_refuses_an_older_release_and_a_bad_signature(monkeypatch):
    """The refresh after a run answers takes the same gate: no older signed release, no unsigned manifest."""
    redis = FakeRedis()
    _serve(monkeypatch, _signed("1.2.3"))
    assert (await ensure_liumd_binary(_ctx(Runner(), redis=redis), budget_seconds=60)).ok

    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, _signed("1.2.2"))
    runner = Runner()
    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner, redis=redis), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )
    assert (result.ok, result.refused, result.build_sha) == (False, True, "old-build")
    assert not any(c.startswith(("curl ", "install ")) for c in runner.calls)

    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, {**_signed("1.2.4"), "manifest.json.sig": b"00" * 64})
    result = await liumd_binary_refresh.refresh_liumd_from_answer(
        _ctx(runner, redis=redis), "old-build", "x86_64-unknown-linux-gnu", budget_seconds=60
    )
    assert (result.ok, result.refused) == (False, True)
    assert not any(c.startswith(("curl ", "install ")) for c in runner.calls)


@pytest.mark.asyncio
async def test_an_older_release_is_refused_on_a_node_that_never_saw_the_newer_one(monkeypatch):
    """The floor is the validator's too, not only the node's: a node new to it gets no older release either."""
    redis = FakeRedis()
    _serve(monkeypatch, _signed("2.0.0"))
    assert (await ensure_liumd_binary(_ctx(Runner(), redis=redis, executor="a"), budget_seconds=60)).ok

    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, _signed("1.9.9"))
    result = await ensure_liumd_binary(_ctx(Runner(), redis=redis, executor="b"), budget_seconds=60)

    assert (result.ok, result.refused) == (False, True)


@pytest.mark.parametrize(("first", "then", "accepted"), [("1.2.3", "1.2.3", True), ("1.9.0", "1.10.0", True),
                                                         ("1.10.0", "1.9.0", False)])
@pytest.mark.asyncio
async def test_the_release_compares_by_number_and_the_same_one_stays_accepted(monkeypatch, first, then, accepted):
    redis = FakeRedis()
    _serve(monkeypatch, _signed(first))
    assert (await ensure_liumd_binary(_ctx(Runner(), redis=redis), budget_seconds=60)).ok
    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, _signed(then))

    result = await ensure_liumd_binary(_ctx(Runner(), redis=redis), budget_seconds=60)

    assert (result.ok, result.refused) == (accepted, not accepted)


@pytest.mark.parametrize("release", ["0.9.9", None, "1.2", "v1.2.3", "01.2.3", 123])
@pytest.mark.asyncio
async def test_a_release_below_the_code_minimum_or_not_x_y_z_is_refused(monkeypatch, release):
    files = _signed(release)
    if release is None:
        data = (json.dumps({k: v for k, v in MANIFEST.items() if k != "release"}, indent=2) + "\n").encode()
        files = {"manifest.json": data, "manifest.json.sig": _sig(RELEASE_KEY, data)}
    _serve(monkeypatch, files)
    runner = Runner()

    result = await ensure_liumd_binary(_ctx(runner), budget_seconds=60)

    assert (result.ok, result.refused, runner.installed) == (False, True, False)


@pytest.mark.asyncio
async def test_an_unreadable_floor_installs_nothing_and_keeps_the_local_binary(monkeypatch):
    class Down:
        async def mget(self, keys):
            raise ConnectionError("redis down")

    _serve(monkeypatch, _signed("1.2.3"))
    runner = Runner()
    result = await ensure_liumd_binary(_ctx(runner, redis=Down()), budget_seconds=60)

    assert (result.ok, result.build_sha, runner.installed) == (True, "old-build", False)
    assert "release floor unavailable" in result.detail


@pytest.mark.asyncio
async def test_the_manifest_is_not_parsed_before_its_signature_verifies(monkeypatch):
    _serve(monkeypatch, {"manifest.json": b"not json", "manifest.json.sig": _sig(OTHER_KEY, b"not json")})

    doc, error = await liumd_binary_refresh._fetch_manifest("https://example.test/manifest.json", 5)

    assert doc is None
    assert error == "manifest signature does not verify against the pinned release key"


class _WritesRefused(FakeRedis):
    """Redis that serves reads but refuses writes (maxmemory + noeviction) until `refusing` is cleared."""

    refusing = True

    def pipeline(self, *args, **kwargs):
        if self.refusing:
            raise redis.exceptions.ResponseError("OOM command not allowed when used memory > 'maxmemory'.")
        return super().pipeline(*args, **kwargs)


@pytest.mark.asyncio
async def test_a_floor_that_cannot_be_recorded_installs_nothing(monkeypatch):
    """The floor is written before the install: a release the floor does not remember is never installed, so it
    cannot be rolled back by an older signed one afterwards."""
    store = _WritesRefused()
    for release in ("1.5.0", "1.2.0"):
        liumd_binary_refresh._MANIFEST_CACHE.clear()
        _serve(monkeypatch, _signed(release))
        runner = Runner()
        result = await ensure_liumd_binary(_ctx(runner, redis=store), budget_seconds=60)
        assert (result.ok, result.refused, result.build_sha, runner.installed) == (True, False, "old-build", False)
        assert "nothing installed: release floor not recorded: ResponseError: OOM" in result.detail
        assert not any(c.startswith(("curl ", "install ")) for c in runner.calls)

    store.refusing = False
    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, _signed("1.5.0"))
    assert (await ensure_liumd_binary(_ctx(Runner(), redis=store), budget_seconds=60)).changed
    liumd_binary_refresh._MANIFEST_CACHE.clear()
    _serve(monkeypatch, _signed("1.2.0"))
    runner = Runner(local=NEW_BUILD)
    result = await ensure_liumd_binary(_ctx(runner, redis=store), budget_seconds=60)
    assert (result.ok, result.refused, runner.installed) == (False, True, False)


@pytest.mark.asyncio
async def test_the_floor_is_stored_zero_padded_so_1_10_stays_above_1_9():
    store = FakeRedis()
    ctx = _ctx(Runner(), redis=store)
    for release in ((1, 9, 0), (1, 10, 0)):
        assert await liumd_binary_refresh._raise_floor(ctx, release) == ""

    assert await store.mget(liumd_binary_refresh._floor_keys(ctx)) == [b"000001.000010.000000"] * 2
    assert await liumd_binary_refresh._release_floor(ctx) == (1, 10, 0)


@pytest.mark.asyncio
async def test_concurrent_floor_raises_keep_the_highest_release():
    """The WATCH compare-and-set: whatever order the raises land in, a lower one never overwrites a higher one."""
    import asyncio

    store = FakeRedis()
    ctx = _ctx(Runner(), redis=store)
    releases = [(1, 10, 0), (1, 9, 0), (2, 0, 0), (1, 2, 3)] * 5

    assert set(await asyncio.gather(*(liumd_binary_refresh._raise_floor(ctx, r) for r in releases))) == {""}

    assert await store.mget(liumd_binary_refresh._floor_keys(ctx)) == [b"000002.000000.000000"] * 2


def _vendor(path, release, key=RELEASE_KEY, sign=True):
    data = (json.dumps({**MANIFEST, "release": release}, indent=2) + "\n").encode()
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_bytes(data)
    if sign:
        (path / "manifest.json.sig").write_bytes(_sig(key, data))
    return path


@pytest.mark.parametrize(
    ("vendored", "floor"),
    [
        ({"release": "1.4.2"}, (1, 4, 2)),
        ({"release": "1.4.2", "key": OTHER_KEY}, (1, 0, 0)),
        ({"release": "1.4.2", "sign": False}, (1, 0, 0)),
        ({"release": "1.4"}, (1, 0, 0)),
        (None, (1, 0, 0)),
    ],
)
def test_the_code_floor_is_the_release_of_the_signed_manifest_vendored_with_the_code(
    monkeypatch, tmp_path, vendored, floor
):
    _serve(monkeypatch, {})   # pins the test release key
    dirs = (tmp_path / "image", tmp_path / "repo")
    if vendored is not None:
        _vendor(dirs[1], **vendored)

    assert liumd_binary_refresh._vendored_release(dirs) == floor


def test_the_image_copy_wins_over_the_repo_copy(monkeypatch, tmp_path):
    _serve(monkeypatch, {})
    dirs = (_vendor(tmp_path / "image", "1.6.0"), _vendor(tmp_path / "repo", "1.4.2"))

    assert liumd_binary_refresh._vendored_release(dirs) == (1, 6, 0)


@pytest.mark.asyncio
async def test_after_a_redis_flush_the_vendored_release_still_refuses_an_older_one(monkeypatch, tmp_path):
    """A flush, an eviction or a new Redis drops the recorded floors; the release vendored with the code does not
    go, so a signed release older than it is still refused."""
    _serve(monkeypatch, {})
    monkeypatch.setattr(
        liumd_binary_refresh, "MIN_RELEASE", liumd_binary_refresh._vendored_release((_vendor(tmp_path, "1.5.0"),))
    )
    _serve(monkeypatch, _signed("1.2.0"))
    runner = Runner()

    result = await ensure_liumd_binary(_ctx(runner, redis=FakeRedis()), budget_seconds=60)

    assert (result.ok, result.refused, runner.installed) == (False, True, False)
    assert "release 1.2.0 is below 1.5.0" in result.detail


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
