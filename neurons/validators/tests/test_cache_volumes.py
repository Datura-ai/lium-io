import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from payload_models.payloads import CacheVolume, ContainerCreateRequest, WorkloadKind
from services.docker_service import DockerService, _filler_run_volume_gb, _wipe_volume_cmd


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _make_payload(
    *,
    workload_kind: WorkloadKind,
    cache_volumes: list[CacheVolume] | None,
    active_container_names: list[str] | None = None,
) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        workload_kind=workload_kind,
        cache_volumes=cache_volumes,
        active_container_names=active_container_names or [],
    )


class FakeSshClient:
    """Records commands and replays a canned `docker volume ls` listing."""

    def __init__(self, listed_volumes: list[str]):
        self._listing = "\n".join(listed_volumes)
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        stdout = self._listing if "volume ls" in command else ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")


class HostSshClient(FakeSshClient):
    """Answers the three questions the affordability decision asks: which cache volumes exist, where
    the docker root is, and how much space is free on it."""

    def __init__(self, listed_volumes: list[str], free_gb: float):
        super().__init__(listed_volumes)
        self._free_bytes = int(free_gb * 1024**3)

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        if "docker info" in command:
            return SimpleNamespace(exit_status=0, stdout="/var/lib/docker\n", stderr="")
        if "df -P -B1" in command:
            body = f"Filesystem 1B-blocks Used Available Capacity\n/dev/sda1 1 1 {self._free_bytes} 1% /hostfs"
            return SimpleNamespace(exit_status=0, stdout=body, stderr="")
        stdout = self._listing if "volume ls" in command else ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")


def _removed_volumes(ssh_client: FakeSshClient) -> list[str]:
    return sorted(
        name
        for command in ssh_client.commands
        if "volume rm" in command
        for name in command.split("volume rm ", 1)[1].split(" 2>/dev/null")[0].split()
    )


# ---------------------------------------------------------------- reclaim (levels 2 and 3)


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "granted"), [(5000, False), (5300, True)])
async def test_an_engy_checkpoint_is_cached_only_when_it_is_at_most_30pct_of_free_disk(
    docker_service, free_gb, granted
):
    # DAH-5014: the size comes from the name, not DPHN's 40 GB. 1561 GB is 31% of 5000 GB, 29% of 5300 GB.
    # A 1500 GB run volume could not hold it, so the share rule alone decides.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_1561gb_kimi-k3-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )
    payload.volume_limit_gb = 1500

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert bool(affordable) is granted


@pytest.mark.asyncio
@pytest.mark.parametrize(("volume_limit_gb", "granted"), [(1500, False), (1900, True)])
async def test_an_engy_checkpoint_over_30pct_is_cached_when_the_runs_own_volume_could_hold_it(
    docker_service, volume_limit_gb, granted
):
    # 1561 GB is 41% of 3800 GB free. Refused, it would download into a 1900 GB run volume anyway, on every
    # launch; 3800 - 1561 - 1900 = 339 still clears the floor. A 1500 GB volume could not hold it: refused.
    ssh_client = HostSshClient(["volume_pod"], free_gb=3800)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="engy_cache_hf_1561gb_kimi-k3-1a2b3c4d", target="/root/models")],
    )
    payload.volume_limit_gb = volume_limit_gb

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert bool(affordable) is granted


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "granted"), [(220, False), (230, True)])
async def test_an_engy_checkpoint_and_the_pod_volume_must_leave_the_listing_floor_and_margin(
    docker_service, free_gb, granted
):
    # DAH-5014: real free disk minus the 31 GB checkpoint and the 40 GB pod volume created after this
    # check must stay over the 100 GB floor + 50 GB margin: 220 leaves 149, 230 leaves 159.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_31gb_qwen3-8-27b-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )
    payload.volume_limit_gb = 40

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert bool(affordable) is granted


@pytest.mark.parametrize(
    ("volume_limit_gb", "cache_names", "expected_gb"),
    [
        (1900, ["engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"], 100),  # checkpoint in the cache: no second copy
        (1900, [], 1900),  # cache refused: the run volume still has to hold the checkpoint
        (40, ["engy_cache_hf_31gb_qwen3-8-27b-1a2b3c4d"], 40),  # never grows a smaller request
        (80, ["dphn_cache_hf_x"], 80),  # Dolphin keeps its size
        (None, ["engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"], None),  # no storage limit: nothing to size
    ],
)
def test_the_filler_run_volume_drops_the_checkpoint_once_the_engy_cache_holds_it(
    volume_limit_gb, cache_names, expected_gb
):
    caches = [CacheVolume(name=name, target="/root/models") for name in cache_names]

    assert _filler_run_volume_gb(volume_limit_gb, caches) == expected_gb


@pytest.mark.asyncio
async def test_an_engy_checkpoint_of_unknown_size_is_never_downloaded_into_a_cache(docker_service):
    ssh_client = HostSshClient(["volume_pod"], free_gb=100_000)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_kimi-k3-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )

    assert await docker_service.select_affordable_cache_volumes(ssh_client, payload, {}) == []


@pytest.mark.asyncio
async def test_an_engy_create_sweeps_the_old_dolphin_cache_but_a_dolphin_create_keeps_the_engy_one(docker_service):
    # DAH-5192: a node moved to Engy no longer runs Dolphin; the reverse would drop a TB checkpoint
    engy = [CacheVolume(name="engy_cache_hf_new", target="/c")]
    ssh_client = FakeSshClient(["engy_cache_hf_new", "dphn_cache_hf_old", "volume_pod"])
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=engy)

    assert await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {}) == ["dphn_cache_hf_old"]

    dolphin = [CacheVolume(name="dphn_cache_hf_new", target="/c")]
    ssh_client = FakeSshClient(["dphn_cache_hf_new", "engy_cache_hf_old"])
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=dolphin)

    assert await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {}) == []


# ---------------------------------------------------------------- DAH-5192: partial caches, thresholds, rentals

_KIMI = "engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"


class DuHostSshClient(HostSshClient):
    """A host whose existing cache volume holds `held_kb` (the helper's `du -sk` answer)."""

    def __init__(self, listed_volumes: list[str], free_gb: float, du_stdout: str):
        super().__init__(listed_volumes, free_gb)
        self._du_stdout = du_stdout

    async def run(self, command: str, *args, **kwargs):
        if "du -sk" in command:
            self.commands.append(command)
            return SimpleNamespace(exit_status=0, stdout=self._du_stdout, stderr="")
        return await super().run(command, *args, **kwargs)


def _kimi_filler(volume_limit_gb: int | None = 1900) -> ContainerCreateRequest:
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER, cache_volumes=[CacheVolume(name=_KIMI, target="/root/models")]
    )
    payload.volume_limit_gb = volume_limit_gb
    return payload


def _kb(gb: float) -> str:
    return f"{int(gb * 10**9 / 1024)}\t/cache\n"


@pytest.mark.asyncio
async def test_a_complete_cache_is_mounted_without_measuring_free_disk(docker_service):
    ssh_client = DuHostSshClient([_KIMI], free_gb=10, du_stdout=_kb(1560))

    assert await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {}) == _kimi_filler().cache_volumes
    assert not any("df -P" in command for command in ssh_client.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "granted"), [(3800, True), (1200, False)])
async def test_an_interrupted_download_is_resumed_when_the_node_can_afford_what_it_still_lacks(
    docker_service, free_gb, granted
):
    # A rental removed the filler at 400 of 1561 GB. 3800 free: the 1161 GB left fits, the volume is remounted and the
    # next run resumes it. 1200 free: 1200 - 1161 - 100 is under the floor, so the partial is not mounted.
    ssh_client = DuHostSshClient([_KIMI], free_gb=free_gb, du_stdout=_kb(400))

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {})

    assert bool(affordable) is granted
    assert any("df -P" in command for command in ssh_client.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("du_stdout", ["", "du: /cache: I/O error\n", "garbage"])
async def test_an_unmeasurable_cache_is_mounted_as_before(docker_service, du_stdout):
    ssh_client = DuHostSshClient([_KIMI], free_gb=10, du_stdout=du_stdout)

    assert await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {}) != []


@pytest.mark.asyncio
@pytest.mark.parametrize("volume_limit_gb", [1900, None])
@pytest.mark.parametrize(("free_gb", "granted"), [(2600, False), (3500, False), (3600, True), (5100, True)])
async def test_a_big_checkpoint_is_cached_only_when_the_node_still_fits_kimi_k3_beside_it(
    docker_service, free_gb, granted, volume_limit_gb
):
    # The cache stays used, and the backend gives kimi-k3 only to a node with 2000 GB free. 3500 - 1561 = 1939 GB:
    # cached, the node would never fit kimi-k3 again, so the run downloads into its own volume. 3600 leaves 2039 GB.
    # 3700-5100 GB B300s were refused by the 30% share in prod (26 Oct) and re-downloaded on every run.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(volume_limit_gb), {})

    assert bool(affordable) is granted


def _rental(volume_limit_gb: int | None, storage_limit_gb: int | None = None) -> ContainerCreateRequest:
    payload = _make_payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, cache_volumes=None)
    payload.volume_limit_gb = volume_limit_gb
    payload.storage_limit_gb = storage_limit_gb
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize(("volume_limit_gb", "storage_limit_gb"), [(2000, None), (1300, 650)])
async def test_a_rental_that_fits_beside_the_cache_keeps_it(docker_service, volume_limit_gb, storage_limit_gb):
    # 2200 GB really free covers the volume and the container storage: the cache stays
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=2200)

    rental = _rental(volume_limit_gb, storage_limit_gb)
    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}) is False
    assert _removed_volumes(ssh_client) == []


@pytest.mark.asyncio
async def test_a_whole_node_rental_sized_on_the_listed_cache_evicts_it(docker_service):
    # DAH-5260: the 8x B300 lists 3950 GB free, the 1561 GB kimi-k3 cache included. The backend sizes a whole-node
    # rental on that (2620 GB volume + 1310 GB storage); 2389 GB really free cannot hold it, so the cache goes.
    ssh_client = HostSshClient([_KIMI, "dphn_cache_hf_x", "volume_pod"], free_gb=2389)

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(2620, 1310), {}) is True
    assert _removed_volumes(ssh_client) == ["dphn_cache_hf_x", _KIMI]


@pytest.mark.asyncio
async def test_the_container_storage_counts_toward_what_the_rental_needs(docker_service):
    # 2300 GB volume alone fits 2389 GB; with its 1150 GB storage the renter could write past the disk
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=2389)

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(2300, 1150), {}) is True
    assert _removed_volumes(ssh_client) == [_KIMI]


@pytest.mark.asyncio
@pytest.mark.parametrize("free_gb", [100_000, 100, 10])
@pytest.mark.parametrize("volumes", [[_KIMI, "dphn_cache_hf_x"], [_KIMI], ["dphn_cache_hf_x"]])
async def test_an_unsized_rental_keeps_every_cache_however_short_the_disk(docker_service, volumes, free_gb):
    # Rustam 2 Nov: no --storage-opt means the renter did not ask for the disk, so the cache survives the rental.
    # Evicting up front would re-download up to 1.5 TB after every such rental (the churn DAH-5014 stopped).
    ssh_client = DuHostSshClient([*volumes, "volume_pod"], free_gb=free_gb, du_stdout=_kb(1560))

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(None), {}) is False
    assert _removed_volumes(ssh_client) == []
    # decided on the request alone: no df or du read on the create path
    assert not any("df -P" in command or "du -sk" in command for command in ssh_client.commands)


@pytest.mark.asyncio
async def test_a_cache_removal_that_never_answers_is_given_up_and_the_rent_goes_on(docker_service, monkeypatch):
    # a wedged dockerd must not hold the create (and the node's create lock) forever
    class HangingRemoval(HostSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "volume rm" in command:
                await asyncio.sleep(3600)
            return await super().run(command, *args, **kwargs)

    monkeypatch.setattr("services.docker_service.CACHE_VOLUME_REMOVE_TIMEOUT_SECONDS", 0.01)
    ssh_client = HangingRemoval([_KIMI, "volume_pod"], free_gb=10)

    assert await asyncio.wait_for(docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(500), {}), 5) is True


@pytest.mark.asyncio
async def test_rental_evicts_the_oldest_cache_first_and_keeps_the_rest_once_it_fits(docker_service, monkeypatch):
    # A stale listing (the download kept writing until the filler was removed) must not fail a paid create: the
    # older cache goes first and the free-disk re-check stops the eviction there.
    class InspectingSshClient(FakeSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "volume inspect" in command:
                self.commands.append(command)
                body = "engy_cache_hf_new 2026-10-22T10:00:00Z\nengy_cache_hf_old 2026-10-01T10:00:00Z"
                return SimpleNamespace(exit_status=0, stdout=body, stderr="")
            return await super().run(command, *args, **kwargs)

    ssh_client = InspectingSshClient(["engy_cache_hf_new", "engy_cache_hf_old", "volume_pod"])
    monkeypatch.setattr(docker_service, "get_docker_root_dir", AsyncMock(return_value="/var/lib/docker"))
    monkeypatch.setattr(
        docker_service, "_get_fs_available_bytes", AsyncMock(side_effect=[50 * 1024**3, 60 * 1024**3, 900 * 1024**3])
    )  # before, after the idle filler images went, after the older cache went

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(500), {}) is True
    assert _removed_volumes(ssh_client) == ["engy_cache_hf_old"]


@pytest.mark.asyncio
async def test_a_rental_that_still_does_not_fit_after_the_eviction_is_clamped_to_the_freed_disk(docker_service):
    # a `volume rm` that failed (it hides errors behind `|| true`) frees nothing: 2389 GB stays free, so the legacy
    # 2620 GB volume would be preallocated on disk that was never freed and the create would fail
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=2389)
    rental = _rental(2620, 1310)

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}) is True
    assert rental.volume_limit_gb == 2389 - 1310


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("removal_works", "listing_fails", "volume_gb"),
    [(True, False, 2620), (False, False, 2389 - 1310), (True, True, 2389 - 1310)],
)
async def test_a_rental_still_short_is_clamped_only_when_a_removal_failed(
    docker_service, removal_works, listing_fails, volume_gb
):
    # DAH-5260 review: every removal worked and the disk is still short, so the cache was not what was missing and the
    # size stays as ordered; a removal that left its volume behind clamps, and so does a re-listing that failed
    class RemovingSshClient(HostSshClient):
        removed = False

        async def run(self, command: str, *args, **kwargs):
            if "volume rm" in command and removal_works:
                self._listing = "volume_pod"
                self.removed = True
            if "volume ls" in command and listing_fails and self.removed:
                return SimpleNamespace(exit_status=1, stdout="", stderr="")
            return await super().run(command, *args, **kwargs)

    rental = _rental(2620, 1310)

    assert await docker_service.reclaim_dphn_cache_for_rental(
        RemovingSshClient([_KIMI, "volume_pod"], free_gb=2389), rental, {}
    ) is True
    assert rental.volume_limit_gb == volume_gb


class KilledFillerHostSshClient(HostSshClient):
    """prod 4 Nov, 8x B300: the filler the create killed is still exiting and mounts kimi-k3, so `volume rm` fails.
    The wipe unlinks the files (disk freed) only when every user is in its allowed list; the volume stays listed."""

    def __init__(self, users: list[str], **kwargs):
        super().__init__([_KIMI, "volume_pod"], **kwargs)
        self.users = users

    async def run(self, command: str, *args, **kwargs):
        if "volume rm" in command and "/proc/1/root" in command:
            allowed = command.split("ok=' ", 1)[1].split(" '", 1)[0].split()
            if all(user in allowed for user in self.users):
                self._free_bytes += 1561 * 1024**3
        return await super().run(command, *args, **kwargs)


@pytest.mark.asyncio
async def test_a_cache_the_killed_filler_still_mounts_is_wiped_and_the_order_keeps_its_size(docker_service):
    ssh_client = KilledFillerHostSshClient(["filler_79ba7fa2"], free_gb=2389)
    rental = _rental(2620, 1310)

    assert await docker_service.reclaim_dphn_cache_for_rental(
        ssh_client, rental, {}, killed_fillers=["filler_79ba7fa2"]
    ) is True
    assert rental.volume_limit_gb == 2620
    assert _removed_volumes(ssh_client) == [_KIMI]


@pytest.mark.asyncio
@pytest.mark.parametrize("killed_fillers", [[], ["filler_79ba7fa2"]])
async def test_a_cache_a_live_filler_mounts_is_not_wiped_and_the_order_clamps(docker_service, killed_fillers):
    ssh_client = KilledFillerHostSshClient(["filler_79ba7fa2", "filler_live"], free_gb=2389)
    rental = _rental(2620, 1310)

    assert await docker_service.reclaim_dphn_cache_for_rental(
        ssh_client, rental, {}, killed_fillers=killed_fillers
    ) is True
    assert rental.volume_limit_gb == 2389 - 1310


def test_the_wipe_refuses_a_volume_another_container_mounts():
    command = _wipe_volume_cmd(_KIMI, "filler_a", "filler_b")
    assert "ok=' filler_a filler_b '" in command and "exit 6" in command
    assert 'find "/proc/1/root$p" -mindepth 1 -maxdepth 1 -exec rm -rf {} +' in command


@pytest.mark.asyncio
async def test_an_emptied_cache_left_behind_is_swept_and_never_mounted_as_ready(docker_service):
    # the wipe leaves an empty kimi-k3 volume: another model's filler sweeps it, the same model's is charged it all
    other = _make_payload(
        workload_kind=WorkloadKind.FILLER, cache_volumes=[CacheVolume(name="engy_cache_hf_80gb_other", target="/c")]
    )
    assert await docker_service.sweep_stale_cache_volumes(FakeSshClient([_KIMI, "volume_pod"]), other, {}) == [_KIMI]

    ssh_client = DuHostSshClient([_KIMI], free_gb=1200, du_stdout=_kb(0))
    assert await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {}) == []


_DPHN = ["dphn_cache_dp_runtime-1a2b3c4d", "dphn_cache_hf_auto-5e6f7a8b"]


class KilledFillerDolphinHost(HostSshClient):
    """prod 4 Nov, 8x B300: the killed filler is still exiting (rm timed out, D state) and mounts both Dolphin
    volumes, so `volume rm` fails; each wipe frees `wiped_gb` when every user is in its allowed list."""

    def __init__(self, users: list[str], wiped_gb: float, **kwargs):
        super().__init__([*_DPHN, "volume_pod"], **kwargs)
        self.users = users
        self.wiped_gb = wiped_gb

    async def run(self, command: str, *args, **kwargs):
        if "volume rm" in command and "/proc/1/root" in command:
            allowed = command.split("ok=' ", 1)[1].split(" '", 1)[0].split()
            if all(user in allowed for user in self.users):
                self._free_bytes += int(self.wiped_gb * 1024**3)
        return await super().run(command, *args, **kwargs)


def _one_eighth_rental(monkeypatch, docker_service, volume_gb: int, storage_gb: int, existing_bytes: int):
    monkeypatch.setattr(docker_service, "_get_existing_vloopback_bytes", AsyncMock(return_value=existing_bytes))
    rental = _rental(volume_gb, storage_gb)
    rental.disk_share = 0.125
    return rental


# the two 4 Nov creates: free and needed GiB from the logs; the Dolphin set's on-disk size there was not
# logged, so each wipe frees half the gap plus a little (illustrative)
_DOLPHIN_CASES = [
    pytest.param("filler_killed_a", 271, 135, 587 * 10**9, 2466.1, id="14:51Z"),
    pytest.param("filler_killed_b", 358, 179, int(1596.3 * 1024**3), 2324.0, id="18:37Z"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("filler", "volume_gb", "storage_gb", "existing_bytes", "free_gb"), _DOLPHIN_CASES)
async def test_dolphin_caches_the_killed_filler_still_mounts_are_wiped_and_the_order_keeps_its_size(
    docker_service, monkeypatch, filler, volume_gb, storage_gb, existing_bytes, free_gb
):
    rental = _one_eighth_rental(monkeypatch, docker_service, volume_gb, storage_gb, existing_bytes)
    gap_gb = (await docker_service._rental_disk_fit(None, rental)).needs_free_gb - free_gb
    ssh_client = KilledFillerDolphinHost([filler], wiped_gb=gap_gb / 2 + 1, free_gb=free_gb)

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}, killed_fillers=[filler]) is True
    assert _removed_volumes(ssh_client) == _DPHN
    assert any("/proc/1/root" in command and _DPHN[0] in command for command in ssh_client.commands)
    assert rental.volume_limit_gb == volume_gb


@pytest.mark.asyncio
@pytest.mark.parametrize(("filler", "volume_gb", "storage_gb", "existing_bytes", "free_gb"), _DOLPHIN_CASES)
async def test_dolphin_caches_a_live_filler_mounts_are_not_wiped_and_the_order_clamps(
    docker_service, monkeypatch, filler, volume_gb, storage_gb, existing_bytes, free_gb
):
    rental = _one_eighth_rental(monkeypatch, docker_service, volume_gb, storage_gb, existing_bytes)
    ssh_client = KilledFillerDolphinHost([filler, "filler_live"], wiped_gb=500, free_gb=free_gb)

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}, killed_fillers=[filler]) is True
    assert rental.volume_limit_gb < volume_gb


@pytest.mark.asyncio
@pytest.mark.parametrize(("held_gb", "free_gb", "granted"), [(0, 100, False), (0, 400, True), (12, 100, True)])
async def test_an_emptied_dolphin_cache_is_charged_its_download_and_a_filled_one_mounts_as_before(
    docker_service, held_gb, free_gb, granted
):
    # the wipe leaves the volume empty: mounting it is a 40 GB download again, refused under the floor + margin
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=[CacheVolume(name=_DPHN[1], target="/c")])
    payload.volume_limit_gb = 40
    ssh_client = DuHostSshClient([_DPHN[1]], free_gb=free_gb, du_stdout=_kb(held_gb))

    assert bool(await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})) is granted


@pytest.mark.asyncio
async def test_a_timed_out_removal_still_clamps_the_rental_to_the_free_disk(docker_service, monkeypatch):
    class HangingRemoval(HostSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "volume rm" in command:
                await asyncio.sleep(3600)
            return await super().run(command, *args, **kwargs)

    monkeypatch.setattr("services.docker_service.CACHE_VOLUME_REMOVE_TIMEOUT_SECONDS", 0.01)
    rental = _rental(500)

    await asyncio.wait_for(
        docker_service.reclaim_dphn_cache_for_rental(HangingRemoval([_KIMI, "volume_pod"], free_gb=300), rental, {}), 5
    )
    assert rental.volume_limit_gb == 300


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "removed"), [(2389, []), (1000, [_KIMI])])
async def test_a_recreate_on_its_existing_volume_needs_only_its_storage_and_is_never_clamped(
    docker_service, free_gb, removed
):
    # the pod's 2620 GB volume is already preallocated, outside df free: only the 1310 GB storage is new
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=free_gb)
    rental = _rental(2620, 1310)
    rental.local_volume = "volume_pod"

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}) is bool(removed)
    assert _removed_volumes(ssh_client) == removed
    assert rental.volume_limit_gb == 2620


@pytest.mark.asyncio
async def test_a_rental_takes_the_engy_cache_before_an_older_dolphin_cache(docker_service, monkeypatch):
    # the listing offered the Engy cache as free disk, never the Dolphin one, so the Engy family goes first
    class InspectingSshClient(FakeSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "volume inspect" in command:
                self.commands.append(command)
                body = f"{_KIMI} 2026-10-22T10:00:00Z\ndphn_cache_hf_x 2026-10-01T10:00:00Z"
                return SimpleNamespace(exit_status=0, stdout=body, stderr="")
            return await super().run(command, *args, **kwargs)

    ssh_client = InspectingSshClient([_KIMI, "dphn_cache_hf_x", "volume_pod"])
    monkeypatch.setattr(docker_service, "get_docker_root_dir", AsyncMock(return_value="/var/lib/docker"))
    monkeypatch.setattr(
        docker_service,
        "_get_fs_available_bytes",
        AsyncMock(side_effect=[50 * 1024**3, 60 * 1024**3] + [900 * 1024**3] * 3),
    )

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(500), {}) is True
    assert _removed_volumes(ssh_client) == [_KIMI]


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_after_gb", "volume_gb"), [(3950, 1310), (2389, 789)])
async def test_a_half_node_rental_evicts_the_cache_its_share_was_sized_on(
    docker_service, monkeypatch, free_after_gb, volume_gb
):
    # 1310 GB volume + 655 GB storage at half the node needs 3930 GB free plus the sizing's 20 GB overhead; 2389 GB
    # beside the cache is not enough, so the cache goes. A removal that freed nothing clamps the volume to what the
    # sizing will give.
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=2389)
    gb = 1024**3
    monkeypatch.setattr(docker_service, "_get_existing_vloopback_bytes", AsyncMock(return_value=0))
    monkeypatch.setattr(
        docker_service, "_get_fs_available_bytes", AsyncMock(side_effect=[2389 * gb] * 2 + [free_after_gb * gb] * 3)
    )
    rental = _rental(1310, 655)
    rental.disk_share = 0.5

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}) is True
    assert _removed_volumes(ssh_client) == [_KIMI]
    assert rental.volume_limit_gb == volume_gb


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "removed"), [(130, [_KIMI]), (150, [])])
async def test_a_fresh_sized_rental_counts_the_sizing_overhead_in_its_fit(docker_service, monkeypatch, free_gb, removed):
    # DAH-5260 live test: 82 + 41 GB fit 130 GB free, but resolve_volume_sizing keeps 20 GB back, so the pod got
    # 116 GB of 123 and the cache stayed. The fit check leaves out the same 20 GB.
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=free_gb)
    monkeypatch.setattr(docker_service, "_get_existing_vloopback_bytes", AsyncMock(return_value=0))
    rental = _rental(82, 41)
    rental.disk_share = 1.0

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}) is bool(removed)
    assert _removed_volumes(ssh_client) == removed


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "removed"), [(105, [_KIMI]), (115, [])])
async def test_a_fresh_sized_rental_fits_only_under_the_df_guard_too(docker_service, monkeypatch, free_gb, removed):
    # DAH-5260 review: 500 GB of pod volumes already on the node make the pool fit, but resolve_volume_sizing's
    # df_guard caps the volume at df free less 10 GB, so 100 GB needs 110 GB free or the cache goes
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=free_gb)
    monkeypatch.setattr(docker_service, "_get_existing_vloopback_bytes", AsyncMock(return_value=500 * 1024**3))
    rental = _rental(100, 50)
    rental.disk_share = 1.0

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, rental, {}) is bool(removed)
    assert _removed_volumes(ssh_client) == removed
