"""DAH-2977 — a pre-pulled image the backend stops serving is removed after a grace period.

The clock starts at the first sweep that no longer lists the image and resets when it comes
back; only images the puller tracks are touched; mandatory (default) images never are; docker
refusing the removal (a container still uses it) keeps it tracked; 0 turns eviction off.
"""

import asyncio
import json
from unittest.mock import MagicMock

import docker
import pytest

docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

from services import pre_pull_service  # noqa: E402
from services.pre_pull_service import PrePuller  # noqa: E402

REPO = "daturaai/pytorch"
CU128_REF = f"{REPO}:2.11.0-py3.12-cuda12.8-devel-ubuntu24.04-dind-lium1"
OLD_REF = f"{REPO}:2.8.0-py3.11-cuda12.8-devel-ubuntu22.04"
CUDA_REF = "nvidia/cuda:13.0.3-devel-ubuntu22.04"
DIGEST_CU128 = "sha256:" + "a" * 64
DIGEST_OLD = "sha256:" + "d" * 64
DIGEST_CUDA = "sha256:" + "b" * 64
DAY = 24 * 3600
GIB = pre_pull_service.GIB


def _entry(ref: str, digest: str) -> dict:
    repo, _, tag = ref.rpartition(":")
    return {
        "docker_image": repo,
        "docker_image_tag": tag,
        "docker_image_size": 4_000_000_000,
        "docker_image_digest": digest,
        "pre_pull": True,
    }


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(pre_pull_service.time, "time", fake.time)
    return fake


@pytest.fixture(autouse=True)
def idle_node_with_room(monkeypatch):
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: [])
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=1000 * GIB))
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_START_JITTER_SECONDS", 0)
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_EVICT_UNLISTED_AFTER_SECONDS", DAY)
    monkeypatch.setattr(pre_pull_service, "_pull_pinned", lambda *a, **k: ("pull_ok", None))


def _client(refuse: set[str] = frozenset()) -> MagicMock:
    """Every tracked digest is present; ``images.remove`` records calls and refuses ``refuse``."""
    client = MagicMock()
    client.images.get.side_effect = lambda ref: MagicMock()
    client.containers.list.return_value = []
    client.removed = []

    def remove(ref):
        if ref in refuse:
            raise RuntimeError(f"conflict: unable to remove {ref} (must be forced) - image is being used")
        client.removed.append(ref)

    client.images.remove.side_effect = remove
    return client


def _puller(tmp_path, client, tracked: dict[str, str]) -> PrePuller:
    puller = PrePuller(client, state_path=str(tmp_path / "state.json"))
    puller._first_sweep = False
    for ref, digest in tracked.items():
        puller.state.record_present(ref, digest, 4_000_000_000)
    return puller


def _sweep(puller, entries, protected=frozenset()):
    asyncio.run(puller.sweep(entries, protected=protected))


def test_default_is_one_day():
    from core.config import Settings

    assert Settings.model_fields["PRE_PULL_EVICT_UNLISTED_AFTER_SECONDS"].default == DAY


def test_unlisted_image_is_removed_only_after_the_grace(tmp_path, clock):
    client = _client()
    puller = _puller(tmp_path, client, {CU128_REF: DIGEST_CU128, OLD_REF: DIGEST_OLD})
    served = [_entry(CU128_REF, DIGEST_CU128)]

    _sweep(puller, served)  # OLD_REF drops off the list here: the clock starts
    clock.now += DAY - 60
    _sweep(puller, served)
    assert client.removed == []
    assert OLD_REF in puller.state.images

    clock.now += 120
    _sweep(puller, served)
    # Both the tag and the pinned digest reference go, so the layers are reclaimed.
    assert client.removed == [OLD_REF, f"{REPO}@{DIGEST_OLD}"]
    assert OLD_REF not in puller.state.images
    assert CU128_REF in puller.state.images


def test_relisted_image_resets_the_clock(tmp_path, clock):
    client = _client()
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD})

    _sweep(puller, [])
    clock.now += DAY - 60
    _sweep(puller, [_entry(OLD_REF, DIGEST_OLD)])  # back in the top-N
    assert "unlisted_since" not in puller.state.images[OLD_REF]
    clock.now += DAY - 60
    _sweep(puller, [])  # unlisted again: a fresh clock
    clock.now += 120
    _sweep(puller, [])
    assert client.removed == []
    assert OLD_REF in puller.state.images


def test_empty_list_counts_as_unlisted(tmp_path, clock):
    """TOP_N=0 on the backend (the fleet kill switch) frees the disk a day later."""
    client = _client()
    puller = _puller(tmp_path, client, {CUDA_REF: DIGEST_CUDA})

    _sweep(puller, [])
    clock.now += DAY
    _sweep(puller, [])

    assert client.removed == [CUDA_REF, f"nvidia/cuda@{DIGEST_CUDA}"]
    assert puller.state.images == {}


def test_mandatory_image_is_never_evicted(tmp_path, clock):
    client = _client()
    puller = _puller(tmp_path, client, {CUDA_REF: DIGEST_CUDA})
    puller.protected = frozenset({CUDA_REF})

    _sweep(puller, [])
    clock.now += 2 * DAY
    _sweep(puller, [])

    assert client.removed == []


def test_image_in_use_stays_tracked(tmp_path, clock):
    """Docker untags an image in use but refuses its last reference, the pinned digest."""
    client = _client(refuse={f"{REPO}@{DIGEST_OLD}"})
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD})

    _sweep(puller, [])
    clock.now += DAY
    _sweep(puller, [])

    assert OLD_REF in puller.state.images


def test_busy_node_keeps_the_image_until_idle(tmp_path, clock):
    """A rental may be starting on the unlisted image: removal waits for an idle node."""
    client = _client()
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD})
    rental = MagicMock(status="running")
    rental.name = "pod_abc"

    _sweep(puller, [])
    clock.now += DAY
    client.containers.list.return_value = [rental]
    _sweep(puller, [])
    assert client.removed == []
    assert OLD_REF in puller.state.images

    client.containers.list.return_value = []
    _sweep(puller, [])
    assert client.removed == [OLD_REF, f"{REPO}@{DIGEST_OLD}"]


def test_rental_starting_mid_eviction_stops_it(tmp_path, clock):
    client = _client()
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD, CUDA_REF: DIGEST_CUDA})
    rental = MagicMock(status="created")
    rental.name = "pod_abc"
    _sweep(puller, [])
    clock.now += DAY

    def remove_then_rental_starts(ref):
        client.removed.append(ref)
        client.containers.list.return_value = [rental]

    client.images.remove.side_effect = remove_then_rental_starts
    _sweep(puller, [])

    assert len(puller.state.images) == 1


def test_image_made_mandatory_during_eviction_is_kept(tmp_path, clock, monkeypatch):
    """The loop may publish a new mandatory set while eviction waits on docker."""
    client = _client()
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD})
    _sweep(puller, [])
    clock.now += DAY

    def idle_but_refreshed(_client):
        puller.protected = frozenset({OLD_REF})
        return None

    monkeypatch.setattr(pre_pull_service, "rental_activity", idle_but_refreshed)
    _sweep(puller, [])

    assert client.removed == []


def test_zero_disables_eviction(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_EVICT_UNLISTED_AFTER_SECONDS", 0)
    client = _client()
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD})

    _sweep(puller, [])
    clock.now += 30 * DAY
    _sweep(puller, [])

    assert client.removed == []
    assert OLD_REF in puller.state.images


def test_untracked_images_are_never_touched(tmp_path, clock):
    """Only images this puller pulled are candidates: a renter's or the default image is not in its state."""
    client = _client()
    puller = _puller(tmp_path, client, {})

    _sweep(puller, [])
    clock.now += 2 * DAY
    _sweep(puller, [])

    assert client.removed == []


def test_unlisted_since_survives_a_restart(tmp_path, clock):
    client = _client()
    puller = _puller(tmp_path, client, {OLD_REF: DIGEST_OLD})
    _sweep(puller, [])
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["images"][OLD_REF]["unlisted_since"] == clock.now

    clock.now += DAY
    restarted = PrePuller(client, state_path=str(tmp_path / "state.json"))
    restarted._first_sweep = False
    _sweep(restarted, [])

    assert OLD_REF in client.removed
