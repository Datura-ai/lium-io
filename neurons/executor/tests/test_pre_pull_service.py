"""DAH-2977 — idle-time pre-pull of the top-N official templates.

Off by default (today's behaviour byte-for-byte); on, at most one digest-pinned pull per
sweep, never while a rental exists or is starting, never below the disk floor (LRU
pre-pulled images are evicted first), never past the per-image timeout.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock

import docker
import pytest

# conftest replaces the docker module with a MagicMock; except-clauses in the
# services need a real exception class to catch, and the pull helper unpacks
# docker.auth's registry resolution.
docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
docker.auth.resolve_repository_name = lambda repo: ("docker.io", repo)
docker.auth.get_config_header = lambda api, registry: None

from services import cache_template_service, pre_pull_service  # noqa: E402
from services.pre_pull_service import PrePuller, PrePullState, _pull_pinned, rental_activity  # noqa: E402

REPO = "daturaai/pytorch"
DEFAULT_TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"
CU128_TAG = "2.12.0-py3.12-cuda12.8-devel-ubuntu24.04-dind"
CU128_REF = f"{REPO}:{CU128_TAG}"
CLUSTER_REF = "daturaai/lium-cluster:0.0.7"
DIGEST_DEFAULT = "sha256:70bd5fa697877594b753a146e207ca4de66d9b875d606ae09e6ee7bac8f4f423"
DIGEST_CU128 = "sha256:2d19c94ce8a37c6fa364f8a6211d8b6dc1a44ece574c4c22ab5579925ce7a4c8"
DIGEST_CLUSTER = "sha256:9f1f8b0c5e5d9c3a7b6e4d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1"
GIB = pre_pull_service.GIB


def _entry(repo: str, tag: str, digest: str | None, size: int = 4_300_000_000, pre_pull: bool = True) -> dict:
    data = {"docker_image": repo, "docker_image_tag": tag, "docker_image_size": size, "pre_pull": pre_pull}
    if digest:
        data["docker_image_digest"] = digest
    return data


def _client(present_digests: set[str] = frozenset(), containers: list = ()) -> MagicMock:
    client = MagicMock()

    def images_get(ref: str):
        if any(ref.endswith(f"@{digest}") for digest in present_digests):
            return MagicMock()
        raise docker.errors.ImageNotFound(ref)

    client.images.get.side_effect = images_get
    client.containers.list.return_value = list(containers)
    return client


def _container(name: str, image: str = CU128_REF, status: str = "running") -> MagicMock:
    container = MagicMock()
    container.name = name
    container.status = status
    container.attrs = {"Config": {"Image": image}}
    return container


def _proc(cmdline: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.info = {"cmdline": cmdline}
    return proc


@pytest.fixture(autouse=True)
def quiet_node(monkeypatch):
    """Idle host with plenty of disk, no start jitter, and an instant fake pull."""
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: [])
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=1000 * GIB))
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_START_JITTER_SECONDS", 0)
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_MIN_FREE_GB", 200)
    pulls: list[tuple] = []

    def fake_pull(client, repo, tag, digest, timeout_seconds):
        pulls.append((repo, tag, digest, timeout_seconds))
        return "pull_ok", None

    monkeypatch.setattr(pre_pull_service, "_pull_pinned", fake_pull)
    return pulls


def _sweep(puller: PrePuller, entries: list[dict]) -> None:
    asyncio.run(puller.sweep(entries))


# --- flag off = today's behaviour -------------------------------------------------------


def _one_loop_iteration(monkeypatch, templates: list[dict]) -> dict:
    """Run run_cache_template_prefetch through exactly one sweep and record what it did."""
    seen: dict = {"params": None, "ensured": [], "swept": [], "pullers": 0}

    async def fetch(session, url, params):
        seen["params"] = dict(params)
        return templates, 200, None

    async def ensure(client, data, state, keep_tags=frozenset()):
        seen["ensured"].append((data["docker_image_tag"], set(keep_tags)))

    class FakePuller:
        def __init__(self, client, state_path=None):
            seen["pullers"] += 1

        async def sweep(self, entries):
            seen["swept"].append([e["docker_image_tag"] for e in entries])

    async def stop(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(cache_template_service, "_fetch_templates", fetch)
    monkeypatch.setattr(cache_template_service, "_ensure_template", ensure)
    monkeypatch.setattr(cache_template_service, "PrePuller", FakePuller)
    monkeypatch.setattr(cache_template_service, "_get_gpu_info", lambda: ("NVIDIA H100 80GB HBM3", "580.65.06", None))
    monkeypatch.setattr(cache_template_service.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cache_template_service.run_cache_template_prefetch(state_path=None))
    return seen


def test_flag_off_does_not_ask_for_or_touch_pre_pull_entries(monkeypatch):
    monkeypatch.setattr(cache_template_service.settings, "PRE_PULL_TEMPLATES_ENABLED", False)
    default = _entry(REPO, DEFAULT_TAG, DIGEST_DEFAULT, pre_pull=False)

    seen = _one_loop_iteration(monkeypatch, [default])

    assert "include_pre_pull" not in seen["params"]
    assert seen["ensured"] == [(DEFAULT_TAG, set())]
    assert seen["pullers"] == 0 and seen["swept"] == []


def test_flag_on_asks_backend_and_routes_pre_pull_entries_to_the_puller(monkeypatch):
    monkeypatch.setattr(cache_template_service.settings, "PRE_PULL_TEMPLATES_ENABLED", True)
    default = _entry(REPO, DEFAULT_TAG, DIGEST_DEFAULT, pre_pull=False)
    extra = _entry(REPO, CU128_TAG, DIGEST_CU128)

    seen = _one_loop_iteration(monkeypatch, [default, extra])

    assert seen["params"]["include_pre_pull"] == "true"
    # the mandatory path still handles only the default image, now shielding the extra's tag
    assert seen["ensured"] == [(DEFAULT_TAG, {CU128_TAG})]
    assert seen["swept"] == [[CU128_TAG]]


def test_cleanup_old_tags_keeps_pre_pull_tags_of_the_same_repository():
    client = MagicMock()
    image = MagicMock()
    image.tags = [f"{REPO}:{DEFAULT_TAG}", CU128_REF, f"{REPO}:2.4.0-old"]
    client.images.list.return_value = [image]

    asyncio.run(cache_template_service._cleanup_old_tags(client, REPO, DEFAULT_TAG, frozenset({CU128_TAG})))

    client.images.remove.assert_called_once_with(f"{REPO}:2.4.0-old")


# --- idle guard -------------------------------------------------------------------------


def test_sweep_pulls_one_missing_image_per_sweep_when_idle(quiet_node, caplog):
    client = _client(present_digests={DIGEST_DEFAULT})
    puller = PrePuller(client, state_path=None)

    with caplog.at_level(logging.INFO):
        _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128), _entry(*CLUSTER_REF.split(":"), DIGEST_CLUSTER)])

    assert quiet_node == [(REPO, CU128_TAG, DIGEST_CU128, pre_pull_service.settings.PRE_PULL_TIMEOUT_SECONDS)]
    assert puller.state.images[CU128_REF]["digest"] == DIGEST_CU128
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("pre_pull image="))
    assert f"image={CU128_REF}" in line and "outcome=pull_ok" in line and "seconds=" in line


def test_sweep_pulls_nothing_when_everything_is_present(quiet_node):
    client = _client(present_digests={DIGEST_CU128})

    _sweep(PrePuller(client, state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []


def test_sweep_ignores_entries_without_a_digest(quiet_node):
    _sweep(PrePuller(_client(), state_path=None), [_entry(REPO, CU128_TAG, digest=None)])

    assert quiet_node == []


def test_no_pull_while_a_rental_container_exists(quiet_node, caplog):
    client = _client(containers=[_container("pod_1f2e3d", image=f"{REPO}:{DEFAULT_TAG}")])

    with caplog.at_level(logging.INFO):
        _sweep(PrePuller(client, state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []
    assert any("node busy (rental container pod_1f2e3d)" in r.getMessage() for r in caplog.records)


def test_no_pull_while_the_validator_drives_docker_over_ssh(quiet_node, monkeypatch):
    # a rental's own `docker pull` / `docker run` arrive through this process on the host
    procs = [_proc(["sshd: root"]), _proc(["docker", "system", "dial-stdio"])]
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: procs)

    _sweep(PrePuller(_client(), state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []


def test_a_rental_container_still_being_created_counts_as_starting():
    assert rental_activity(_client(containers=[_container("pod_new", status="created")])) == "rental container pod_new"


def test_filler_infra_and_exited_rental_containers_do_not_count():
    client = _client(
        containers=[
            _container("filler_abc"),
            _container("executor-1"),
            _container("autoheal"),
            _container("pod_old", status="exited"),
        ]
    )

    assert rental_activity(client) is None


def test_rental_container_image_counts_as_used_for_lru(quiet_node):
    client = _client(present_digests={DIGEST_CU128}, containers=[_container("pod_1", image=CU128_REF)])
    puller = PrePuller(client, state_path=None)
    puller.state.record_present(CU128_REF, DIGEST_CU128, 1)
    puller.state.images[CU128_REF]["pulled_at"] = 1.0

    _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert puller.state.images[CU128_REF]["last_used_at"] > 1.0


# --- the pull itself: timeout, preemption, retag ----------------------------------------


def _pull_client(events: list[dict]) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    response = MagicMock()
    client.api._post.return_value = response
    client.api._stream_helper.return_value = iter(events)
    return client, response


def test_pull_ok_pulls_by_digest_and_retags(monkeypatch):
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: None)
    client, response = _pull_client([{"status": "Downloading"}, {"status": "Pull complete"}])

    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=60)

    assert (outcome, detail) == ("pull_ok", None)
    assert client.api._post.call_args.kwargs["params"] == {"fromImage": REPO, "tag": DIGEST_CU128}
    assert client.api._post.call_args.kwargs["timeout"] == 60
    client.images.get.assert_called_once_with(f"{REPO}@{DIGEST_CU128}")
    client.images.get.return_value.tag.assert_called_once_with(REPO, CU128_TAG)
    response.close.assert_called_once()


def test_pull_is_cancelled_when_a_rental_starts_mid_pull(monkeypatch):
    activity = iter([None, "rental container pod_9"])
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: next(activity))
    monkeypatch.setattr(pre_pull_service, "ACTIVITY_CHECK_SECONDS", 0)
    client, response = _pull_client([{"status": "Downloading"}] * 5)

    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=60)

    assert (outcome, detail) == ("preempted", "rental container pod_9")
    response.close.assert_called_once()  # closing the stream is what aborts the daemon-side pull
    client.images.get.assert_not_called()


def test_pull_stops_at_the_per_image_timeout(monkeypatch):
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: None)
    clock = iter([0.0, 0.0, 10.0, 100.0])  # deadline, next_check, then each event's check
    monkeypatch.setattr(pre_pull_service.time, "monotonic", lambda: next(clock))
    client, response = _pull_client([{"status": "Downloading"}] * 3)

    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=30)

    assert (outcome, detail) == ("timeout", "exceeded 30s")
    response.close.assert_called_once()
    client.images.get.assert_not_called()


def test_registry_error_event_is_a_failed_pull(monkeypatch):
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: None)
    client, _ = _pull_client([{"error": "toomanyrequests: rate limit"}])

    assert _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, 60) == ("pull_failed", "toomanyrequests: rate limit")


def test_pull_exception_is_logged_as_one_failed_line_and_not_recorded(quiet_node, monkeypatch, caplog):
    def boom(*_):
        raise RuntimeError("daemon gone")

    monkeypatch.setattr(pre_pull_service, "_pull_pinned", boom)
    puller = PrePuller(_client(), state_path=None)

    with caplog.at_level(logging.INFO):
        _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("pre_pull image=")]
    assert len(lines) == 1 and "outcome=pull_failed" in lines[0] and "daemon gone" in lines[0]
    assert CU128_REF not in puller.state.images


# --- disk guard -------------------------------------------------------------------------


def test_disk_guard_evicts_least_recently_used_pre_pulled_image_first(quiet_node, monkeypatch, caplog):
    free = iter([205 * GIB, 215 * GIB])  # need 3 × 4.3 GB ≈ 12 GiB on top of the 200 GiB floor
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=next(free)))
    client = _client()
    puller = PrePuller(client, state_path=None)
    puller.state.images = {
        "daturaai/a:1": {"digest": "sha256:aaa", "pulled_at": 1.0, "last_used_at": 100.0},  # used recently
        "daturaai/b:1": {"digest": "sha256:bbb", "pulled_at": 50.0},  # never used since its pull → LRU
    }

    with caplog.at_level(logging.INFO):
        _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    removed = [call.args[0] for call in client.images.remove.call_args_list]
    assert removed == ["daturaai/b:1", "daturaai/b@sha256:bbb"]
    assert set(puller.state.images) == {"daturaai/a:1", CU128_REF}
    assert len(quiet_node) == 1
    assert any("evicted daturaai/b:1" in r.getMessage() for r in caplog.records)


def test_disk_guard_skips_the_pull_when_nothing_can_be_evicted(quiet_node, monkeypatch, caplog):
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=150 * GIB))
    client = _client()

    with caplog.at_level(logging.INFO):
        _sweep(PrePuller(client, state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []
    client.images.remove.assert_not_called()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("pre_pull image="))
    assert "outcome=insufficient_disk" in line and "floor 200 GiB" in line


def test_disk_guard_never_evicts_what_docker_refuses_to_remove(quiet_node, monkeypatch):
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=150 * GIB))
    client = _client()
    client.images.remove.side_effect = RuntimeError("conflict: image is being used by running container")
    puller = PrePuller(client, state_path=None)
    puller.state.images = {"daturaai/b:1": {"digest": "sha256:bbb", "pulled_at": 50.0}}

    _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []
    assert "daturaai/b:1" in puller.state.images  # still there, still tracked


# --- rate limiting and state ------------------------------------------------------------


def test_first_sweep_waits_a_random_jitter_then_never_again(quiet_node, monkeypatch):
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_START_JITTER_SECONDS", 900)
    monkeypatch.setattr(pre_pull_service.random, "uniform", lambda lo, hi: 123.0)
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(pre_pull_service.asyncio, "sleep", fake_sleep)
    puller = PrePuller(_client(), state_path=None)

    _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128)])
    _sweep(puller, [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert slept == [123.0]


def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "pre_pull_state.json"
    state = PrePullState(str(path))
    state.record_present(CU128_REF, DIGEST_CU128, 4_300_000_000)
    state.flush()

    reloaded = PrePullState(str(path))

    assert reloaded.images[CU128_REF]["digest"] == DIGEST_CU128
    assert json.loads(path.read_text())["schema"] == 1


def test_corrupt_state_file_starts_empty(tmp_path):
    path = tmp_path / "pre_pull_state.json"
    path.write_text("{not json")

    assert PrePullState(str(path)).images == {}
