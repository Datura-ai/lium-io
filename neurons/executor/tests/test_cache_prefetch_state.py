"""DAH-2470 — the cache-prefetch loop's structured state file.

The validator reads this document when it zeroes a node for a bad image digest, and it
is the only thing a reader will have: the loop's log lines never leave the provider's
machine. So every exit path must be named, every error text must survive, and neither
the document nor a broken write may ever disturb the loop.
"""

import asyncio
import json
from unittest.mock import MagicMock

import docker
import pytest

# conftest replaces the docker module with a MagicMock; except-clauses in the
# service need a real exception class to catch.
docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

from services.cache_prefetch_state import (  # noqa: E402
    MAX_ERROR_CHARS,
    MAX_PAYLOAD_BYTES,
    SCHEMA_VERSION,
    CachePrefetchState,
    Outcome,
)

from services import cache_template_service  # noqa: E402

REPO = "daturaai/pytorch"
TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"
IMAGE_REF = f"{REPO}:{TAG}"
FRESH_DIGEST = "sha256:70bd5fa697877594b753a146e207ca4de66d9b875d606ae09e6ee7bac8f4f423"
STALE_DIGEST = "sha256:2d19c94ce8a37c6fa364f8a6211d8b6dc1a44ece574c4c22ab5579925ce7a4c8"


def _make_client(local_digests: list[str] | None = None, image_absent: bool = False) -> MagicMock:
    client = MagicMock()
    if image_absent:
        client.images.get.side_effect = docker.errors.ImageNotFound("absent")
    else:
        image = MagicMock()
        image.attrs = {"RepoDigests": [f"{REPO}@{digest}" for digest in (local_digests or [])]}
        client.images.get.return_value = image
    client.images.list.return_value = []
    return client


def _template(digest: str | None = None, size: int = 0) -> dict:
    data = {"docker_image": REPO, "docker_image_tag": TAG, "docker_image_size": size}
    if digest is not None:
        data["docker_image_digest"] = digest
    return data


def _run(client, template, state):
    asyncio.run(cache_template_service._ensure_template(client, template, state))


def _image(state: CachePrefetchState) -> dict:
    return state.as_dict()["images"][IMAGE_REF]


# --- image-level outcomes ----------------------------------------------------------------


def test_up_to_date_at_backend_digest():
    state = CachePrefetchState(path=None)

    _run(_make_client(local_digests=[FRESH_DIGEST]), _template(FRESH_DIGEST), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.UP_TO_DATE
    assert record["expected_digest"] == FRESH_DIGEST
    assert record["local_digests"] == [f"{REPO}@{FRESH_DIGEST}"]


def test_up_to_date_at_registry_digest():
    # Legacy path: no backend digest, so the daemon's remote digest is the comparison.
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.return_value = MagicMock(id=STALE_DIGEST)

    _run(client, _template(), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.UP_TO_DATE
    assert record["remote_digest"] == STALE_DIGEST
    assert record["last_remote_read_ok_at"] is not None


def test_remote_digest_unreadable_keeps_the_registry_error():
    # DAH-2470's leading hypothesis. The error text is the whole point: it says whether
    # the registry rate-limited us, rejected auth, or was simply unreachable.
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.side_effect = RuntimeError("429 Too Many Requests")

    _run(client, _template(), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.REMOTE_DIGEST_UNREADABLE
    assert "429 Too Many Requests" in record["last_error"]
    assert "429 Too Many Requests" in record["last_remote_error"]
    assert record["remote_digest"] is None
    client.images.pull.assert_not_called()


def test_repeated_unreadable_remote_accumulates_a_count():
    # One occurrence is noise; a count in the hundreds against an unchanged local digest
    # is the answer. That distinction only exists if the counter accumulates.
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.side_effect = RuntimeError("registry unreachable")

    for _ in range(3):
        _run(client, _template(), state)

    assert _image(state)["outcome_counts"][Outcome.REMOTE_DIGEST_UNREADABLE] == 3


def test_insufficient_disk_records_both_numbers(monkeypatch):
    state = CachePrefetchState(path=None)
    monkeypatch.setattr(
        cache_template_service.psutil, "disk_usage", lambda _: MagicMock(free=1_000)
    )
    client = _make_client(image_absent=True)

    _run(client, _template(size=10_000), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.INSUFFICIENT_DISK
    assert record["last_disk_required_bytes"] == 30_000
    assert record["last_disk_available_bytes"] == 1_000
    client.images.pull.assert_not_called()


def test_lock_held(monkeypatch):
    state = CachePrefetchState(path=None)

    class _Denied:
        def __enter__(self):
            return False

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cache_template_service, "cache_pull_lock", lambda: _Denied())
    client = _make_client(image_absent=True)

    _run(client, _template(FRESH_DIGEST), state)

    assert _image(state)["last_outcome"] == Outcome.LOCK_HELD
    client.images.pull.assert_not_called()


def test_pull_ok():
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)

    _run(client, _template(FRESH_DIGEST), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.PULL_OK
    assert record["last_pull_attempt_at"] is not None
    assert record["last_pull_ok_at"] is not None
    assert record["last_pull_error"] is None


def test_pull_failed_is_recorded_and_re_raised():
    # Today a failed pull aborts the sweep and backs off. Recording must not change that.
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)
    client.images.pull.side_effect = RuntimeError("manifest unknown")

    with pytest.raises(RuntimeError):
        _run(client, _template(FRESH_DIGEST), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.PULL_FAILED
    assert "manifest unknown" in record["last_pull_error"]


def test_local_read_error_is_kept():
    state = CachePrefetchState(path=None)
    client = _make_client()
    client.images.get.side_effect = RuntimeError("daemon busy")

    _run(client, _template(FRESH_DIGEST), state)

    assert "daemon busy" in _image(state)["last_local_error"]


def test_cleanup_error_is_kept():
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)
    client.images.list.side_effect = RuntimeError("cannot list")

    _run(client, _template(FRESH_DIGEST), state)

    assert "cannot list" in _image(state)["last_cleanup_error"]


def test_digest_change_is_timestamped():
    state = CachePrefetchState(path=None)

    state.note_local_digests(IMAGE_REF, [f"{REPO}@{STALE_DIGEST}"])
    assert _image(state)["local_digest_first_seen_at"] is not None
    assert _image(state)["digest_changed_at"] is None

    state.note_local_digests(IMAGE_REF, [f"{REPO}@{FRESH_DIGEST}"])
    assert _image(state)["digest_changed_at"] is not None


def test_malformed_template_gets_its_own_field():
    # No usable image_ref exists, so it would only pollute the images map. It also cannot
    # live in the outcome slot: the sweep still finishes, and sweep_ok would bury it.
    state = CachePrefetchState(path=None)

    _run(_make_client(), {"docker_image": REPO}, state)

    doc = state.as_dict()
    assert REPO in doc["last_malformed_template"]
    assert doc["last_malformed_template_at"] is not None
    assert doc["outcome_counts"][Outcome.MALFORMED_TEMPLATE] == 1
    assert doc["images"] == {}


# --- loop-level outcomes -----------------------------------------------------------------


def _prefetch(state_path=None):
    asyncio.run(cache_template_service.run_cache_template_prefetch(state_path))


def test_prefetch_disabled_without_backend_url(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_template_service.settings, "COMPUTE_REST_API_URL", "")
    path = tmp_path / "state.json"

    _prefetch(str(path))

    doc = json.loads(path.read_text())
    assert doc["last_outcome"] == Outcome.PREFETCH_DISABLED_NO_BACKEND_URL
    assert doc["backend_url"] is None


def test_docker_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cache_template_service.settings, "COMPUTE_REST_API_URL", "https://lium.io/api"
    )
    monkeypatch.setattr(
        cache_template_service.docker,
        "from_env",
        MagicMock(side_effect=RuntimeError("no docker socket")),
    )
    path = tmp_path / "state.json"

    _prefetch(str(path))

    doc = json.loads(path.read_text())
    assert doc["last_outcome"] == Outcome.DOCKER_UNAVAILABLE
    assert doc["docker_available"] is False
    assert "no docker socket" in doc["docker_error"]
    assert doc["backend_url"].endswith("/executors/default-docker-image")


def test_gpu_unknown():
    state = CachePrefetchState(path=None)

    state.note_gpu("unknown", "unknown", error="NVML shared library not found")
    state.record_loop_outcome(Outcome.GPU_UNKNOWN, error="NVML shared library not found")

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.GPU_UNKNOWN
    assert doc["gpu_resolved_at"] is None
    assert "NVML" in doc["gpu_error"]


def test_backend_no_templates():
    state = CachePrefetchState(path=None)

    state.note_backend(status=503, template_count=0, error="HTTP 503")
    state.record_loop_outcome(Outcome.BACKEND_NO_TEMPLATES, error="HTTP 503")

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.BACKEND_NO_TEMPLATES
    assert doc["last_backend_status"] == 503
    assert doc["last_backend_template_count"] == 0


def test_loop_error():
    state = CachePrefetchState(path=None)

    state.note_loop_error(RuntimeError("connection reset"))
    state.record_loop_outcome(Outcome.LOOP_ERROR, error=RuntimeError("connection reset"))

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.LOOP_ERROR
    assert "connection reset" in doc["last_loop_error"]
    assert doc["last_loop_error_at"] is not None


def test_sweep_ok_clears_a_stale_loop_error():
    # Without this, a loop that erred once and then recovered would keep reporting
    # loop_error forever, and a reader would chase a problem that had already passed.
    state = CachePrefetchState(path=None)

    state.record_loop_outcome(Outcome.LOOP_ERROR, error="connection reset")
    state.record_loop_outcome(Outcome.SWEEP_OK)

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.SWEEP_OK
    assert doc["last_error"] is None
    # The error is still on record, just no longer the headline.
    assert doc["outcome_counts"][Outcome.LOOP_ERROR] == 1


# --- the document itself -----------------------------------------------------------------


def test_header_is_always_published():
    # started_at + sweep_count are what stop a post-recreate reset reading as a healthy node.
    state = CachePrefetchState(
        path=None, backend_url="https://lium.io/api", refresh_interval_seconds=900
    )
    state.begin_sweep()
    state.begin_sweep()

    doc = state.as_dict()
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["executor_version"]
    assert doc["started_at"]
    assert doc["updated_at"]
    assert doc["sweep_count"] == 2
    assert doc["refresh_interval_seconds"] == 900


def test_error_text_is_clipped():
    state = CachePrefetchState(path=None)

    state.record_image_outcome(IMAGE_REF, Outcome.PULL_FAILED, error="x" * 5_000)

    assert len(_image(state)["last_error"]) <= MAX_ERROR_CHARS + 1


def test_document_is_capped_and_marked_truncated():
    state = CachePrefetchState(path=None)
    for index in range(40):
        ref = f"{REPO}-{index}:{TAG}"
        state.record_image_outcome(ref, Outcome.PULL_FAILED, error="y" * MAX_ERROR_CHARS)
        state.note_local_digests(ref, [f"{REPO}@{STALE_DIGEST}"])

    payload = state.render()

    assert len(payload.encode("utf-8")) <= MAX_PAYLOAD_BYTES
    assert json.loads(payload)["truncated"] is True


def test_flush_writes_atomically_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state = CachePrefetchState(path=str(path))
    state.begin_sweep()
    state.record_image_outcome(IMAGE_REF, Outcome.UP_TO_DATE)

    state.flush()

    assert json.loads(path.read_text())["images"][IMAGE_REF]["last_outcome"] == Outcome.UP_TO_DATE
    assert list(path.parent.iterdir()) == [path]


def test_flush_failure_is_swallowed(tmp_path):
    # A read-only filesystem must cost the loop nothing.
    path = tmp_path / "state.json"
    path.mkdir()  # a directory where the file should go: os.replace will refuse
    state = CachePrefetchState(path=str(path))

    state.flush()  # must not raise

    assert path.is_dir()
