"""The default image's first pull on a new node: its real error, and a prompt retry.

A new node's first digest-pinned pull of a multi-GB image "ended" in about a second with nothing
pulled. ``images.pull`` does not raise on an ``error`` event inside the pull stream, so the state
file recorded the follow-up lookup's "No such image" instead of the registry's answer, and the loop
then waited ERROR_INTERVAL_SECONDS (5 min) before trying again while the validator was already
checking the node for that image.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import docker
import pytest

docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

from services import cache_prefetch_state, cache_template_service  # noqa: E402
from services.cache_prefetch_state import CachePrefetchState, Outcome  # noqa: E402
from services.cache_template_service import PullStreamError  # noqa: E402

REPO = "daturaai/pytorch"
TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"
IMAGE_REF = f"{REPO}:{TAG}"
DIGEST = "sha256:70bd5fa697877594b753a146e207ca4de66d9b875d606ae09e6ee7bac8f4f423"
TEMPLATE = {
    "docker_image": REPO,
    "docker_image_tag": TAG,
    "docker_image_size": 0,
    "docker_image_digest": DIGEST,
}
REGISTRY_ERROR = "read tcp 10.0.0.2:51234->203.0.113.7:443: read: connection reset by peer"


def _client(stream: list) -> MagicMock:
    client = MagicMock()
    client.images.get.side_effect = docker.errors.ImageNotFound("No such image")
    client.api.pull.return_value = iter(stream)
    client.images.list.return_value = []
    return client


# --- the pull stream's own error -----------------------------------------------------------


@pytest.mark.parametrize(
    "error_event",
    [
        {"errorDetail": {"message": REGISTRY_ERROR}, "error": REGISTRY_ERROR},
        {"error": REGISTRY_ERROR},
    ],
)
def test_pull_stream_error_is_raised_and_recorded(error_event):
    state = CachePrefetchState(path=None)
    client = _client([{"status": "Pulling fs layer", "id": "a1"}, error_event])

    with pytest.raises(PullStreamError, match="connection reset by peer"):
        asyncio.run(cache_template_service._ensure_template(client, TEMPLATE, state))

    record = state.as_dict()["images"][IMAGE_REF]
    assert record["last_outcome"] == Outcome.PULL_FAILED
    assert REGISTRY_ERROR in record["last_pull_error"]
    assert "No such image" not in record["last_pull_error"]
    # The lookup that used to hide the error is never reached.
    assert all("@" not in call.args[0] for call in client.images.get.call_args_list)


def test_clean_stream_returns_the_pulled_image():
    client = MagicMock()
    pulled = MagicMock()
    client.api.pull.return_value = iter([{"status": "Digest: " + DIGEST}])
    client.images.get.return_value = pulled

    image = cache_template_service._pull(client, REPO, DIGEST)

    assert image is pulled
    client.images.get.assert_called_once_with(f"{REPO}@{DIGEST}")


# --- the loop's retry before its first completed sweep --------------------------------------


class _Stop(BaseException):
    """Ends the loop from inside a patched sleep; the loop only catches Exception."""


def _drive_loop(monkeypatch, ensure_outcomes: list, sleeps_before_stop: int, tmp_path, gpu=None):
    """Run the loop with one template until `sleeps_before_stop` sleeps; return the delays."""
    monkeypatch.setattr(cache_template_service.settings, "COMPUTE_REST_API_URL", "https://backend")
    monkeypatch.setattr(cache_template_service.settings, "CACHE_TEMPLATE_REFRESH_SECONDS", 900)
    monkeypatch.setattr(cache_template_service.settings, "PRE_PULL_TEMPLATES_ENABLED", False)
    monkeypatch.setattr(cache_template_service.docker, "from_env", MagicMock())
    gpu = gpu or (lambda: ("NVIDIA H100", "580", None))
    monkeypatch.setattr(cache_template_service, "_get_gpu_info", gpu)
    monkeypatch.setattr(
        cache_template_service, "_fetch_templates", AsyncMock(return_value=([TEMPLATE], 200, None))
    )
    monkeypatch.setattr(
        cache_template_service, "_ensure_template", AsyncMock(side_effect=ensure_outcomes)
    )
    monkeypatch.setattr(cache_template_service.random, "uniform", lambda low, high: high)
    delays: list[float] = []

    async def sleep(seconds):
        delays.append(seconds)
        if len(delays) >= sleeps_before_stop:
            raise _Stop

    monkeypatch.setattr(cache_template_service.asyncio, "sleep", sleep)
    state_path = tmp_path / "state.json"
    with pytest.raises(_Stop):
        asyncio.run(cache_template_service.run_cache_template_prefetch(str(state_path)))
    return delays, json.loads(state_path.read_text())


def test_first_pull_failure_is_retried_within_a_minute(monkeypatch, tmp_path):
    failure = PullStreamError(REGISTRY_ERROR)
    delays, doc = _drive_loop(
        monkeypatch, [failure, failure, None, failure], sleeps_before_stop=4, tmp_path=tmp_path
    )

    jitter = cache_template_service.FIRST_SWEEP_RETRY_JITTER_SECONDS
    # Two fast retries, then the refresh sleep after the first completed sweep, then an error
    # after it backs off the full interval as before.
    assert delays == [15 + jitter, 30 + jitter, 900, cache_template_service.ERROR_INTERVAL_SECONDS]
    assert doc["first_sweep_ok_at"] is not None


def test_fast_retries_are_bounded(monkeypatch, tmp_path):
    failure = PullStreamError(REGISTRY_ERROR)
    delays, doc = _drive_loop(monkeypatch, [failure] * 6, sleeps_before_stop=6, tmp_path=tmp_path)

    jitter = cache_template_service.FIRST_SWEEP_RETRY_JITTER_SECONDS
    interval = cache_template_service.ERROR_INTERVAL_SECONDS
    assert delays == [15 + jitter, 30 + jitter, 60 + jitter, 120 + jitter, interval, interval]
    assert doc["first_sweep_ok_at"] is None
    assert doc["last_outcome"] == Outcome.LOOP_ERROR


def test_an_unknown_gpu_at_boot_uses_none_of_the_fast_retries(monkeypatch, tmp_path):
    answers = iter([("unknown", "unknown", "NVML not ready"), ("NVIDIA H100", "580", None)])
    failure = PullStreamError(REGISTRY_ERROR)
    delays, _ = _drive_loop(
        monkeypatch, [failure], sleeps_before_stop=2, tmp_path=tmp_path, gpu=lambda: next(answers)
    )

    jitter = cache_template_service.FIRST_SWEEP_RETRY_JITTER_SECONDS
    assert delays == [cache_template_service.ERROR_INTERVAL_SECONDS, 15 + jitter]


def test_retry_jitter_stays_inside_its_bound():
    for attempt in range(cache_template_service.FIRST_SWEEP_FAST_RETRIES):
        base = 15 * 2**attempt
        delay = cache_template_service._first_sweep_retry_delay(attempt, 300)
        assert base <= delay <= base + cache_template_service.FIRST_SWEEP_RETRY_JITTER_SECONDS
    assert cache_template_service._first_sweep_retry_delay(4, 300) is None


# --- the state document ------------------------------------------------------------------


def test_first_sweep_ok_at_is_set_once(monkeypatch):
    clock = iter(f"2026-09-20T10:00:0{second}Z" for second in range(10))
    monkeypatch.setattr(cache_prefetch_state, "_utcnow", lambda: next(clock))
    state = CachePrefetchState(path=None)
    assert state.as_dict()["first_sweep_ok_at"] is None

    state.record_loop_outcome(Outcome.LOOP_ERROR, error="x")
    assert state.as_dict()["first_sweep_ok_at"] is None

    state.record_loop_outcome(Outcome.SWEEP_OK)
    first = state.as_dict()["first_sweep_ok_at"]
    state.record_loop_outcome(Outcome.SWEEP_OK)

    assert first is not None
    assert state.as_dict()["first_sweep_ok_at"] == first
    assert state.as_dict()["last_outcome_at"] != first
