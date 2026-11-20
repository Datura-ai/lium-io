"""DAH-6025: the renter sees how far a slow image pull has got instead of one 'Pulling' line."""

import asyncio
from types import SimpleNamespace

import pytest
from services.docker_service import DockerService
from services.rental_docker_sdk import _track_pull_progress, pull_progress_text

# event shapes as dockerd streams them from /images/create
EVENTS = [
    {"status": "Pulling from lmsysorg/sglang", "id": "latest"},
    {"status": "Already exists", "id": "a1"},
    {"status": "Pulling fs layer", "id": "b2"},
    {"status": "Waiting", "id": "c3"},
    {"status": "Downloading", "id": "b2", "progressDetail": {"current": 1_500_000_000, "total": 3_000_000_000}},
    {"status": "Downloading", "id": "c3", "progressDetail": {"current": 200_000_000, "total": 900_000_000}},
    {"status": "Download complete", "id": "c3"},
    {"status": "Extracting", "id": "c3", "progressDetail": {"current": 100, "total": 900}},
    {"status": "Digest: sha256:abc"},
]


def test_downloaded_layer_is_not_done_until_extracted():
    progress: dict = {}
    for event in EVENTS:
        _track_pull_progress(progress, event)
    _track_pull_progress(progress, {"status": "Pull complete", "id": "c3"})

    assert pull_progress_text(progress) == "2 of 3 layers done, 1.7 GB downloaded"


@pytest.mark.parametrize(
    "statuses, done",
    [
        (["Pulling fs layer", "Downloading", "Download complete", "Extracting"], False),  # fresh, unpacking
        (["Pulling fs layer", "Downloading", "Download complete", "Extracting", "Pull complete"], True),
        (["Already exists"], True),  # cached and already extracted
        (["Already exists", "Extracting"], False),  # cached blob whose snapshot still unpacks (containerd)
        (["Already exists", "Extracting", "Pull complete"], True),
    ],
)
def test_a_layer_is_done_only_once_registered(statuses, done):
    progress: dict = {}
    for status in statuses:
        event = {"status": status, "id": "l1"}
        if status == "Downloading":
            event["progressDetail"] = {"current": 10, "total": 20}
        _track_pull_progress(progress, event)

    assert progress["l1"][1] is done


def test_progress_counts_layers_and_downloaded_bytes():
    progress: dict = {}
    assert pull_progress_text(progress) is None
    for event in EVENTS:
        _track_pull_progress(progress, event)

    assert pull_progress_text(progress) == "1 of 3 layers done, 1.7 GB downloaded"


@pytest.mark.asyncio
async def test_reporter_streams_only_changed_progress_to_the_renter():
    logs = []

    async def stream_log(msg, status, tag):
        logs.append((msg, status, tag))

    client = SimpleNamespace(pull_progress={})
    fake_service = SimpleNamespace(stream_log=stream_log)
    reporter = asyncio.ensure_future(
        DockerService._report_pull_progress(fake_service, client, "img:1", "tag", every=0.01)
    )
    await asyncio.sleep(0.05)  # nothing listed yet: no line
    for event in EVENTS:
        _track_pull_progress(client.pull_progress, event)
    await asyncio.sleep(0.05)
    reporter.cancel()

    assert logs == [("Pulling docker image img:1: 1 of 3 layers done, 1.7 GB downloaded", "success", "tag")]


@pytest.mark.asyncio
async def test_export_reporter_says_the_load_is_still_running():
    logs = []

    async def stream_log(msg, status, tag):
        logs.append(msg)

    reporter = asyncio.ensure_future(
        DockerService._report_export_progress(SimpleNamespace(stream_log=stream_log), "lium-build-x", 4_200_000_000, "tag", every=0.01)
    )
    await asyncio.sleep(0.05)
    reporter.cancel()

    assert logs and all(m.startswith("Still loading built image lium-build-x onto host (4.2 GB image, ") for m in logs)
