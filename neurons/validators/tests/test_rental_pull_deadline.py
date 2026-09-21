"""The rental image pull deadline is one hour, and a pull that outlives it fails the create.

The deadline used to be three hours. In the week to 18 Sep 2026 two customer pulls ran the full
three hours (same image, ghcr.io/tensorlink-ai/cascade-worker:worker-v0.8.0) after the renter had
deleted the pod at ~22 min — each one held a docker-sdk thread, the SSH session and the host's
download for the remaining ~2.5 h. No customer rental that reached RUNNING in the 30 days to 18 Sep
took longer than 44 min from create to ready, so an hour still covers every pull that has succeeded.
"""

import asyncio
import threading

import pytest
from services.rental_docker_sdk import (
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
)
from test_rental_docker_sdk import PullApiClient

from services import docker_service

ONE_HOUR_SECONDS = 60 * 60


class HangingPullApiClient(PullApiClient):
    """A pull whose event stream never ends until the test releases it."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.closed = False

    def _stream_helper(self, response, *, decode=False):
        self.release.wait(timeout=5)
        return iter(())

    def close(self):
        self.closed = True


def test_client_factory_and_docker_service_use_the_one_hour_deadline():
    assert RentalDockerSdkClientFactory().pull_timeout_seconds == ONE_HOUR_SECONDS
    assert docker_service._DOCKER_PULL_TIMEOUT_SECONDS == ONE_HOUR_SECONDS


@pytest.mark.asyncio
async def test_default_client_hands_the_one_hour_deadline_to_the_docker_http_call():
    # The deadline also caps the HTTP read on /images/create, so a registry that stops sending
    # bytes is cut off by the same bound as a pull that keeps trickling.
    api_client = PullApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.pull(image="registry.example/app:tag")

    assert api_client.post_calls[0]["kwargs"]["timeout"] == ONE_HOUR_SECONDS


@pytest.mark.asyncio
async def test_pull_that_outlives_the_deadline_fails_the_create_and_closes_the_client():
    api_client = HangingPullApiClient()
    client = RentalDockerSdkClient(api_client, pull_timeout_seconds=0.05)

    try:
        with pytest.raises(RentalDockerOperationError, match="pull timed out after 0.05 seconds"):
            await client.pull(image="registry.example/slow:tag")
    finally:
        api_client.release.set()

    assert api_client.closed is True


@pytest.mark.asyncio
async def test_pull_that_finishes_inside_the_deadline_succeeds():
    # Negative control: the deadline is not a delay — a pull that completes is untouched by it.
    api_client = HangingPullApiClient()
    api_client.release.set()
    client = RentalDockerSdkClient(api_client, pull_timeout_seconds=0.5)

    await asyncio.wait_for(client.pull(image="registry.example/fast:tag"), timeout=2)

    assert api_client.closed is False
