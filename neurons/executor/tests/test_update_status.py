"""DAH-3419: the executor reports its own update state (`GET /update-status`).

The signed digest comes from ``<COMPUTE_REST_API_URL>/watchtower/digest``; the running
digest from the runner container's image. Both are read through the real code path
(``collect_update_status``) with a fake docker client and a fake HTTP response.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import services.update_status_service as svc
from services.update_status_service import (
    ExpectedDigestCache,
    collect_update_status,
    find_runner_container,
    verify_signed_digest,
)

OLD_DIGEST = "sha256:" + "8c" * 32
NEW_DIGEST = "sha256:" + "d1" * 32
EXECUTOR_DIGEST = "sha256:" + "39" * 32
RUNNER_IMAGE = "daturaai/compute-subnet-executor-runner"


class _DockerException(Exception):
    pass


class _NotFound(_DockerException):
    """docker-py: NotFound < APIError < DockerException."""


@pytest.fixture(autouse=True)
def _docker_errors(monkeypatch):
    """conftest replaces ``docker`` with a MagicMock; give the service real exception classes."""
    monkeypatch.setattr(svc.docker, "errors", SimpleNamespace(NotFound=_NotFound, DockerException=_DockerException))


def _status(client, cache):
    return collect_update_status(lambda: client, cache)


def _container(name: str, image_id: str):
    return SimpleNamespace(name=name, attrs={"Image": image_id})


def _client(runner=None, own=None, images=None):
    """A docker client: ``containers.get`` by CVM name or own id, ``containers.list`` by label."""
    client = MagicMock()
    images = images or {}

    def containers_get(reference):
        if reference == "executor-runner" and runner is not None and runner.name == "executor-runner":
            return runner
        if own is not None and reference == "own-container-id":
            return own
        raise _NotFound(reference)

    client.containers.get.side_effect = containers_get
    client.containers.list.side_effect = lambda all, filters: (
        [runner] if runner is not None and runner.name != "executor-runner" else []
    )
    client.images.get.side_effect = lambda image_id: SimpleNamespace(attrs={"RepoDigests": images.get(image_id, [])})
    return client


def _cache(digest=None, error=None):
    cache = ExpectedDigestCache()
    cache.get = lambda now=None: (digest, error)
    return cache


# ── verify_signed_digest ──────────────────────────────────────────────────────

def test_verify_signed_digest_checks_the_validator_signature_over_digest_and_timestamp():
    """The message the validator signs is ``"<digest>:<timestamp>"`` (the same format the
    updater in watchtower/ verifies); a valid signature returns the digest."""
    keypair = MagicMock()
    keypair.verify.return_value = True
    with patch.object(svc.bittensor, "Keypair", return_value=keypair):
        digest = verify_signed_digest({"digest": NEW_DIGEST, "timestamp": 1_000_000, "signature": "0xabc"}, now=1_000_100)
    assert digest == NEW_DIGEST
    keypair.verify.assert_called_once_with(f"{NEW_DIGEST}:1000000", "0xabc")


def test_verify_signed_digest_rejects_a_bad_signature_and_a_stale_timestamp():
    """Regression: a forged or replayed response would make the node report a digest the
    validator never signed. A signature that does not verify, and a timestamp more than
    10 minutes off, are both refused."""
    keypair = MagicMock()
    keypair.verify.return_value = False
    with patch.object(svc.bittensor, "Keypair", return_value=keypair):
        with pytest.raises(ValueError, match="signature"):
            verify_signed_digest({"digest": NEW_DIGEST, "timestamp": 1_000_000, "signature": "0xabc"}, now=1_000_000)
    keypair.verify.return_value = True
    with patch.object(svc.bittensor, "Keypair", return_value=keypair):
        with pytest.raises(ValueError, match="timestamp"):
            verify_signed_digest({"digest": NEW_DIGEST, "timestamp": 1_000_000, "signature": "0xabc"}, now=1_000_000 + 601)


def test_verify_signed_digest_rejects_a_malformed_digest_before_verifying():
    """A response without a ``sha256:`` digest never reaches the key check."""
    with patch.object(svc.bittensor, "Keypair") as keypair_class:
        with pytest.raises(ValueError, match="digest"):
            verify_signed_digest({"digest": "latest", "timestamp": 1, "signature": "0x"}, now=1)
    keypair_class.assert_not_called()


# ── ExpectedDigestCache ───────────────────────────────────────────────────────

def test_expected_digest_cache_refreshes_once_per_ttl_and_keeps_the_last_good_digest():
    """Regression: every ``/update-status`` call hit the endpoint (500 nodes polled by a
    dashboard is a request storm), and one failed refresh blanked the digest. The endpoint
    is called once per TTL, and a failure after a success keeps the digest and adds the error."""
    response = MagicMock()
    response.json.return_value = {"digest": NEW_DIGEST, "timestamp": 1, "signature": "0x"}
    cache = ExpectedDigestCache(ttl_seconds=300)
    with patch.object(svc.requests, "get", return_value=response) as get, patch.object(
        svc, "verify_signed_digest", return_value=NEW_DIGEST
    ):
        assert cache.get(now=1000.0) == (NEW_DIGEST, None)
        assert cache.get(now=1100.0) == (NEW_DIGEST, None)
        assert get.call_count == 1
        get.side_effect = svc.requests.ConnectionError("endpoint down")
        assert cache.get(now=1400.0) == (NEW_DIGEST, "ConnectionError: endpoint down")
        assert get.call_count == 2


# ── find_runner_container / collect_update_status ────────────────────────────

def test_find_runner_container_uses_the_compose_label_when_the_cvm_name_is_absent():
    """The standard stack's runner is ``executor-executor-runner-1``; it is found by the
    compose service label."""
    runner = _container("executor-executor-runner-1", "sha256:runnerimage")
    client = _client(runner=runner)
    assert find_runner_container(client) is runner
    client.containers.list.assert_called_once_with(all=True, filters={"label": "com.docker.compose.service=executor-runner"})


def test_collect_update_status_reports_update_pending_when_the_runner_digest_differs(monkeypatch):
    """A node behind a stale mirror: the runner runs the old digest, the validator signed
    the new one. ``update_pending`` is true and both digests are in the report, with the
    executor's own image digest beside them."""
    monkeypatch.setenv("HOSTNAME", "own-container-id")
    runner = _container("executor-executor-runner-1", "sha256:runnerimage")
    own = _container("own", "sha256:executorimage")
    client = _client(
        runner=runner,
        own=own,
        images={
            "sha256:runnerimage": [f"{RUNNER_IMAGE}@{OLD_DIGEST}"],
            "sha256:executorimage": [f"daturaai/compute-subnet-executor@{EXECUTOR_DIGEST}"],
        },
    )

    status = _status(client, _cache(digest=NEW_DIGEST))

    assert status == {
        "runner": {
            "container": "executor-executor-runner-1",
            "running_digest": OLD_DIGEST,
            "expected_digest": NEW_DIGEST,
            "update_pending": True,
            "error": None,
        },
        "executor": {"running_digest": EXECUTOR_DIGEST},
    }


def test_collect_update_status_is_current_when_the_digests_match(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "own-container-id")
    runner = _container("executor-runner", "sha256:runnerimage")
    client = _client(runner=runner, images={"sha256:runnerimage": [f"{RUNNER_IMAGE}@{NEW_DIGEST}"]})

    status = _status(client, _cache(digest=NEW_DIGEST))

    assert status["runner"]["update_pending"] is False
    assert status["runner"]["container"] == "executor-runner"


def test_collect_update_status_is_unknown_not_current_when_a_digest_is_missing(monkeypatch):
    """Regression: a missing runner or an unreachable endpoint must not read as "current".
    ``update_pending`` is null and the error names what is missing."""
    monkeypatch.setenv("HOSTNAME", "own-container-id")
    no_runner = _status(_client(), _cache(digest=NEW_DIGEST))
    assert no_runner["runner"]["update_pending"] is None
    assert no_runner["runner"]["error"] == "runner container not found"

    runner = _container("executor-runner", "sha256:runnerimage")
    client = _client(runner=runner, images={"sha256:runnerimage": [f"{RUNNER_IMAGE}@{OLD_DIGEST}"]})
    no_endpoint = _status(client, _cache(digest=None, error="ConnectionError: endpoint down"))
    assert no_endpoint["runner"]["running_digest"] == OLD_DIGEST
    assert no_endpoint["runner"]["update_pending"] is None
    assert no_endpoint["runner"]["error"] == "ConnectionError: endpoint down"


def test_collect_update_status_reports_a_daemon_that_does_not_answer_instead_of_failing():
    """Regression: a wedged docker daemon made the route answer 500. The client factory
    raises the transport error docker-py raises for a dead socket; the report carries it
    in ``runner.error`` beside the endpoint error, and ``update_pending`` is null."""
    def factory():
        raise svc.requests.ConnectionError("docker socket refused")

    status = collect_update_status(factory, _cache(digest=None, error="ConnectionError: endpoint down"))

    assert status["runner"]["update_pending"] is None
    assert status["runner"]["running_digest"] is None
    assert status["runner"]["error"] == "ConnectionError: docker socket refused | ConnectionError: endpoint down"
    assert status["executor"]["running_digest"] is None


def test_collect_update_status_names_an_ambiguous_runner_label():
    """Two containers carry the runner label (mid `docker compose up -d`): the report says
    so instead of "runner container not found"."""
    client = _client()
    client.containers.list.side_effect = lambda all, filters: [
        _container("executor-executor-runner-1", "sha256:a"),
        _container("executor-executor-runner-1-previous-abc", "sha256:b"),
    ]

    status = _status(client, _cache(digest=NEW_DIGEST))

    assert status["runner"]["update_pending"] is None
    assert status["runner"]["error"] == "RunnerLookupError: 2 containers carry com.docker.compose.service=executor-runner"
