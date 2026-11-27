"""
Tests for watchtower.py - Docker image monitoring and update service.
"""

import time
import pytest
from unittest.mock import DEFAULT, Mock, patch
import docker
import requests

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from watchtower import (
    CANONICAL_REGISTRY_HOST,
    DigestMismatchError,
    RunnerLookupError,
    container_create_kwargs,
    find_runner_container,
    is_previous_runner_name,
    previous_runner_name,
    fetch_verified_digest,
    verify_watchtower_signature,
    find_container_by_name,
    pull_and_restart_containers,
    pull_image_by_digest,
    recreate_container,
    check_and_update,
    classify_runtime_probe_error,
    probe_nvidia_runtime,
    RUNTIME_PROBE_FAILED,
    RUNTIME_PROBE_LABEL,
    RUNTIME_PROBE_LABEL_KEY,
    RUNTIME_PROBE_NVIDIA_MISMATCH,
)
from models import WatchtowerDigestResponse


# ── verify_watchtower_signature ───────────────────────────────────────────────

@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_accepts_valid_signature(mock_keypair_class):
    """Should complete without raising for a valid, recent-timestamped signature."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    payload = WatchtowerDigestResponse(
        digest="sha256:abc123",
        timestamp=int(time.time()),
        signature="0xvalid_signature",
    )

    # Act / Assert — no exception raised for a valid signature
    verify_watchtower_signature(payload)
    mock_keypair.verify.assert_called_once()


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_raises_on_invalid_signature(mock_keypair_class):
    """Should raise when keypair.verify returns False."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = False
    mock_keypair_class.return_value = mock_keypair
    payload = WatchtowerDigestResponse(
        digest="sha256:abc123",
        timestamp=int(time.time()),
        signature="0xinvalid_signature",
    )

    # Act / Assert — False from verify triggers an exception with a clear message
    with pytest.raises(Exception, match="Invalid signature"):
        verify_watchtower_signature(payload)


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_uses_digest_colon_timestamp_format(mock_keypair_class):
    """signing_data passed to keypair.verify should be '{digest}:{timestamp}'."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    now = int(time.time())
    payload = WatchtowerDigestResponse(
        digest="sha256:test",
        timestamp=now,
        signature="0xsig",
    )

    # Act
    verify_watchtower_signature(payload)

    # Assert — exact signing data format matches what the validator signs
    call_args = mock_keypair.verify.call_args
    signing_data = call_args[0][0]
    assert signing_data == f"sha256:test:{now}"


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_uses_watchtower_validator_hotkey(mock_keypair_class):
    """Keypair should be constructed with the WATCHTOWER_VALIDATOR_HOTKEY constant."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    test_hotkey = "5TestHotkeyAddress"
    payload = WatchtowerDigestResponse(
        digest="sha256:abc",
        timestamp=int(time.time()),
        signature="0xsig",
    )

    # Act
    with patch('watchtower.WATCHTOWER_VALIDATOR_HOTKEY', test_hotkey):
        verify_watchtower_signature(payload)

    # Assert — hotkey constant is passed to the Keypair constructor
    mock_keypair_class.assert_called_once_with(ss58_address=test_hotkey)


@patch('watchtower.bittensor.Keypair')
def test_verify_watchtower_signature_raises_on_stale_timestamp(mock_keypair_class):
    """Should raise when payload timestamp is more than 10 minutes in the past."""
    # Arrange
    mock_keypair = Mock()
    mock_keypair.verify.return_value = True
    mock_keypair_class.return_value = mock_keypair
    payload = WatchtowerDigestResponse(
        digest="sha256:abc123",
        timestamp=1000,  # far in the past
        signature="0xsig",
    )

    # Act / Assert — stale timestamp is rejected even with an otherwise valid signature
    with pytest.raises(Exception, match="timestamp out of allowed range"):
        verify_watchtower_signature(payload)


# ── fetch_verified_digest ─────────────────────────────────────────────────────

@patch('watchtower.verify_watchtower_signature')
@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_digest_on_success(mock_get, mock_verify):
    """Should return the digest when the endpoint responds and signature verifies."""
    # Arrange
    mock_response = Mock()
    mock_response.json.return_value = {
        "digest": "sha256:newdigest",
        "timestamp": 1234567890,
        "signature": "0xsignature",
    }
    mock_get.return_value = mock_response

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — verified digest is returned and the correct URL was used
    assert digest == "sha256:newdigest"
    mock_get.assert_called_once_with(
        "http://test-endpoint.com/digest", headers={"User-Agent": "lium-watchtower/1.2.5"}, timeout=30
    )
    mock_verify.assert_called_once()


def test_the_version_the_digest_request_announces_is_the_pyproject_version():
    # regression: pyproject.toml is bumped for a release and the User-Agent keeps the old number, so the
    # platform's per-version count (the read that gates the wallet swap) reports a fleet that never updated
    import tomllib
    from pathlib import Path

    import config

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        assert config.WATCHTOWER_VERSION == tomllib.load(handle)["project"]["version"]
    assert config.WATCHTOWER_USER_AGENT == f"lium-watchtower/{config.WATCHTOWER_VERSION}"


@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_on_request_exception(mock_get):
    """Should return None when the HTTP request itself fails."""
    # Arrange
    mock_get.side_effect = requests.RequestException("Connection error")

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — network errors are handled and None is returned
    assert digest is None


@patch('watchtower.verify_watchtower_signature')
@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_when_signature_invalid(mock_get, mock_verify):
    """Should return None when signature verification raises."""
    # Arrange
    mock_response = Mock()
    mock_response.json.return_value = {
        "digest": "sha256:tampered",
        "timestamp": 1234567890,
        "signature": "0xbadsignature",
    }
    mock_get.return_value = mock_response
    mock_verify.side_effect = Exception("Invalid signature")

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — verification failure causes None to be returned
    assert digest is None


@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_on_http_error(mock_get):
    """Should return None when the endpoint returns an HTTP error status."""
    # Arrange
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — HTTP errors propagate as None
    assert digest is None


@patch('watchtower.requests.get')
def test_fetch_verified_digest_returns_none_on_invalid_json(mock_get):
    """Should return None when the response body is not valid JSON."""
    # Arrange
    mock_response = Mock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response

    # Act
    with patch('watchtower.WATCHTOWER_ENDPOINT_URL', "http://test-endpoint.com/digest"):
        digest = fetch_verified_digest()

    # Assert — malformed response is handled gracefully
    assert digest is None


# ── find_container_by_name ────────────────────────────────────────────────────

def test_find_container_by_name_returns_container_when_found():
    """Should return the container object when it exists by name."""
    # Arrange
    mock_client = Mock()
    mock_container = Mock()
    mock_container.short_id = "abc123"
    mock_container.status = "running"
    mock_client.containers.get.return_value = mock_container

    # Act
    result = find_container_by_name(mock_client, "executor-runner")

    # Assert — correct container object returned and looked up by name
    assert result is mock_container
    mock_client.containers.get.assert_called_once_with("executor-runner")


def test_find_container_by_name_returns_none_when_not_found():
    """Should return None when the container does not exist."""
    # Arrange
    mock_client = Mock()
    mock_client.containers.get.side_effect = docker.errors.NotFound("not found")

    # Act
    result = find_container_by_name(mock_client, "executor-runner")

    # Assert — NotFound is handled and None returned
    assert result is None


def test_find_container_by_name_returns_none_on_exception():
    """Should return None on any unexpected Docker error."""
    # Arrange
    mock_client = Mock()
    mock_client.containers.get.side_effect = Exception("Docker error")

    # Act
    result = find_container_by_name(mock_client, "executor-runner")

    # Assert — unexpected errors are swallowed and None returned
    assert result is None


# ── fixtures: a runner container as `docker inspect` reports it ──────────────

OLD_DIGEST = "sha256:" + "8c" * 32
NEW_DIGEST = "sha256:" + "d1" * 32
IMAGE = "daturaai/compute-subnet-executor-runner"
COMPOSE_LABELS = {
    "com.docker.compose.project": "executor",
    "com.docker.compose.service": "executor-runner",
    "com.docker.compose.config-hash": "abc",
}


def _fake_image(digest: str, cmd=None, entrypoint=None):
    image = Mock()
    image.attrs = {
        "RepoDigests": [f"{IMAGE}@{digest}"],
        "Config": {"Cmd": cmd, "Entrypoint": entrypoint or ["/entrypoint.sh"]},
    }
    return image


def _fake_container(name="executor-executor-runner-1", cmd=None, entrypoint=None, status="running", container_id=None):
    """The standard stack's runner: compose labels, `.env` bind, `unless-stopped`, one network."""
    container = Mock()
    container.name = name
    container.id = container_id or "0123456789abcdef" * 4
    container.short_id = container.id[:12]
    container.status = status
    container.attrs = {
        "Id": container.id,
        "Name": f"/{name}",
        "Image": "sha256:oldimageid",
        "Config": {
            "Cmd": cmd,
            "Entrypoint": entrypoint or ["/entrypoint.sh"],
            "Env": ["PATH=/usr/local/bin:/usr/bin"],
            "Labels": COMPOSE_LABELS,
            "WorkingDir": "/root/executor",
            "User": "",
        },
        "HostConfig": {
            "Binds": [
                "/var/run/docker.sock:/var/run/docker.sock:rw",
                "/home/lium/compute-subnet/neurons/executor/.env:/root/executor/.env:rw",
            ],
            "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
            "NetworkMode": "executor_default",
        },
        "NetworkSettings": {
            "Networks": {"executor_default": {"Aliases": ["executor-runner", container.id[:12]]}},
        },
    }
    return container


def _client_with(container, old_image=None, pulled_image=None, stale_probes=()):
    client = Mock()
    client.images.get.return_value = old_image or _fake_image(OLD_DIGEST)
    client.images.pull.return_value = pulled_image or _fake_image(NEW_DIGEST)
    created = Mock()
    created.name, created.short_id = "executor-executor-runner-1", "newid"

    def containers_get(reference):
        if reference == "executor-runner":
            raise docker.errors.NotFound("no CVM-named runner")
        return created

    client.containers.get.side_effect = containers_get
    client.containers.list.return_value = [container] if container else []
    # the probe's own listing (B-176, stale probes by label) answers `client.stale_probes`; every
    # other listing keeps `return_value`, which the runner-lookup tests set per case
    client.stale_probes = list(stale_probes)

    def containers_list(all=False, filters=None):
        if (filters or {}).get("label") == RUNTIME_PROBE_LABEL:
            return list(client.stale_probes)
        return DEFAULT

    client.containers.list.side_effect = containers_list
    # the runtime probe container (B-176): starts and exits 0 unless a test says otherwise
    client.containers.create.return_value.wait.return_value = {"StatusCode": 0}
    client.api.create_container.return_value = {"Id": "newid"}
    client.api.create_endpoint_config.side_effect = lambda aliases=None: {"Aliases": aliases}
    client.api.create_networking_config.side_effect = lambda endpoints: {"EndpointsConfig": endpoints}
    return client


# ── pull_image_by_digest ─────────────────────────────────────────────────────

def test_pull_image_by_digest_pulls_the_digest_reference_not_the_tag():
    """Regression: a pull by tag lets a registry mirror answer with an old image (P119,
    DAH-3419). The daemon is asked for `image@digest`, and the returned reference is
    the one the container is created from."""
    client = _client_with(None)

    reference = pull_image_by_digest(client, IMAGE, NEW_DIGEST)

    assert reference == f"{IMAGE}@{NEW_DIGEST}"
    client.images.pull.assert_called_once_with(f"{IMAGE}@{NEW_DIGEST}")


def test_pull_image_by_digest_falls_back_to_docker_hub_when_the_daemon_path_fails():
    """Regression: the mirror answers 404 (or a proxy times out) for a digest it has not
    cached, and the update stalls. The second pull names `registry-1.docker.io`, which the
    daemon's `registry-mirrors` do not apply to."""
    client = _client_with(None)
    client.images.pull.side_effect = [docker.errors.NotFound("manifest unknown"), _fake_image(NEW_DIGEST)]

    reference = pull_image_by_digest(client, IMAGE, NEW_DIGEST)

    assert reference == f"{CANONICAL_REGISTRY_HOST}/{IMAGE}@{NEW_DIGEST}"
    assert [c.args[0] for c in client.images.pull.call_args_list] == [
        f"{IMAGE}@{NEW_DIGEST}",
        f"{CANONICAL_REGISTRY_HOST}/{IMAGE}@{NEW_DIGEST}",
    ]


def test_pull_image_by_digest_rejects_an_image_without_the_requested_digest():
    """Regression: the daemon resolves the reference to a cached image whose RepoDigests
    do not carry the digest. Both attempts return such an image; the pull fails instead
    of restarting the runner on the wrong image."""
    client = _client_with(None)
    client.images.pull.return_value = _fake_image(OLD_DIGEST)

    with pytest.raises(DigestMismatchError):
        pull_image_by_digest(client, IMAGE, NEW_DIGEST)

    assert client.images.pull.call_count == 2


# ── find_runner_container ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_own_container(monkeypatch, tmp_path):
    """Outside a container by default (CVM stack, `docker run`): the runner lookup is not scoped."""
    monkeypatch.setattr("watchtower.MOUNTINFO_PATH", str(tmp_path / "absent"))


OWN_ID = "e" * 64


def _in_project(monkeypatch, tmp_path, client, project):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 0:1 / / rw - overlay overlay rw\n"
        f"2 1 259:1 /var/lib/docker/containers/{OWN_ID}/hostname /etc/hostname rw - ext4 /dev/root rw\n"
    )
    monkeypatch.setattr("watchtower.MOUNTINFO_PATH", str(mountinfo))
    own = Mock()
    own.labels = {"com.docker.compose.project": project} if project else {}
    get = client.containers.get.side_effect

    def containers_get(reference):
        return own if reference == OWN_ID else get(reference)

    client.containers.get.side_effect = containers_get


def test_find_runner_container_is_scoped_to_its_own_compose_project(monkeypatch, tmp_path):
    """Regression: a host with this stack's runner and a stopped runner from another compose project
    matched the service label twice, so every cycle raised and the node never updated. The lookup
    now asks the daemon for this watchtower's own project only."""
    runner = _fake_container()
    client = _client_with(runner)
    _in_project(monkeypatch, tmp_path, client, "executor")

    assert find_runner_container(client) is runner
    assert client.containers.list.call_args_list[0].kwargs == {
        "all": True,
        "filters": {"label": ["com.docker.compose.service=executor-runner", "com.docker.compose.project=executor"]},
    }


def test_find_runner_container_is_not_scoped_without_a_compose_project(monkeypatch, tmp_path):
    """A watchtower started with `docker run` has no project label: the lookup stays host-wide."""
    runner = _fake_container()
    client = _client_with(runner)
    _in_project(monkeypatch, tmp_path, client, None)

    assert find_runner_container(client) is runner
    assert client.containers.list.call_args_list[0].kwargs == {
        "all": True, "filters": {"label": "com.docker.compose.service=executor-runner"}
    }


@pytest.mark.parametrize("error", [docker.errors.APIError("daemon busy"), docker.errors.NotFound("gone")])
def test_find_runner_container_fails_closed_when_its_own_container_cannot_be_inspected(monkeypatch, tmp_path, error):
    """Regression: a failed inspect of its own container read as "not in compose", and the next calls
    searched every stack on the host by name. It now raises before any lookup, so nothing is touched."""
    client = _client_with(_fake_container(name="executor-runner"))
    _in_project(monkeypatch, tmp_path, client, "executor")
    get = client.containers.get.side_effect

    def containers_get(reference):
        if reference == OWN_ID:
            raise error
        return get(reference)

    client.containers.get.side_effect = containers_get

    with pytest.raises(RunnerLookupError):
        find_runner_container(client)
    assert [c.args for c in client.containers.get.call_args_list] == [(OWN_ID,)]
    client.containers.list.assert_not_called()


@pytest.mark.parametrize(
    "project, listed, expected, removed",
    [
        # scoped: the daemon answers only this project's runner; a foreign CVM-named container and a
        # foreign `executor-runner-previous-*` are never looked up, so never returned or removed
        ("executor", ["runner"], "runner", []),
        # scoped, this project's own leftover beside its runner: cleared as before
        ("executor", ["runner", "own_leftover"], "runner", ["own_leftover"]),
        # unscoped (CVM / docker run): the CVM name wins, as before
        (None, [], "cvm", []),
        # unscoped, no CVM runner: the CVM leftover is found by name and restored, as before
        (None, [], "cvm_leftover", []),
    ],
)
def test_find_runner_container_name_lookups_only_outside_a_project(monkeypatch, tmp_path, project, listed, expected, removed):
    runner = _fake_container()
    own_leftover = _fake_container(name="executor-executor-runner-1-previous-abc123", status="exited")
    cvm = _fake_container(name="executor-runner")
    cvm_leftover = _fake_container(name="executor-runner-previous-abcd", status="exited")
    by_name = {"runner": runner, "own_leftover": own_leftover, "cvm": cvm, "cvm_leftover": cvm_leftover}
    client = _client_with(None)
    _in_project(monkeypatch, tmp_path, client, project)
    get = client.containers.get.side_effect

    def containers_get(reference):
        if reference == "executor-runner" and expected == "cvm":
            return cvm
        if reference == "executor-runner":
            raise docker.errors.NotFound("no CVM runner")
        return get(reference)

    client.containers.get.side_effect = containers_get

    def containers_list(all=False, filters=None):
        if "name" in (filters or {}):
            return [cvm_leftover] if expected == "cvm_leftover" else []
        return [by_name[n] for n in listed]

    client.containers.list.side_effect = containers_list

    assert find_runner_container(client) is by_name[expected]
    if project:
        assert not any("name" in (c.kwargs.get("filters") or {}) for c in client.containers.list.call_args_list)
        assert not any(c.args == ("executor-runner",) for c in client.containers.get.call_args_list)
    for name, container in by_name.items():
        assert container.remove.called == (name in removed)


def test_find_runner_container_prefers_the_cvm_name():
    """The CVM stack names the container `executor-runner`; that lookup wins and no label query runs."""
    client = Mock()
    cvm = _fake_container(name="executor-runner")
    client.containers.get.return_value = cvm

    assert find_runner_container(client) is cvm
    client.containers.list.assert_not_called()


def test_find_runner_container_finds_the_compose_service_by_label():
    """Regression: the standard stack's container is `executor-executor-runner-1`, so a
    lookup by the CVM name finds nothing and the node is never updated. The compose
    service label identifies it whatever the project name is."""
    runner = _fake_container()
    client = _client_with(runner)

    assert find_runner_container(client) is runner
    assert client.containers.list.call_args_list[0].kwargs == {
        "all": True, "filters": {"label": "com.docker.compose.service=executor-runner"}
    }


def test_find_runner_container_raises_when_the_label_is_ambiguous():
    """Regression: two containers carry the service label (while `docker compose up -d`
    renames the old runner and creates the new one, or a second project on the host) and
    the lookup answered None, which reads as "no runner" and leads to a pull and a
    CVM-shaped `executor-runner` created beside the real ones. The lookup now raises."""
    client = _client_with(None)
    client.containers.list.return_value = [_fake_container(), _fake_container(name="other-executor-runner-1")]

    with pytest.raises(RunnerLookupError):
        find_runner_container(client)


def test_find_runner_container_raises_when_listing_fails():
    """A docker error while listing is not "no runner": it raises, so nothing is created."""
    client = _client_with(None)
    client.containers.list.side_effect = docker.errors.APIError("daemon busy")

    with pytest.raises(RunnerLookupError):
        find_runner_container(client)


def _leftover(runner, status="exited"):
    """The old runner as `recreate_container` leaves it when it dies between the rename and the remove."""
    return _fake_container(
        name=previous_runner_name(runner.name, "fedcba987654"),
        status=status,
        container_id="fedcba9876543210" * 4,
    )


def test_find_runner_container_skips_and_removes_a_leftover_previous_container():
    """Regression (taiberium, #1359): `recreate_container` died between renaming the old runner
    to `<name>-previous-<id>` and removing it. The leftover keeps the compose service label, so
    the lookup saw two runners and raised on every cycle: the node never updated again. The
    leftover is recognised by its name, removed (the step the interrupted update did not reach)
    and the runner under the real name is returned."""
    runner = _fake_container()
    leftover = _leftover(runner)
    client = _client_with(runner)
    client.containers.list.return_value = [leftover, runner]

    assert find_runner_container(client) is runner
    leftover.remove.assert_called_once_with(force=True)
    runner.remove.assert_not_called()


def test_find_runner_container_finishes_the_switch_beside_a_runner_that_was_never_started():
    """Regression: the update died between `create_container` and `start`. The runner under the
    real name is `created` with the signed digest, so the digest check would say "up to date" on
    every cycle while nothing runs (or only the leftover does). The lookup finishes the switch:
    starts the runner, removes the leftover."""
    runner = _fake_container(status="created")
    leftover = _leftover(runner, status="running")
    client = _client_with(runner)
    client.containers.list.return_value = [runner, leftover]

    assert find_runner_container(client) is runner
    runner.start.assert_called_once_with()
    leftover.remove.assert_called_once_with(force=True)


def test_find_runner_container_keeps_both_when_the_stopped_runner_cannot_start():
    """The runner under the real name does not start (a port taken, a broken image), so the
    leftover may be the only working copy: nothing is removed, the runner is still returned, and
    the cycle goes on instead of raising for ever."""
    runner = _fake_container(status="exited")
    runner.start.side_effect = docker.errors.APIError("port is already allocated")
    leftover = _leftover(runner, status="running")
    client = _client_with(runner)
    client.containers.list.return_value = [runner, leftover]

    assert find_runner_container(client) is runner
    leftover.remove.assert_not_called()


def test_find_runner_container_finds_the_cvm_leftover_by_name():
    """The CVM runner `executor-runner` carries no compose label, so its leftover is not in the
    label listing. Without the name listing the lookup answered None and a second runner was
    created beside the leftover. It is found by name, renamed back and returned."""
    cvm = _fake_container(name="executor-runner")
    leftover = _leftover(cvm, status="running")
    client = _client_with(None)
    client.containers.list.side_effect = lambda all, filters: [leftover] if "name" in filters else []

    assert find_runner_container(client) is leftover
    leftover.rename.assert_called_once_with("executor-runner")
    client.containers.run.assert_not_called()


def test_find_runner_container_still_returns_the_runner_when_the_leftover_cannot_be_removed():
    """A daemon error on the remove does not turn into "no runner" or a raise: the runner is
    returned and the remove is retried on the next cycle."""
    runner = _fake_container()
    leftover = _leftover(runner)
    leftover.remove.side_effect = docker.errors.APIError("device busy")
    client = _client_with(runner)
    client.containers.list.return_value = [runner, leftover]

    assert find_runner_container(client) is runner


def test_find_runner_container_restores_a_lone_leftover():
    """`recreate_container` died after the rename and before the create (or the create failed
    and the rename back failed): the leftover is the only runner on the host. It gets its name
    back and is returned, so the next update recreates it under the compose name."""
    runner = _fake_container()
    leftover = _leftover(runner, status="running")
    client = _client_with(None)
    client.containers.list.return_value = [leftover]

    assert find_runner_container(client) is leftover
    leftover.rename.assert_called_once_with("executor-executor-runner-1")
    leftover.remove.assert_not_called()


def test_find_runner_container_raises_when_only_leftovers_remain():
    """Two leftovers and no runner: which one to restore is a person's call. (Two label matches
    raised before this change too; the test pins that the leftover branch keeps raising here.)"""
    runner = _fake_container()
    first = _leftover(runner)
    second = _fake_container(name=previous_runner_name(runner.name, "0123456789ab"), status="exited", container_id="a" * 64)
    client = _client_with(None)
    client.containers.list.return_value = [first, second]

    with pytest.raises(RunnerLookupError):
        find_runner_container(client)
    first.remove.assert_not_called()
    second.remove.assert_not_called()


def test_previous_runner_name_is_what_the_lookup_recognises():
    """Pins the two sides of the suffix to each other: a rename in `recreate_container` that the
    lookup does not recognise brings the stuck cycle back."""
    assert is_previous_runner_name(previous_runner_name("executor-executor-runner-1", "a41fe4ee0a94"))
    assert not is_previous_runner_name("executor-executor-runner-1")
    assert not is_previous_runner_name("other-executor-runner-1")
    assert not is_previous_runner_name("executor-runner-previous-build")


# ── container_create_kwargs / recreate_container ─────────────────────────────

def test_container_create_kwargs_copies_the_compose_configuration():
    """Regression: the recreated runner used to get two hard-coded binds and no labels, so
    compose no longer recognised it and the `.env` bind pointed at `~/.env`. The old
    container's name, labels, binds, restart policy and network are what the new one gets."""
    runner = _fake_container()
    client = _client_with(runner)

    kwargs = container_create_kwargs(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    assert kwargs["image"] == f"{IMAGE}@{NEW_DIGEST}"
    assert kwargs["name"] == "executor-executor-runner-1"
    assert kwargs["labels"] == COMPOSE_LABELS
    assert kwargs["host_config"] is runner.attrs["HostConfig"]
    assert kwargs["environment"] == ["PATH=/usr/local/bin:/usr/bin"]
    assert kwargs["working_dir"] == "/root/executor"
    assert kwargs["user"] is None
    assert kwargs["networking_config"] == {
        "EndpointsConfig": {"executor_default": {"Aliases": ["executor-runner"]}}
    }


def test_container_create_kwargs_leaves_inherited_cmd_and_entrypoint_to_the_new_image():
    """Regression: copying the Cmd/Entrypoint the old container took from its image pins
    the new container to the old image's entrypoint. Values equal to the old image's are
    dropped; a value set on the container itself is kept."""
    runner = _fake_container(cmd=["--interval", "60"], entrypoint=["/entrypoint.sh"])
    client = _client_with(runner, old_image=_fake_image(OLD_DIGEST, cmd=None, entrypoint=["/entrypoint.sh"]))

    kwargs = container_create_kwargs(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    assert kwargs["entrypoint"] is None
    assert kwargs["command"] == ["--interval", "60"]


def test_container_create_kwargs_takes_no_execution_settings_from_an_unverified_image():
    """DAH-5942 F-4: a runner made by `docker compose create` holds an image nobody verified.
    Env entries, working directory and user that came from that image are left to the signed
    image, compose-set values are kept, and an uninspectable image refuses the rebuild instead
    of copying its entrypoint."""
    runner = _fake_container(status="created")
    runner.attrs["Config"]["Env"] = ["PATH=/usr/bin", "LD_PRELOAD=/evil.so", "COMPOSE_SET=1"]
    runner.attrs["Config"]["WorkingDir"] = "/evil"
    runner.attrs["Config"]["User"] = "evil"
    image = _fake_image(OLD_DIGEST)
    image.attrs["Config"].update({"Env": ["PATH=/usr/bin", "LD_PRELOAD=/evil.so"], "WorkingDir": "/evil", "User": "evil"})
    client = _client_with(runner, old_image=image)

    kwargs = container_create_kwargs(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    assert kwargs["environment"] == ["COMPOSE_SET=1"]
    assert kwargs["working_dir"] is None
    assert kwargs["user"] is None
    assert kwargs["entrypoint"] is None

    client.images.get.side_effect = docker.errors.ImageNotFound("gone")
    with pytest.raises(RuntimeError):
        container_create_kwargs(client, runner, f"{IMAGE}@{NEW_DIGEST}")


def test_recreate_container_keeps_a_runner_on_the_host_at_every_step():
    """Regression: stop and remove the old container, then fail to create the new one, and
    the host has no runner until the next cycle creates a CVM-shaped one. The order is:
    rename the old aside, create the new under the old name, stop the old, start the new,
    remove the old. The old container is removed only after the new one runs."""
    runner = _fake_container()
    client = _client_with(runner)
    order = []
    runner.rename.side_effect = lambda name: order.append(("rename", name))
    runner.stop.side_effect = lambda timeout: order.append("stop")
    runner.remove.side_effect = lambda force: order.append("remove")
    client.api.create_container.side_effect = lambda **kwargs: order.append(("create", kwargs["name"])) or {"Id": "newid"}
    client.api.start.side_effect = lambda cid: order.append(("start", cid))

    recreate_container(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    assert order == [
        ("rename", f"executor-executor-runner-1-previous-{runner.short_id}"),
        ("create", "executor-executor-runner-1"),
        "stop",
        ("start", "newid"),
        "remove",
    ]
    runner.stop.assert_called_once_with(timeout=10)


def test_recreate_container_puts_the_old_name_back_when_create_fails():
    """The daemon refuses the create (bad HostConfig, disk full): the old container gets its
    name back, is never stopped, and the error propagates."""
    runner = _fake_container()
    client = _client_with(runner)
    client.api.create_container.side_effect = docker.errors.APIError("no space left")

    with pytest.raises(docker.errors.APIError):
        recreate_container(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    assert [c.args[0] for c in runner.rename.call_args_list] == [
        f"executor-executor-runner-1-previous-{runner.short_id}",
        "executor-executor-runner-1",
    ]
    runner.stop.assert_not_called()
    runner.remove.assert_not_called()


def test_recreate_container_restores_the_old_runner_when_the_old_one_does_not_stop():
    """Regression: `stop` fails (daemon timeout) after the new container was created under
    the real name, and the host keeps both: a running renamed old runner and a created,
    never started new one. The new container is removed and the old one gets its name back."""
    runner = _fake_container()
    client = _client_with(runner)
    runner.stop.side_effect = docker.errors.APIError("stop timed out")

    with pytest.raises(docker.errors.APIError):
        recreate_container(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    client.api.remove_container.assert_called_once_with("newid", force=True)
    client.api.start.assert_not_called()
    assert runner.rename.call_args_list[-1].args[0] == "executor-executor-runner-1"
    runner.remove.assert_not_called()


def test_recreate_container_restores_the_old_runner_when_the_new_one_does_not_start():
    """Regression: the new container is created but does not start, and the host is left
    with a stopped runner on the new digest that reads "up to date" forever. The new
    container is removed, the old one gets its name back and is started again."""
    runner = _fake_container()
    client = _client_with(runner)
    client.api.start.side_effect = docker.errors.APIError("cannot start")

    with pytest.raises(docker.errors.APIError):
        recreate_container(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    client.api.remove_container.assert_called_once_with("newid", force=True)
    assert runner.rename.call_args_list[-1].args[0] == "executor-executor-runner-1"
    runner.start.assert_called_once()
    runner.remove.assert_not_called()


def test_recreate_container_never_starts_a_runner_that_was_only_created_when_the_new_one_does_not_start():
    """A runner made by `docker compose create` holds an unverified image: when the signed
    one does not start, it gets its name back but stays stopped for the next cycle."""
    runner = _fake_container(status="created")
    client = _client_with(runner)
    client.api.start.side_effect = docker.errors.APIError("cannot start")

    with pytest.raises(docker.errors.APIError):
        recreate_container(client, runner, f"{IMAGE}@{NEW_DIGEST}")

    client.api.remove_container.assert_called_once_with("newid", force=True)
    assert runner.rename.call_args_list[-1].args[0] == "executor-executor-runner-1"
    runner.start.assert_not_called()


# ── pull_and_restart_containers ───────────────────────────────────────────────

def test_pull_and_restart_containers_rebuilds_the_compose_runner_from_the_digest():
    """The standard stack: pull `image@digest`, then recreate `executor-executor-runner-1`
    with its own configuration. The CVM-style `containers.run` fallback is not used."""
    runner = _fake_container()
    client = _client_with(runner)

    result = pull_and_restart_containers(client, IMAGE, NEW_DIGEST)

    assert result is True
    client.images.pull.assert_called_once_with(f"{IMAGE}@{NEW_DIGEST}")
    runner.stop.assert_called_once_with(timeout=10)
    runner.remove.assert_called_once_with(force=True)
    assert client.api.create_container.call_args.kwargs["image"] == f"{IMAGE}@{NEW_DIGEST}"
    assert client.api.create_container.call_args.kwargs["name"] == "executor-executor-runner-1"
    client.containers.run.assert_not_called()


def test_pull_and_restart_containers_creates_container_when_none_exists():
    """No runner on the host (first CVM boot): one is created from the digest reference with
    the CVM configuration."""
    client = _client_with(None)

    result = pull_and_restart_containers(client, IMAGE, NEW_DIGEST)

    assert result is True
    client.api.create_container.assert_not_called()
    assert client.containers.run.call_args.args[0] == f"{IMAGE}@{NEW_DIGEST}"
    assert client.containers.run.call_args.kwargs["name"] == "executor-runner"


def test_pull_and_restart_containers_returns_false_and_keeps_the_runner_when_both_pulls_fail():
    """Both pull attempts fail (mirror and Docker Hub): the result is False and the
    runner is not stopped."""
    runner = _fake_container()
    client = _client_with(runner)
    client.images.pull.side_effect = docker.errors.APIError("Pull failed")

    result = pull_and_restart_containers(client, IMAGE, NEW_DIGEST)

    assert result is False
    assert client.images.pull.call_count == 2
    runner.stop.assert_not_called()


def test_pull_and_restart_containers_returns_false_when_the_runner_cannot_be_identified():
    """Regression: an ambiguous lookup after the pull led to a CVM-shaped `executor-runner`
    beside the real one. Nothing is created; the result is False."""
    client = _client_with(None)
    client.containers.list.return_value = [_fake_container(), _fake_container(name="other-executor-runner-1")]

    result = pull_and_restart_containers(client, IMAGE, NEW_DIGEST)

    assert result is False
    client.containers.run.assert_not_called()
    client.api.create_container.assert_not_called()


def test_pull_and_restart_containers_returns_false_on_unexpected_error():
    """Should return False on any unexpected exception."""
    client = _client_with(None)
    client.images.pull.return_value = _fake_image(NEW_DIGEST)
    client.containers.run.side_effect = Exception("boom")

    result = pull_and_restart_containers(client, IMAGE, NEW_DIGEST)

    assert result is False


# ── check_and_update ──────────────────────────────────────────────────────────

@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_pulls_when_digests_differ(mock_settings, mock_docker, mock_fetch, mock_pull):
    """The running runner's digest is read from the container found by label; it differs
    from the signed digest, so the update runs with that digest."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    client = _client_with(_fake_container())
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST
    mock_pull.return_value = True

    check_and_update()

    mock_pull.assert_called_once_with(client, IMAGE, NEW_DIGEST)


@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_pulls_nothing_when_the_runner_is_not_identified(mock_settings, mock_docker, mock_fetch, mock_pull):
    """Regression: two labelled containers (mid `docker compose up -d`) read as "no runner",
    the signed digest was fetched and a runner was pulled and created. The cycle ends
    before the endpoint is called."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    client = _client_with(None)
    client.containers.list.return_value = [_fake_container(), _fake_container(name="other-executor-runner-1")]
    mock_docker.return_value = client

    check_and_update()

    mock_fetch.assert_not_called()
    mock_pull.assert_not_called()


@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_goes_on_when_a_leftover_previous_container_sits_beside_the_runner(
    mock_settings, mock_docker, mock_fetch, mock_pull
):
    """Regression (taiberium, #1359): the listing holds the runner and a `-previous-` leftover
    of an interrupted update. The cycle used to end at the lookup on every run; now the
    leftover goes, the runner's digest is compared and the update runs."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    runner = _fake_container()
    leftover = _leftover(runner)
    client = _client_with(runner)
    client.containers.list.return_value = [leftover, runner]
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST
    mock_pull.return_value = True

    check_and_update()

    leftover.remove.assert_called_once_with(force=True)
    mock_pull.assert_called_once_with(client, IMAGE, NEW_DIGEST)


@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.container_image_digest')
@patch('watchtower.find_runner_container')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_skips_pull_when_digests_match(
    mock_settings, mock_docker, mock_find, mock_get_digest, mock_fetch, mock_pull
):
    """Should not pull when the current digest already matches the remote one."""
    # Arrange
    mock_settings.WATCHTOWER_IMAGE = "test-image"
    mock_docker.return_value = Mock()
    mock_find.return_value = _fake_container()
    mock_get_digest.return_value = "sha256:same"
    mock_fetch.return_value = "sha256:same"

    # Act
    check_and_update()

    # Assert — identical digests mean the image is already up to date
    mock_pull.assert_not_called()


@patch('watchtower.fetch_verified_digest')
@patch('watchtower.container_image_digest')
@patch('watchtower.find_runner_container')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_skips_when_remote_fetch_fails(
    mock_settings, mock_docker, mock_find, mock_get_digest, mock_fetch
):
    """Should return early without error when the remote digest cannot be fetched."""
    # Arrange
    mock_settings.WATCHTOWER_IMAGE = "test-image"
    mock_docker.return_value = Mock()
    mock_find.return_value = _fake_container()
    mock_get_digest.return_value = "sha256:current"
    mock_fetch.return_value = None

    # Act / Assert — None from fetch does not cause an exception
    check_and_update()


@patch('watchtower.docker.from_env')
def test_check_and_update_handles_docker_connection_error(mock_docker):
    """Should handle Docker client initialisation errors without raising."""
    # Arrange
    mock_docker.side_effect = Exception("Docker connection error")

    # Act / Assert — top-level exception handler prevents propagation
    check_and_update()


# ── B-176: the update waits for a working NVIDIA runtime ─────────────────────

NVML_MISMATCH = (
    "500 Server Error: Internal Server Error (\"failed to create task for container: failed to "
    "create shim task: OCI runtime create failed: runc create failed: unable to start container "
    "process: error during container init: error running prestart hook #0: exit status 1, stdout: , "
    "stderr: Auto-detected mode as 'legacy'\nnvidia-container-cli: initialization error: nvml error: "
    "driver/library version mismatch: unknown\")"
)


def test_classify_runtime_probe_error_names_the_driver_library_mismatch():
    """The daemon's message for a driver package updated without a reboot (ticket-0325) is the
    mismatch code; any other start failure is the generic one (negative control)."""
    assert classify_runtime_probe_error(NVML_MISMATCH) == RUNTIME_PROBE_NVIDIA_MISMATCH
    assert classify_runtime_probe_error("500 Server Error: no such image") == RUNTIME_PROBE_FAILED


@pytest.mark.parametrize("nvml_error", [
    # persistence daemon stopped / device node gone: NVML initialises and fails, no version disagreement
    "nvidia-container-cli: initialization error: nvml error: unknown error: unknown",
    # the container toolkit cannot find the driver library at all: a toolkit or driver install problem
    "nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file",
    "nvidia-container-cli: initialization error: nvml error: driver not loaded: unknown",
])
def test_classify_runtime_probe_error_keeps_other_nvml_failures_generic(nvml_error):
    """Regression (taiberium, #1369): `initialization error` and `nvml error` alone matched every
    NVML start failure, so a stopped persistence daemon or a missing `libnvidia-ml.so.1` was
    logged as `NVIDIA_RUNTIME_MISMATCH` and the doc sent the provider to reboot. Only the
    phrase NVML prints for the version disagreement is the mismatch."""
    prefix = "500 Server Error: Internal Server Error (\"failed to create task for container: "
    assert classify_runtime_probe_error(prefix + nvml_error + "\")") == RUNTIME_PROBE_FAILED


def test_probe_nvidia_runtime_requests_every_gpu_and_removes_the_probe_container():
    """The probe asks the daemon for all GPUs with entrypoint `true` from the runner's own image
    (no pull), labels the container so a leftover is found again, reads exit 0 as ready, and
    removes the container."""
    client = _client_with(_fake_container())

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is True and probe.reason_code is None
    kwargs = client.containers.create.call_args.kwargs
    assert client.containers.create.call_args.args[0] == "sha256:oldimageid"
    assert kwargs["entrypoint"] == ["true"]
    assert kwargs["labels"] == {RUNTIME_PROBE_LABEL_KEY: "nvidia-runtime"}
    assert kwargs["device_requests"][0]["Count"] == -1
    assert kwargs["device_requests"][0]["Capabilities"] == [["gpu"]]
    client.containers.create.return_value.remove.assert_called_once_with(force=True)


def test_probe_nvidia_runtime_reads_a_failed_start_as_the_mismatch_and_still_removes_the_container():
    """Regression: on the ticket-0325 host `docker run --gpus all` fails in the NVIDIA prestart
    hook. The probe reports the mismatch code with the daemon's text and cleans up."""
    client = _client_with(_fake_container())
    client.containers.create.return_value.start.side_effect = docker.errors.APIError(NVML_MISMATCH)

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is False
    assert probe.reason_code == RUNTIME_PROBE_NVIDIA_MISMATCH
    assert "driver/library version mismatch" in probe.error
    client.containers.create.return_value.remove.assert_called_once_with(force=True)


def test_probe_nvidia_runtime_reads_a_nonzero_exit_as_not_ready():
    client = _client_with(_fake_container())
    client.containers.create.return_value.wait.return_value = {"StatusCode": 125}

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is False and probe.reason_code == RUNTIME_PROBE_FAILED


def test_probe_nvidia_runtime_reads_a_hung_probe_as_not_ready_and_removes_it():
    """A container that never exits (a wedged hook) hits `wait`'s timeout: not ready, generic
    code, and the container is still force-removed."""
    client = _client_with(_fake_container())
    client.containers.create.return_value.wait.side_effect = requests.exceptions.ReadTimeout("wait timed out")

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is False and probe.reason_code == RUNTIME_PROBE_FAILED
    assert "timed out" in probe.error
    client.containers.create.return_value.remove.assert_called_once_with(force=True)


def _stale_probe(short_id):
    leftover = Mock()
    leftover.short_id, leftover.status = short_id, "exited"
    return leftover


def test_probe_nvidia_runtime_removes_labelled_stale_probes_before_creating_one():
    """Regression (taiberium, #1369): a probe whose `remove` failed stayed on the host, and every
    held cycle created another one beside it. The probe lists the containers that carry its
    label, every state included, removes each with force, and only then creates its own."""
    first, second = _stale_probe("stale1"), _stale_probe("stale2")
    client = _client_with(_fake_container(), stale_probes=[first, second])

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is True
    assert client.containers.list.call_args_list[0].kwargs == {"all": True, "filters": {"label": RUNTIME_PROBE_LABEL}}
    first.remove.assert_called_once_with(force=True)
    second.remove.assert_called_once_with(force=True)
    client.containers.create.assert_called_once()


def test_probe_nvidia_runtime_aborts_when_a_stale_probe_cannot_be_removed():
    """A leftover the daemon will not remove is a failed probe with no new container: creating
    one more on top of it is what the cleanup exists to stop. The hold names the leftover."""
    stuck = _stale_probe("stuck1")
    stuck.remove.side_effect = docker.errors.APIError("removal of container stuck1 is already in progress")
    client = _client_with(_fake_container(), stale_probes=[stuck])

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is False and probe.reason_code == RUNTIME_PROBE_FAILED
    assert "stale runtime probe container not removed" in probe.error
    assert "stuck1" in probe.error
    client.containers.create.assert_not_called()


def test_probe_nvidia_runtime_aborts_when_the_stale_probe_listing_fails():
    """Cannot list means cannot know what is left over: no new container, a failed probe."""
    client = _client_with(_fake_container())
    client.containers.list.side_effect = docker.errors.APIError("daemon busy")

    probe = probe_nvidia_runtime(client, "sha256:oldimageid")

    assert probe.ok is False and probe.reason_code == RUNTIME_PROBE_FAILED
    client.containers.create.assert_not_called()


@patch('watchtower.logger')
@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_removes_the_probe_container_a_held_cycle_left_behind(
    mock_settings, mock_docker, mock_fetch, mock_pull, mock_logger
):
    """Regression (taiberium, #1369): two held cycles on a host whose daemon refused the first
    probe's `remove`. Before, the second cycle created a second probe container beside the
    first and the host gained one per cycle. Now the second cycle finds the first by its label,
    removes it, and only then runs its own probe: one container on the host at any time."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    mock_settings.WATCHTOWER_INTERVAL = 300
    client = _client_with(_fake_container())
    probe_container = client.containers.create.return_value
    probe_container.short_id, probe_container.status = "probe1", "created"
    probe_container.start.side_effect = docker.errors.APIError(NVML_MISMATCH)
    probe_container.remove.side_effect = [docker.errors.APIError("device or resource busy"), None, None]
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST

    check_and_update()
    assert client.containers.create.call_count == 1
    assert probe_container.remove.call_count == 1

    client.stale_probes = [probe_container]  # the daemon now lists the leftover under the label
    check_and_update()

    assert client.containers.create.call_count == 2
    assert probe_container.remove.call_count == 3  # the stale removal, then the second probe's own
    mock_pull.assert_not_called()
    held = [str(c.args[0]) for c in mock_logger.warning.call_args_list if "Update held" in str(c.args[0])]
    assert len(held) == 2
    stale = [str(c.args[0]) for c in mock_logger.warning.call_args_list if "stale runtime probe" in str(c.args[0])]
    assert len(stale) == 1 and "probe1" in stale[0]


@patch('watchtower.logger')
@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_holds_an_unrelated_nvml_error_without_the_mismatch_remedy(
    mock_settings, mock_docker, mock_fetch, mock_pull, mock_logger
):
    """Regression (taiberium, #1369): a host whose NVML fails for another reason (persistence
    daemon down) is held, but the log says `RUNTIME_PROBE_FAILED` with the daemon's text, not
    `NVIDIA_RUNTIME_MISMATCH`, so the doc does not send the provider to reboot."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    mock_settings.WATCHTOWER_INTERVAL = 300
    client = _client_with(_fake_container())
    client.containers.create.return_value.start.side_effect = docker.errors.APIError(
        "500 Server Error: Internal Server Error (\"failed to create task for container: "
        "nvidia-container-cli: initialization error: nvml error: unknown error: unknown\")"
    )
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST

    check_and_update()

    mock_pull.assert_not_called()
    held = [str(c.args[0]) for c in mock_logger.warning.call_args_list if "Update held" in str(c.args[0])]
    assert len(held) == 1
    assert RUNTIME_PROBE_FAILED in held[0]
    assert RUNTIME_PROBE_NVIDIA_MISMATCH not in held[0]
    assert "nvml error: unknown error" in held[0]


@patch('watchtower.logger')
@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_holds_when_the_runner_has_no_image_id(
    mock_settings, mock_docker, mock_fetch, mock_pull, mock_logger
):
    """No image id to probe with is a hold, not a crash into the outer handler."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    mock_settings.WATCHTOWER_INTERVAL = 300
    runner = _fake_container()
    runner.attrs["Image"] = None
    client = _client_with(runner)
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST

    check_and_update()

    mock_pull.assert_not_called()
    client.containers.create.assert_not_called()
    held = [str(c.args[0]) for c in mock_logger.warning.call_args_list if "Update held" in str(c.args[0])]
    assert len(held) == 1 and RUNTIME_PROBE_FAILED in held[0]


@patch('watchtower.logger')
@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_holds_the_update_while_the_nvidia_runtime_probe_fails(
    mock_settings, mock_docker, mock_fetch, mock_pull, mock_logger
):
    """Regression (ticket-0325): a new signed digest on a host whose NVIDIA runtime is broken
    pulled and recreated the runner; the new executor never started. Now nothing is pulled
    and the log says why, with the reason code."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    mock_settings.WATCHTOWER_INTERVAL = 300
    client = _client_with(_fake_container())
    client.containers.create.return_value.start.side_effect = docker.errors.APIError(NVML_MISMATCH)
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST

    check_and_update()

    mock_pull.assert_not_called()
    held = [str(c.args[0]) for c in mock_logger.warning.call_args_list if "Update held" in str(c.args[0])]
    assert len(held) == 1
    assert RUNTIME_PROBE_NVIDIA_MISMATCH in held[0]
    assert "driver/library version mismatch" in held[0]


@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_resumes_once_the_nvidia_runtime_probe_passes(
    mock_settings, mock_docker, mock_fetch, mock_pull
):
    """Two cycles on the same host: the first holds (hook fails), the second, after the
    reboot, pulls with the same signed digest. No state is kept between cycles."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    mock_settings.WATCHTOWER_INTERVAL = 300
    client = _client_with(_fake_container())
    probe_container = client.containers.create.return_value
    probe_container.start.side_effect = [docker.errors.APIError(NVML_MISMATCH), None]
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST
    mock_pull.return_value = True

    check_and_update()
    mock_pull.assert_not_called()

    check_and_update()
    mock_pull.assert_called_once_with(client, IMAGE, NEW_DIGEST)


@patch('watchtower.probe_nvidia_runtime')
@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_skips_the_probe_when_there_is_no_runner_yet(
    mock_settings, mock_docker, mock_fetch, mock_pull, mock_probe
):
    """A first CVM boot has no runner image on the host to probe with and nothing running
    to break; the create path is unchanged."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    client = _client_with(None)
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST
    mock_pull.return_value = True

    check_and_update()

    mock_probe.assert_not_called()
    mock_pull.assert_called_once_with(client, IMAGE, NEW_DIGEST)


@patch('watchtower.probe_nvidia_runtime')
@patch('watchtower.pull_and_restart_containers')
@patch('watchtower.fetch_verified_digest')
@patch('watchtower.docker.from_env')
@patch('watchtower.settings')
def test_check_and_update_skips_the_probe_for_a_runner_that_was_only_created(
    mock_settings, mock_docker, mock_fetch, mock_pull, mock_probe
):
    """The probe would run the created runner's unverified image; it goes straight to the
    signed pull instead."""
    mock_settings.WATCHTOWER_IMAGE = IMAGE
    client = _client_with(_fake_container(status="created"))
    mock_docker.return_value = client
    mock_fetch.return_value = NEW_DIGEST
    mock_pull.return_value = True

    check_and_update()

    mock_probe.assert_not_called()
    mock_pull.assert_called_once_with(client, IMAGE, NEW_DIGEST)


# ── the trusted validator: named by a message the subnet owner signed, real signatures ───────────────────────


import json

import validator_trust_service as trust_service
from bittensor_wallet import Keypair
from datura.validator_trust import SignedTrustMessage, trust_line
from validator_trust_service import ValidatorTrust

OWNER = Keypair.create_from_uri("//WatchtowerTrustOwner")
NEW_OWNER = Keypair.create_from_uri("//WatchtowerTrustNewOwner")
OLD_VALIDATOR = Keypair.create_from_uri("//WatchtowerTrustOldValidator")
NEW_VALIDATOR = Keypair.create_from_uri("//WatchtowerTrustNewValidator")


def _signed_digest(keypair, digest="sha256:rotation", ts=None):
    ts = int(time.time()) if ts is None else ts
    return WatchtowerDigestResponse(
        digest=digest, timestamp=ts, signature="0x" + keypair.sign(f"{digest}:{ts}").hex()
    )


def _message(owner: Keypair, validator: Keypair, revision: int) -> SignedTrustMessage:
    line = trust_line(validator.ss58_address, revision)
    return SignedTrustMessage(message=line, signature="0x" + owner.sign(line).hex())


def _image(tmp_path, signed: SignedTrustMessage):
    image = tmp_path / f"image-{signed.message.rsplit(':', 1)[1]}.json"
    image.write_text(json.dumps(signed._asdict()))
    return image


def _start(tmp_path, image=None) -> ValidatorTrust:
    """A watchtower process starting from the image copy `image` (none when absent)."""
    trust = ValidatorTrust(image_message=image or tmp_path / "absent.json")
    trust.load()
    return trust


def _down(*args, **kwargs):
    raise requests.ConnectionError("down")


def _backend(monkeypatch, answer) -> None:
    """The message route answers `answer(url)` (a dict, None for a 404, or it raises); chain reads go to `_chain`."""
    monkeypatch.setattr(trust_service, "fetch_json", lambda url, body=None: answer(url))


def _chain(monkeypatch, read) -> None:
    monkeypatch.setattr(trust_service, "read_subnet_owner_coldkey", read)


def _accepts(monkeypatch, trust: ValidatorTrust, keypair: Keypair) -> bool:
    """Whether a digest signed by `keypair` passes the watchtower's check while it holds `trust`."""
    import watchtower

    monkeypatch.setattr(watchtower, "validator_trust", trust)
    try:
        verify_watchtower_signature(_signed_digest(keypair))
        return True
    except Exception as e:
        assert "Invalid signature" in str(e)
        return False


@pytest.fixture(autouse=True)
def _release_owner(monkeypatch):
    monkeypatch.setattr(trust_service, "RELEASE_SUBNET_OWNER_COLDKEY", OWNER.ss58_address)
    monkeypatch.setattr("watchtower.WATCHTOWER_VALIDATOR_HOTKEY", OLD_VALIDATOR.ss58_address)


def test_acceptance_validator_change_holds_with_backend_and_chain_down(tmp_path, monkeypatch):
    running = _start(tmp_path, _image(tmp_path, _message(OWNER, OLD_VALIDATOR, 1)))
    assert _accepts(monkeypatch, running, OLD_VALIDATOR)

    # the backend serves the owner's message naming the new validator
    _chain(monkeypatch, lambda url: OWNER.ss58_address)
    _backend(monkeypatch, lambda url: _message(OWNER, NEW_VALIDATOR, 2)._asdict())
    running.refresh()
    assert _accepts(monkeypatch, running, NEW_VALIDATOR)
    assert not _accepts(monkeypatch, running, OLD_VALIDATOR)

    # backend and chain both go down: the new validator stays
    _backend(monkeypatch, _down)
    _chain(monkeypatch, _down)
    running.refresh()
    assert running.report() == {
        "validator": NEW_VALIDATOR.ss58_address, "revision": 2, "subnet_owner": OWNER.ss58_address
    }
    assert _accepts(monkeypatch, running, NEW_VALIDATOR)
    assert not _accepts(monkeypatch, running, OLD_VALIDATOR)

    # restarted on a release that carries the new message, backend and chain still down
    restarted = _start(tmp_path, _image(tmp_path, _message(OWNER, NEW_VALIDATOR, 2)))
    restarted.refresh()
    assert _accepts(monkeypatch, restarted, NEW_VALIDATOR)
    assert not _accepts(monkeypatch, restarted, OLD_VALIDATOR)


def test_a_restart_on_an_older_image_takes_the_new_validator_back_at_the_first_message_read(tmp_path, monkeypatch):
    # no volume: the taken message lives in memory, so a restart starts from the image copy again
    restarted = _start(tmp_path, _image(tmp_path, _message(OWNER, OLD_VALIDATOR, 1)))
    assert _accepts(monkeypatch, restarted, OLD_VALIDATOR)
    _chain(monkeypatch, _down)
    _backend(monkeypatch, lambda url: _message(OWNER, NEW_VALIDATOR, 2)._asdict())
    restarted.refresh()
    assert _accepts(monkeypatch, restarted, NEW_VALIDATOR)
    assert not _accepts(monkeypatch, restarted, OLD_VALIDATOR)


def test_higher_revision_wins_and_a_lower_or_equal_one_is_refused(tmp_path):
    trust = _start(tmp_path, _image(tmp_path, _message(OWNER, OLD_VALIDATOR, 5)))
    assert not trust.offer(_message(OWNER, NEW_VALIDATOR, 4))
    assert not trust.offer(_message(OWNER, NEW_VALIDATOR, 5))
    assert trust.report()["validator"] == OLD_VALIDATOR.ss58_address
    assert trust.offer(_message(OWNER, NEW_VALIDATOR, 6))
    assert trust.report()["validator"] == NEW_VALIDATOR.ss58_address
    # a rollback needs a higher revision too
    assert not trust.offer(_message(OWNER, OLD_VALIDATOR, 5))
    assert trust.offer(_message(OWNER, OLD_VALIDATOR, 7))
    assert trust.report() == {
        "validator": OLD_VALIDATOR.ss58_address, "revision": 7, "subnet_owner": OWNER.ss58_address
    }


@pytest.mark.parametrize(
    "answer",
    [
        _down,
        lambda url: None,  # 404
        lambda url: {"message": "garbage"},
        lambda url: {"message": trust_line(NEW_VALIDATOR.ss58_address, 9), "signature": "0x" + "00" * 64},
        lambda url: _message(NEW_VALIDATOR, NEW_VALIDATOR, 9)._asdict(),  # not the owner's signature
    ],
)
def test_the_last_good_message_is_kept_on_any_error(tmp_path, monkeypatch, answer):
    trust = _start(tmp_path, _image(tmp_path, _message(OWNER, OLD_VALIDATOR, 1)))
    _backend(monkeypatch, answer)
    trust.refresh_message()
    assert trust.report()["validator"] == OLD_VALIDATOR.ss58_address
    assert _accepts(monkeypatch, trust, OLD_VALIDATOR)
    assert not _accepts(monkeypatch, trust, NEW_VALIDATOR)


def test_the_baked_hotkey_is_trusted_only_while_no_message_is_held(tmp_path, monkeypatch):
    import watchtower

    trust = _start(tmp_path)  # no image copy
    monkeypatch.setattr(watchtower, "validator_trust", trust)
    assert watchtower.trusted_validator_hotkeys() == (OLD_VALIDATOR.ss58_address,)
    assert trust.offer(_message(OWNER, NEW_VALIDATOR, 1))
    assert watchtower.trusted_validator_hotkeys() == (NEW_VALIDATOR.ss58_address,)
    assert not _accepts(monkeypatch, trust, OLD_VALIDATOR)


def test_an_image_copy_not_signed_by_the_release_owner_is_ignored(tmp_path, monkeypatch):
    trust = _start(tmp_path, _image(tmp_path, _message(NEW_OWNER, NEW_VALIDATOR, 1)))
    assert trust.report()["validator"] is None
    assert _accepts(monkeypatch, trust, OLD_VALIDATOR)
    assert not _accepts(monkeypatch, trust, NEW_VALIDATOR)


def test_an_owner_change_needs_every_chain_node_and_then_verifies_new_messages(tmp_path, monkeypatch):
    trust = _start(tmp_path, _image(tmp_path, _message(OWNER, OLD_VALIDATOR, 1)))
    first_node = trust_service.PUBLIC_CHAIN_NODES[0]
    _chain(monkeypatch, lambda url: NEW_OWNER.ss58_address if url == first_node else OWNER.ss58_address)
    assert not trust.refresh_owner()
    assert trust.owner == OWNER.ss58_address
    _chain(monkeypatch, lambda url: NEW_OWNER.ss58_address)
    assert trust.refresh_owner()
    # the held validator stays until the new owner signs a higher revision; the old owner is no longer heard
    assert trust.report()["validator"] == OLD_VALIDATOR.ss58_address
    assert not trust.offer(_message(OWNER, NEW_VALIDATOR, 2))
    assert trust.offer(_message(NEW_OWNER, NEW_VALIDATOR, 2))


def test_a_non_prod_build_trusts_its_own_hotkey_and_reads_no_message(tmp_path, monkeypatch):
    import watchtower

    trust = ValidatorTrust(image_message=_image(tmp_path, _message(OWNER, NEW_VALIDATOR, 1)), enabled=False)
    trust.load()
    _backend(monkeypatch, lambda url: pytest.fail("a non-prod build read the message"))
    trust.refresh()
    assert not trust.offer(_message(OWNER, NEW_VALIDATOR, 2))
    monkeypatch.setattr(watchtower, "validator_trust", trust)
    assert watchtower.trusted_validator_hotkeys() == (OLD_VALIDATOR.ss58_address,)


def test_the_shipped_message_is_signed_by_the_release_owner_and_names_the_baked_hotkey():
    from datura.validator_trust import RELEASE_SUBNET_OWNER_COLDKEY, verify_trust_message

    import config

    data = json.loads(trust_service.IMAGE_MESSAGE_PATH.read_text())
    trusted = verify_trust_message(SignedTrustMessage(data["message"], data["signature"]), RELEASE_SUBNET_OWNER_COLDKEY)
    assert trusted.validator == config.WATCHTOWER_VALIDATOR_HOTKEY


def test_the_message_route_sits_next_to_the_digest_endpoint():
    assert trust_service.VALIDATOR_MESSAGE_URL == "https://lium.io/api/watchtower/validator-message"


def _serve(handler_body):
    """A real HTTP server on localhost whose GET runs `handler_body(handler)`; returns (url, stop)."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            handler_body(self)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/watchtower/validator-message", server.shutdown


def test_fetch_json_gives_up_on_a_reply_that_arrives_in_short_reads_within_the_total_bound(monkeypatch):
    """review lium-io#2384: a server sending a byte every 0.3 s never trips a socket read timeout; the whole request
    is bounded instead (aiohttp total timeout), so a slow chain node cannot hold the digest check"""
    import time as _time

    def drip(handler):
        handler.send_response(200)
        handler.send_header("Content-Length", "256")
        handler.end_headers()
        for _ in range(256):
            try:
                handler.wfile.write(b" ")
                handler.wfile.flush()
            except OSError:
                return
            _time.sleep(0.3)

    url, stop = _serve(drip)
    monkeypatch.setattr(trust_service, "FETCH_TIMEOUT_SECONDS", 1.0)
    try:
        started = _time.monotonic()
        with pytest.raises(TimeoutError):
            trust_service.fetch_json(url)
        assert _time.monotonic() - started < 3
    finally:
        stop()


def test_fetch_json_reads_a_reply_and_refuses_one_over_the_cap():
    def reply(body):
        def handle(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        return handle

    url, stop = _serve(reply(b'{"message": "m", "signature": "s"}'))
    try:
        assert trust_service.fetch_json(url) == {"message": "m", "signature": "s"}
    finally:
        stop()
    url, stop = _serve(reply(b" " * (trust_service.FETCH_LIMIT_BYTES + 8192)))
    try:
        with pytest.raises(ValueError, match="over"):
            trust_service.fetch_json(url)
    finally:
        stop()

def test_trust_a_staging_build_follows_only_its_own_owner_key_and_never_reads_the_chain(tmp_path, monkeypatch):
    staging = Keypair.create_from_uri("//WatchtowerTrustStagingOwner")
    _chain(monkeypatch, Mock(side_effect=AssertionError("chain read")))
    trust = ValidatorTrust(image_message=_image(tmp_path, _message(OWNER, NEW_VALIDATOR, 8)),
                           owner=staging.ss58_address, read_chain=False, message=_message(staging, OLD_VALIDATOR, 1))
    trust.load()
    assert trust.report()["validator"] == OLD_VALIDATOR.ss58_address
    assert trust.offer(_message(OWNER, NEW_VALIDATOR, 2)) is False
    assert trust.refresh_owner() is False and trust.owner == staging.ss58_address
    assert trust.offer(_message(staging, NEW_VALIDATOR, 2)) is True


def test_a_disabled_watchtower_stays_up_idle_and_never_checks():
    """WATCHTOWER_ENABLED=false: main() must not return (under unless-stopped an exit is restarted every second)."""
    import watchtower as wt

    class Stop(Exception):
        pass

    with patch.object(wt.settings, "WATCHTOWER_ENABLED", False), \
            patch.object(wt.time, "sleep", side_effect=[None, Stop()]) as sleep, \
            patch.object(wt, "check_and_update") as check, patch.object(wt.validator_trust, "load") as load:
        with pytest.raises(Stop):
            wt.main()
    assert sleep.call_count == 2
    check.assert_not_called()
    load.assert_not_called()

