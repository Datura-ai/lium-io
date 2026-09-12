"""
Tests for watchtower.py - Docker image monitoring and update service.
"""

import time
import pytest
from unittest.mock import Mock, patch
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
    fetch_verified_digest,
    verify_watchtower_signature,
    find_container_by_name,
    pull_and_restart_containers,
    pull_image_by_digest,
    recreate_container,
    check_and_update,
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
    mock_get.assert_called_once_with("http://test-endpoint.com/digest", timeout=30)
    mock_verify.assert_called_once()


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


def _fake_container(name="executor-executor-runner-1", cmd=None, entrypoint=None):
    """The standard stack's runner: compose labels, `.env` bind, `unless-stopped`, one network."""
    container = Mock()
    container.name = name
    container.id = "0123456789abcdef" * 4
    container.short_id = container.id[:12]
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


def _client_with(container, old_image=None, pulled_image=None):
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
    client.containers.list.assert_called_once_with(
        all=True, filters={"label": "com.docker.compose.service=executor-runner"}
    )


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
