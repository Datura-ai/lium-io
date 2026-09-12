from unittest.mock import Mock

import docker
import pytest

from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.errors import ApiFailure


class FakeAPIError(Exception):
    pass


@pytest.fixture(autouse=True)
def real_docker_api_error(monkeypatch):
    # conftest replaces the docker module with a MagicMock; the except-clauses in
    # docker_ops need a real exception class to match against
    monkeypatch.setattr(docker.errors, "APIError", FakeAPIError, raising=False)


def _ops() -> DockerOps:
    ops = DockerOps(VastSettings())
    ops._client = Mock()
    return ops


# --- exec_in_uns fail-closed ---


def test_exec_in_uns_stopped_container_is_api_failure():
    # a stopped vast-uns must refuse in the error envelope, not leak a raw
    # docker APIError as a bare 500
    ops = _ops()
    ops._client.containers.get.return_value = Mock(status="exited")

    with pytest.raises(ApiFailure) as exc_info:
        ops.exec_in_uns(["docker", "ps"])

    assert exc_info.value.code == "host_command_failed"
    assert "status=exited" in exc_info.value.detail


def test_exec_in_uns_docker_api_error_is_api_failure():
    ops = _ops()
    container = Mock(status="running")
    container.exec_run.side_effect = FakeAPIError("500 Server Error: exec failed")
    ops._client.containers.get.return_value = container

    with pytest.raises(ApiFailure) as exc_info:
        ops.exec_in_uns(["docker", "ps"])

    assert exc_info.value.code == "host_command_failed"


# --- vast_rental_containers (contract-safety rail) ---


def test_vast_rental_containers_counts_any_state_and_filters_prefix():
    ops = _ops()
    ops.exec_in_uns = Mock(return_value=(0, "C.123\nother\n"))

    result = ops.vast_rental_containers()

    cmd = ops.exec_in_uns.call_args.args[0]
    assert "ps" in cmd and "-a" in cmd  # stopped contracts still hold the contract
    assert result == ["C.123"]


def test_vast_rental_containers_nested_ps_failure_raises():
    ops = _ops()
    ops.exec_in_uns = Mock(return_value=(1, "Cannot connect to the Docker daemon"))

    with pytest.raises(ApiFailure) as exc_info:
        ops.vast_rental_containers()

    assert exc_info.value.code == "host_command_failed"


# --- executor_network: no silent wrong pick ---


def _network(name: str) -> Mock:
    network = Mock()
    network.name = name
    return network


def test_executor_network_primary_read_failure_raises():
    ops = _ops()
    ops._client.containers.get.side_effect = RuntimeError("docker down")

    with pytest.raises(ApiFailure) as exc_info:
        ops.executor_network()

    assert exc_info.value.code == "host_command_failed"
    ops._client.networks.list.assert_not_called()


def test_executor_network_prefers_own_network():
    ops = _ops()
    ops._client.containers.get.return_value = Mock(
        attrs={"NetworkSettings": {"Networks": {"executor_default": {}}}}
    )

    assert ops.executor_network() == "executor_default"
    ops._client.networks.list.assert_not_called()


def test_executor_network_fallback_single_match():
    ops = _ops()
    ops._client.containers.get.return_value = Mock(attrs={"NetworkSettings": {"Networks": {}}})
    ops._client.networks.list.return_value = [_network("bridge"), _network("executor_default")]

    assert ops.executor_network() == "executor_default"


def test_executor_network_fallback_ambiguous_raises():
    ops = _ops()
    ops._client.containers.get.return_value = Mock(attrs={"NetworkSettings": {"Networks": {}}})
    ops._client.networks.list.return_value = [_network("executor_a"), _network("executor_b")]

    with pytest.raises(ApiFailure) as exc_info:
        ops.executor_network()

    assert exc_info.value.code == "host_command_failed"


def test_executor_network_fallback_no_match_is_none():
    ops = _ops()
    ops._client.containers.get.return_value = Mock(attrs={"NetworkSettings": {"Networks": {}}})
    ops._client.networks.list.return_value = [_network("bridge")]

    assert ops.executor_network() is None


# --- run_vast_uns restart policy ---


def test_run_vast_uns_comes_back_after_reboot():
    ops = _ops()

    ops.run_vast_uns("executor_default")

    kwargs = ops._client.containers.run.call_args.kwargs
    assert kwargs["restart_policy"] == {"Name": "unless-stopped"}


# --- nested daemon.json immutability check ---


def test_nested_daemon_immutable_true_when_i_flag_set():
    ops = _ops()
    ops.exec_in_uns = Mock(return_value=(0, "----i---------e------- /etc/docker/daemon.json"))

    assert ops.nested_daemon_immutable() is True


def test_nested_daemon_immutable_false_without_i_flag():
    ops = _ops()
    ops.exec_in_uns = Mock(return_value=(0, "--------------e------- /etc/docker/daemon.json"))

    assert ops.nested_daemon_immutable() is False


def test_nested_daemon_immutable_false_on_lsattr_failure():
    ops = _ops()
    ops.exec_in_uns = Mock(return_value=(1, "lsattr: No such file or directory"))

    assert ops.nested_daemon_immutable() is False
