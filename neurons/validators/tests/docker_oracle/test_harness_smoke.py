"""Smoke test: the shared oracle harness drives DockerService.create_container
end-to-end to a real ContainerCreated with no SSH and no docker daemon."""

from unittest.mock import Mock

import pytest
from docker_oracle.harness import (
    make_create_request,
    make_docker_service,
    make_executor_info,
    patch_create_happy_path,
)
from payload_models.payloads import ContainerCreated


@pytest.mark.asyncio
async def test_harness_happy_path_reaches_container_created(monkeypatch):
    # Arrange
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request()
    executor_info = make_executor_info(payload)

    # Act
    result = await svc.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    # Assert
    # WHY: every oracle golden builds on this harness reaching the real success
    # boundary with the live SDK retry wrapper actually driving the fake client.
    assert isinstance(result, ContainerCreated), f"expected ContainerCreated, got: {result!r}"
    run_calls = [name for name, _ in harness.docker_client.calls if name == "run_container"]
    assert run_calls == ["run_container"]
