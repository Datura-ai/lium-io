"""DAH-6531: the delete for the node a moved pod left runs only on the host that node booted as. Its
registration can point at the replacement's host by the time the delete is sent; the boot id read on the
connection the delete uses tells the two apart."""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import pytest
from payload_models.payloads import (
    BaseServerRequest,
    ContainerDeleted,
    ContainerDeleteOnNodeLeftRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
)

from tests.test_delete_drops_dirty_volume import (
    _delete,
    _delete_payload,
    _host_answers,
    _wire_delete,
)
from tests.test_docker_service import (  # noqa: F401
    _make_ssh_command_result,
    docker_service,
    mock_dependencies,
)

BOOT_ID_CMD = "cat /proc/sys/kernel/random/boot_id"
NODE_LEFT_BOOT = "11111111-2222-4333-8444-555555555555"


def _payload() -> ContainerDeleteOnNodeLeftRequest:
    return ContainerDeleteOnNodeLeftRequest(
        **_delete_payload().model_dump(exclude={"message_type"}), expected_boot_id=NODE_LEFT_BOOT
    )


def _host(boot: str | None):
    host = _host_answers()

    async def answer(command: str):
        if command == BOOT_ID_CMD:
            return _make_ssh_command_result(
                exit_status=0 if boot else 1, stdout=f"{boot}\n" if boot else ""
            )
        return await host(command)

    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize("boot", ["99999999-8888-4777-8666-555555555555", None])
async def test_a_host_with_another_or_no_boot_id_is_never_touched(
    docker_service, monkeypatch, boot
):
    order = _wire_delete(docker_service, monkeypatch, _host(boot))

    result = await _delete(docker_service, _payload())

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code is FailedContainerErrorCodes.NotTheNodeLeft
    assert order == [("ssh", BOOT_ID_CMD)]


@pytest.mark.asyncio
async def test_the_node_left_deletes_as_any_delete_does(docker_service, monkeypatch):
    order = _wire_delete(docker_service, monkeypatch, _host(NODE_LEFT_BOOT))
    payload = _payload()

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    assert order[0] == ("ssh", BOOT_ID_CMD)
    assert ("remove", payload.container_name) in order


def test_the_backend_message_parses_to_the_checked_delete():
    raw = _payload().model_dump_json()

    assert type(BaseServerRequest.parse(raw)) is ContainerDeleteOnNodeLeftRequest
