"""DAH-3593: the restarting-container exec error names the exit code but never the workload's log.

The rest of this file (log levels of expected outcomes) was cut as low-signal in DAH-4027; this case
stays because an image must not be able to fake a driver fault in the failure text and blame the node.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import services.rental_docker_sdk as sdk_module
from services.rental_docker_sdk import (
    ContainerExecSpec,
    RentalDockerContainerRestartingError,
    RentalDockerSdkClient,
)


# ---------------------------------------------------------------------------
# 2 — the SDK names the container state first, then Docker's 409 text
# ---------------------------------------------------------------------------


class _RestartConflict(Exception):
    def __init__(self):
        super().__init__(
            "409 Client Error: Conflict (\"Container abc is restarting, wait until the container is running\")"
        )
        self.response = SimpleNamespace(status_code=409)


class _RestartingApiClient:
    def __init__(self):
        self.exec_create_calls = 0

    def inspect_container(self, name):
        return {"State": {"Status": "running", "Running": True, "Restarting": False, "ExitCode": 137}}

    def exec_create(self, *args, **kwargs):
        self.exec_create_calls += 1
        raise _RestartConflict()

    def logs(self, name, stdout=True, stderr=True, tail=1):
        # a renter image that prints a provider-shaped fault: must never reach the failure text
        return b"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver\n"


@pytest.mark.asyncio
async def test_exec_on_restarting_container_names_exit_code_and_never_the_workload_log(monkeypatch):
    monkeypatch.setattr(sdk_module, "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS", (0,))
    api = _RestartingApiClient()
    client = RentalDockerSdkClient(api)

    with pytest.raises(RentalDockerContainerRestartingError) as raised:
        await client.exec_in_container(
            ContainerExecSpec(container_name="pod_x", argv=("true",))
        )

    text = str(raised.value)
    assert text.startswith("container restarting, exit_code=137; ")
    # the backend's GPU-fault check reads this text: the workload's own log line stays out of it
    # (review, taiberium 21 Sep), so an image cannot fake a driver fault and blame the node
    assert "NVIDIA" not in text
    assert "last log line" not in text
    # the backend's IMAGE_EXITED_MARKERS read `is restarting` out of the failure detail (review, 21 Sep)
    assert "Docker SDK exec failed: 409 Client Error" in text
    assert "Container abc is restarting" in text
    assert api.exec_create_calls == 2  # the retry budget was spent before the cause was read
    assert isinstance(raised.value.__cause__, _RestartConflict)
