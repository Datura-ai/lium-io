"""A customer's create removes every `filler_*` on the host and confirms the removal.

DAH-3706 (20 Sep 2026): 6 of 459 rented nodes carried a live `filler_*` container beside a paying
pod. The backend's filler stop is best-effort, and a stop that did not confirm left the filler on the
create's `active_container_names` — the names `clean_existing_containers` preserves. The backend no
longer lists a filler on a customer create (lium-platform, same ticket); this is the validator's half:
a CUSTOMER_RENTAL create treats every `filler_*` as stale whatever the list says (an older backend
still lists one), re-reads `docker ps -a` after the removal, and writes the typed event
`FILLER_STILL_RUNNING` for a name that survived — the create goes on, the event makes it countable.
A FILLER create keeps protecting its listed sibling bundles (DAH-2465).
DAH-3980: the customer's rm, that re-read and the unprotected volumes' rm are one bounded SSH command.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from services.docker_service import (
    FILLER_STILL_RUNNING_EVENT,
    ContainerCleanupReport,
    DockerService,
    own_sweep_removals,
)


@pytest.fixture
def svc():
    # local fixtures rather than imports, which pyflakes reads as names every parameter below
    # redefines (F811); same services test_deploy_optimizations / test_docker_service build
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


@pytest.fixture
def docker_service(svc):
    return svc


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


def _listing(stdout: str, exit_status: int = 0, stderr: str = ""):
    result = Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


@pytest.fixture(autouse=True)
def _no_sweeps_from_other_tests():
    own_sweep_removals.clear()


def _events(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event")
        == FILLER_STILL_RUNNING_EVENT
    ]


def _removal(*names_after: str, rm_exit: int = 0, ps_exit: int = 0, stderr: str = ""):
    # what the one removal command prints: the rm's status, the names left, the listing's status
    listed = "".join(f"NAME\t{name}\n" for name in names_after)
    return _listing(f"RM\t{rm_exit}\n{listed}PS\t{ps_exit}\n", stderr=stderr)


async def _clean_for_customer(docker_service, ssh_client, report=None, **over):
    kwargs = dict(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="pod_target",
        active_container_names=[],
        remove_every_filler=True,
        report=report,
    )
    return await docker_service.clean_existing_containers(**{**kwargs, **over})


@pytest.mark.asyncio
async def test_customer_create_removes_a_filler_the_backend_still_lists(
    docker_service, retry_ssh_mock
):
    ssh_client = AsyncMock()
    # before: the host listing; then the removal, whose listing shows the filler gone
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_unconfirmed\npod_sibling\n"),
            _removal("pod_target_new", "pod_sibling"),
        ]
    )

    removed = await _clean_for_customer(
        docker_service, ssh_client, active_container_names=["filler_unconfirmed", "pod_sibling"]
    )

    rm_command = ssh_client.run.await_args_list[1].args[0]
    assert "filler_unconfirmed" in rm_command
    assert "pod_sibling" not in rm_command
    assert sorted(removed) == ["filler_unconfirmed", "pod_target"]
    retry_ssh_mock.assert_not_called()


@pytest.mark.asyncio
async def test_a_filler_that_survives_the_removal_is_logged_as_filler_still_running(
    docker_service, retry_ssh_mock, caplog
):
    ssh_client = AsyncMock()
    # the listing after the rm still names the filler: dockerd said removed, the host says otherwise
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_stuck\nfiller_gone\n"),
            _removal("filler_stuck"),
        ]
    )
    report = ContainerCleanupReport()

    with caplog.at_level(logging.WARNING):
        removed = await _clean_for_customer(
            docker_service,
            ssh_client,
            report,
            active_volume_names=["volume_target", "volume_stuck", "volume_gone"],
        )

    # the create goes on (no raise) and the survivor is reported with typed fields
    assert "filler_stuck" in removed
    [event] = _events(caplog)
    # the create path's default_extra keys the executor as `executor_uuid` (create_container); the backend's
    # rent-path event carries `executor_uuid` too (its `executor_id` is the DB row id) — one Loki query joins on it
    assert event.msg.extra["executor_uuid"] == "exec-1"
    assert event.msg.extra["pod_name"] == "pod_target"
    assert event.msg.extra["container_names"] == ["filler_stuck"]
    assert ssh_client.run.await_count == 2
    assert report.removed_cleanly_without_volume_rm is False
