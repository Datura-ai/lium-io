"""A customer's create removes every `filler_*` on the host and confirms the removal.

rel-box-34 (20 Sep 2026): 6 of 459 rented nodes carried a live `filler_*` container beside a paying
pod. The backend's filler stop is best-effort, and a stop that did not confirm left the filler on the
create's `active_container_names` — the names `clean_existing_containers` preserves. The backend no
longer lists a filler on a customer create (lium-platform, same ticket); this is the validator's half:
a CUSTOMER_RENTAL create treats every `filler_*` as stale whatever the list says (an older backend
still lists one), re-reads `docker ps -a` after the removal, and writes the typed event
`FILLER_STILL_RUNNING` for a name that survived — the create goes on, the event makes it countable.
A FILLER create keeps protecting its listed sibling bundles (DAH-2465).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest

from payload_models.payloads import ContainerCreated, WorkloadKind
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_client

import services.docker_service as ds_module
from services.docker_service import FILLER_STILL_RUNNING_EVENT, DockerService


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


def _events(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event")
        == FILLER_STILL_RUNNING_EVENT
    ]


@pytest.mark.asyncio
async def test_customer_create_removes_a_filler_the_backend_still_lists(
    docker_service, retry_ssh_mock
):
    ssh_client = AsyncMock()
    # before: the host listing; after the removal: the confirmation listing, the filler is gone
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_unconfirmed\npod_sibling\n"),
            _listing("pod_target_new\npod_sibling\n"),
        ]
    )

    removed = await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="pod_target",
        active_container_names=["filler_unconfirmed", "pod_sibling"],
        remove_every_filler=True,
    )

    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "filler_unconfirmed" in rm_command
    assert "pod_sibling" not in rm_command
    assert sorted(removed) == ["filler_unconfirmed", "pod_target"]


@pytest.mark.asyncio
async def test_filler_create_keeps_its_listed_sibling_bundle(docker_service, retry_ssh_mock):
    # DAH-2465: bundle #2's create must not wipe bundle #1 — the default stays the protecting one
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[_listing("filler_bundle_2\nfiller_bundle_1\n"), _listing("filler_bundle_1\n")]
    )

    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="filler_bundle_2",
        active_container_names=["filler_bundle_1"],
    )

    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "filler_bundle_2" in rm_command
    assert "filler_bundle_1" not in rm_command


@pytest.mark.asyncio
async def test_a_filler_that_survives_the_removal_is_logged_as_filler_still_running(
    docker_service, retry_ssh_mock, caplog
):
    ssh_client = AsyncMock()
    # the confirmation listing still names the filler: dockerd said removed, the host says otherwise
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_stuck\nfiller_gone\n"),
            _listing("filler_stuck\n"),
        ]
    )

    with caplog.at_level(logging.WARNING):
        removed = await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="pod_target",
            active_container_names=[],
            remove_every_filler=True,
        )

    # the create goes on (no raise) and the survivor is reported with typed fields
    assert "filler_stuck" in removed
    [event] = _events(caplog)
    # the create path's default_extra keys the executor as `executor_uuid` (create_container); compute-app
    # writes the same value as `executor_id` — one Loki query joins on the value
    assert event.msg.extra["executor_uuid"] == "exec-1"
    assert event.msg.extra["pod_name"] == "pod_target"
    assert event.msg.extra["container_names"] == ["filler_stuck"]
    assert ssh_client.run.await_count == 2
    # the confirmation is bounded like the prerun probe: a wedged dockerd cannot hang the create
    confirm_call = ssh_client.run.await_args_list[1]
    assert confirm_call.kwargs["timeout"] == ds_module._PRERUN_HOST_PROBE_TIMEOUT_SECONDS
    assert confirm_call.kwargs["check"] is False


@pytest.mark.asyncio
async def test_a_confirmed_removal_writes_no_event(docker_service, retry_ssh_mock, caplog):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\nfiller_gone\n"), _listing("")])

    with caplog.at_level(logging.WARNING):
        await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="pod_target",
            active_container_names=[],
            remove_every_filler=True,
        )

    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_a_failed_confirmation_listing_does_not_fail_the_create(
    docker_service, retry_ssh_mock, caplog
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[_listing("pod_target\nfiller_gone\n"), OSError("ssh dropped")]
    )

    with caplog.at_level(logging.WARNING):
        removed = await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="pod_target",
            active_container_names=[],
            remove_every_filler=True,
        )

    assert "filler_gone" in removed
    assert _events(caplog) == []


@pytest.mark.asyncio
async def test_a_confirmation_listing_that_exits_non_zero_is_not_read_as_confirmed(
    docker_service, retry_ssh_mock, caplog
):
    ssh_client = AsyncMock()
    # `check=False`: a non-zero `docker ps -a` returns instead of raising; empty stdout must not
    # pass as "no survivor"
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_gone\n"),
            _listing("", exit_status=1, stderr="Cannot connect to the Docker daemon"),
        ]
    )

    with caplog.at_level(logging.WARNING):
        removed = await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="pod_target",
            active_container_names=[],
            remove_every_filler=True,
        )

    assert "filler_gone" in removed
    assert _events(caplog) == []
    assert any("Unable to confirm the filler removal" in str(r.msg) for r in caplog.records)


@pytest.mark.asyncio
async def test_rm_that_fails_because_the_filler_is_already_gone_does_not_fail_the_create(
    docker_service, retry_ssh_mock
):
    # the backend's delete landed between the listing and the rm: `docker rm -f` exits non-zero for
    # the vanished name; the host says it is gone, so the create goes on
    retry_ssh_mock.side_effect = [
        Exception(
            "[clean_existing_containers] exit_code 1, stderr: No such container: filler_racing"
        ),
        None,
    ]
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_racing\n"),
            _listing("pod_other\n"),  # the re-read after the failed rm
            _listing("pod_other\n"),  # the confirmation
        ]
    )

    removed = await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="pod_target",
        active_container_names=[],
        remove_every_filler=True,
    )

    assert sorted(removed) == ["filler_racing", "pod_target"]


@pytest.mark.asyncio
async def test_rm_that_fails_with_the_container_still_there_raises(docker_service, retry_ssh_mock):
    retry_ssh_mock.side_effect = Exception("[clean_existing_containers] exit_code 1, stderr: busy")
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[_listing("pod_target\nfiller_stuck\n"), _listing("filler_stuck\n")]
    )

    with pytest.raises(Exception, match="busy"):
        await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="pod_target",
            active_container_names=[],
            remove_every_filler=True,
        )


@pytest.mark.asyncio
async def test_rm_failure_on_a_filler_create_still_raises(docker_service, retry_ssh_mock):
    # the tolerant re-read is the customer create's; a FILLER create keeps the old contract
    retry_ssh_mock.side_effect = Exception("[clean_existing_containers] exit_code 1")
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("filler_bundle_2\nfiller_old\n")])

    with pytest.raises(Exception, match="exit_code 1"):
        await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="filler_bundle_2",
            active_container_names=[],
        )
    assert ssh_client.run.await_count == 1


@pytest.mark.asyncio
async def test_no_filler_on_the_host_skips_the_confirmation_listing(docker_service, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\npod_stale\n")])

    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="pod_target",
        active_container_names=[],
        remove_every_filler=True,
    )

    assert ssh_client.run.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workload_kind", "every_filler"),
    [(WorkloadKind.CUSTOMER_RENTAL, True), (WorkloadKind.FILLER, False)],
)
async def test_create_path_asks_for_every_filler_on_a_customer_create_only(
    svc, monkeypatch, workload_kind, every_filler
):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(workload_kind=workload_kind))

    assert isinstance(result, ContainerCreated)
    assert svc.clean_existing_containers.await_args.kwargs["remove_every_filler"] is every_filler
