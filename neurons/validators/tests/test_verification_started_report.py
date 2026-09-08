"""DAH-3019: the validator tells the backend when a miner's executors start their pipelines — one
signed request per miner per cycle, off the hot path, never failing the run."""

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from clients.backend_client import VERIFICATION_STARTED_BATCH_MAX, BackendClient
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo, RequestType
from payload_models.payloads import MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import CYCLE_DONE, EXPRESS_LANE, MinerService
from services.task.service import TaskService

from core.config import settings

STARTED_AT = datetime(2026, 9, 8, 3, 1, 30, tzinfo=UTC)


@pytest.fixture
def client():
    keypair = MagicMock()
    keypair.ss58_address = "5FakeValidatorHotkey"
    keypair.sign = MagicMock(return_value=b"\x00" * 64)
    return BackendClient(base_url="https://api.example.com", keypair=keypair)


def _uuids(n: int) -> list[str]:
    return [str(uuid4()) for _ in range(n)]


def _executor(uuid: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=uuid,
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python3",
        root_dir="/tmp",
    )


def _payload(miner_hotkey: str = "5Miner") -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="2026-09-08 02:53:00",
        miner_hotkey=miner_hotkey,
        miner_coldkey="5Cold",
        miner_address="127.0.0.1",
        miner_port=8091,
    )


# --------------------------------------------------------------------------- BackendClient


@pytest.mark.asyncio
async def test_report_verification_started_posts_one_batch_for_the_miner(client):
    client.post = AsyncMock(return_value=SimpleNamespace(recorded=5))
    uuids = _uuids(5)

    await client.report_verification_started(
        job_batch_id="2026-09-08 02:53:00",
        miner_hotkey="5Miner",
        executor_uuids=uuids,
        started_at=STARTED_AT,
    )

    client.post.assert_awaited_once()
    path = client.post.await_args.args[0]
    kwargs = client.post.await_args.kwargs
    assert path == "/validator/5FakeValidatorHotkey/verification-started"
    assert kwargs["json_data"] == {
        "job_batch_id": "2026-09-08 02:53:00",
        "miner_hotkey": "5Miner",
        "started_at": "2026-09-08T03:01:30+00:00",
        "executor_uuids": uuids,
    }
    assert kwargs["add_signature"] is True
    assert kwargs["timeout"] == 10
    # the route ships with lium-platform#120; a 404 from an older backend is not an outage
    assert kwargs["non_200_log_level"] == logging.WARNING


@pytest.mark.asyncio
async def test_report_verification_started_never_raises(client):
    client.post = AsyncMock(side_effect=RuntimeError("backend down"))

    # A failed report costs the provider a progress bar, never a verdict.
    await client.report_verification_started(
        job_batch_id="2026-09-08 02:53:00",
        miner_hotkey="5Miner",
        executor_uuids=_uuids(3),
        started_at=STARTED_AT,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crafted",
    [
        "../../admin/executors/x",
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301/../../../pods",
        "{3f2504e0-4f89-11d3-9a0c-0305e82c3301}",
        "urn:uuid:3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        "3f2504e04f8911d39a0c0305e82c3301",
        "",
        "not a uuid at all?x=1",
    ],
)
async def test_a_miner_controlled_uuid_that_is_not_a_uuid_is_dropped_and_the_rest_still_goes_out(
    client, crafted
):
    """The backend parses the list as UUIDs and would 422 the whole batch on one bad value."""
    client.post = AsyncMock(return_value=SimpleNamespace(recorded=2))
    good = _uuids(2)

    await client.report_verification_started(
        job_batch_id="2026-09-08 02:53:00",
        miner_hotkey="5Miner",
        executor_uuids=[good[0], crafted, good[1]],
        started_at=STARTED_AT,
    )

    client.post.assert_awaited_once()
    assert client.post.await_args.kwargs["json_data"]["executor_uuids"] == good


@pytest.mark.asyncio
async def test_a_batch_with_no_valid_uuid_sends_nothing(client):
    client.post = AsyncMock()

    await client.report_verification_started(
        job_batch_id="2026-09-08 02:53:00",
        miner_hotkey="5Miner",
        executor_uuids=["nope", ""],
        started_at=STARTED_AT,
    )

    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_miner_with_more_executors_than_the_batch_limit_gets_two_requests(client):
    client.post = AsyncMock(return_value=SimpleNamespace(recorded=1))
    uuids = _uuids(VERIFICATION_STARTED_BATCH_MAX + 1)

    await client.report_verification_started(
        job_batch_id="2026-09-08 02:53:00",
        miner_hotkey="5Miner",
        executor_uuids=uuids,
        started_at=STARTED_AT,
    )

    sent = [call.kwargs["json_data"]["executor_uuids"] for call in client.post.await_args_list]
    assert [len(chunk) for chunk in sent] == [VERIFICATION_STARTED_BATCH_MAX, 1]
    assert [u for chunk in sent for u in chunk] == uuids


@pytest.mark.asyncio
async def test_a_404_from_a_backend_without_the_route_is_a_warning_not_an_error(client, caplog):
    response = MagicMock()
    response.status = 404
    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=response)
    async_cm.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.request = MagicMock(return_value=async_cm)

    with (
        patch.object(BackendClient, "get_session", AsyncMock(return_value=session)),
        caplog.at_level(logging.WARNING),
    ):
        await client.report_verification_started(
            job_batch_id="2026-09-08 02:53:00",
            miner_hotkey="5Miner",
            executor_uuids=_uuids(3),
            started_at=STARTED_AT,
        )

    failed = [r for r in caplog.records if "HTTP POST failed" in r.getMessage()]
    assert len(failed) == 1 and failed[0].levelno == logging.WARNING


# --------------------------------------------------------------------------- TaskService


@pytest.mark.asyncio
async def test_task_service_schedules_one_report_for_the_miner_without_awaiting_it():
    backend = SimpleNamespace(report_verification_started=AsyncMock())
    service = SimpleNamespace(backend_client=backend, _start_reports=set())
    executors = [_executor(u) for u in _uuids(3)]

    TaskService.report_verification_started(service, _payload(), executors)

    # Scheduled, not run: nothing has been awaited yet when the pipelines go on.
    assert len(service._start_reports) == 1
    backend.report_verification_started.assert_not_awaited()

    await asyncio.gather(*service._start_reports)

    backend.report_verification_started.assert_awaited_once()
    kwargs = backend.report_verification_started.await_args.kwargs
    assert kwargs["job_batch_id"] == "2026-09-08 02:53:00"
    assert kwargs["miner_hotkey"] == "5Miner"
    assert kwargs["executor_uuids"] == [e.uuid for e in executors]
    assert kwargs["started_at"].tzinfo is UTC
    assert service._start_reports == set()


async def _requests_for_cycle(client: BackendClient, executors_per_miner: list[int]) -> list[dict]:
    """Drive the real TaskService + BackendClient path for one cycle of fake miners; return the
    JSON bodies that went to the backend."""
    client.post = AsyncMock(return_value=SimpleNamespace(recorded=0))
    service = SimpleNamespace(backend_client=client, _start_reports=set())
    for i, count in enumerate(executors_per_miner):
        TaskService.report_verification_started(
            service, _payload(f"5Miner{i}"), [_executor(u) for u in _uuids(count)]
        )
    await asyncio.gather(*service._start_reports)
    return [call.kwargs["json_data"] for call in client.post.await_args_list]


@pytest.mark.asyncio
async def test_requests_per_cycle_equal_the_miner_count_not_the_executor_count(client):
    bodies = await _requests_for_cycle(client, [5, 5, 5])

    assert len(bodies) == 3
    assert sum(len(b["executor_uuids"]) for b in bodies) == 15
    assert sorted(b["miner_hotkey"] for b in bodies) == ["5Miner0", "5Miner1", "5Miner2"]


@pytest.mark.asyncio
async def test_the_8_sep_prod_cycle_is_104_requests_not_483(client):
    """arhangel66's numbers from validator 5F7X…b13p, cycle 2026-09-08 02:53:00: 483 executors from
    104 miners, two miners with 148 between them. Per executor that was 483 requests in 17 s; per
    miner it is 104, and the two large miners are 2 requests instead of 148."""
    executors_per_miner = [74, 74] + [4] * 29 + [3] * 73
    assert len(executors_per_miner) == 104 and sum(executors_per_miner) == 483

    bodies = await _requests_for_cycle(client, executors_per_miner)

    assert len(bodies) == 104
    assert sum(len(b["executor_uuids"]) for b in bodies) == 483  # what the per-executor shape sent
    assert sorted(len(b["executor_uuids"]) for b in bodies)[-2:] == [74, 74]


# --------------------------------------------------------------------------- MinerService


class _FakeMinerClient:
    def __init__(self, accepted: AcceptSSHKeyRequest):
        future = asyncio.get_running_loop().create_future()
        future.set_result(accepted)
        self.job_state = SimpleNamespace(miner_accepted_ssh_key_or_failed_future=future)
        self.send_model = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _miner_service_recording(order: list[str]) -> MinerService:
    """A MinerService whose task_service records the order of the start report and each create_task."""
    service = MinerService.__new__(MinerService)
    service.ssh_service = MagicMock()
    service.ssh_service.generate_ssh_key.return_value = (b"private-key", b"public-key")
    service.redis_service = MagicMock()
    service.attestation_service = MagicMock()
    service.attestation_service.maybe_issue_nonce = AsyncMock(return_value=None)
    service.in_flight = {}

    def report(miner_info, executors):
        order.append(f"report:{miner_info.miner_hotkey}:{','.join(e.uuid for e in executors)}")

    async def create_task(**kwargs):
        order.append(f"create_task:{kwargs['executor_info'].uuid}")
        return SimpleNamespace(executor_info=kwargs["executor_info"], score=1)

    service.task_service = SimpleNamespace(
        report_verification_started=report, create_task=create_task
    )
    return service


def _wallet() -> MagicMock:
    keypair = MagicMock()
    keypair.ss58_address = "5Val"
    keypair.sign.return_value = b"\x01" * 64
    wallet = MagicMock()
    wallet.get_hotkey.return_value = keypair
    return wallet


def _accepted(*uuids: str) -> AcceptSSHKeyRequest:
    return AcceptSSHKeyRequest(
        message_type=RequestType.AcceptSSHKeyRequest, executors=[_executor(u) for u in uuids]
    )


@pytest.mark.asyncio
async def test_websocket_path_reports_the_miner_once_before_launching_its_executors(monkeypatch):
    order: list[str] = []
    service = _miner_service_recording(order)
    uuids = ["exec-a", "exec-b", "exec-c"]
    monkeypatch.setattr(settings, "USE_REST_API", False)
    monkeypatch.setattr(settings, "JOB_TIME_OUT", 300)
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: _wallet())

    with (
        patch(
            "services.miner_service.MinerClient", return_value=_FakeMinerClient(_accepted(*uuids))
        ),
        patch("services.miner_service.measure_and_attach", AsyncMock()),
    ):
        await service.request_job_to_miner(
            _payload(), MagicMock(), RentedExecutorsResponse(executors={}), {}
        )

    assert order[0] == "report:5Miner:exec-a,exec-b,exec-c"
    assert sorted(order[1:]) == [f"create_task:{u}" for u in uuids]


@pytest.mark.asyncio
async def test_rest_path_reports_the_miner_once_before_launching_its_executors(monkeypatch):
    order: list[str] = []
    service = _miner_service_recording(order)
    uuids = ["exec-a", "exec-b"]
    service._generate_auth_headers = MagicMock(return_value={})
    service._make_rest_request = AsyncMock(
        return_value=(200, _accepted(*uuids).model_dump(mode="json"))
    )
    service._remove_ssh_key_via_rest = AsyncMock()
    monkeypatch.setattr(settings, "USE_REST_API", True)
    monkeypatch.setattr(settings, "JOB_TIME_OUT", 300)
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: _wallet())

    with patch("services.miner_service.measure_and_attach", AsyncMock()):
        await service.request_job_to_miner(
            _payload(), MagicMock(), RentedExecutorsResponse(executors={}), {}
        )

    assert order[0] == "report:5Miner:exec-a,exec-b"
    assert sorted(order[1:]) == [f"create_task:{u}" for u in uuids]


@pytest.mark.asyncio
async def test_the_report_carries_the_claimed_list_not_what_the_express_lane_holds(monkeypatch):
    """The report names the executors this run launches. With the express lane on (DAH-2958) an
    executor the lane is verifying right now is left out of the cycle's wave, so it is left out of
    the cycle's report too — the lane's own run reports it."""
    order: list[str] = []
    service = _miner_service_recording(order)
    service.in_flight = {"exec-b": EXPRESS_LANE}
    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", True)
    monkeypatch.setattr(settings, "USE_REST_API", False)
    monkeypatch.setattr(settings, "JOB_TIME_OUT", 300)
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: _wallet())

    with (
        patch(
            "services.miner_service.MinerClient",
            return_value=_FakeMinerClient(_accepted("exec-a", "exec-b", "exec-c")),
        ),
        patch("services.miner_service.measure_and_attach", AsyncMock()),
    ):
        await service.request_job_to_miner(
            _payload(), MagicMock(), RentedExecutorsResponse(executors={}), {}
        )

    assert order[0] == "report:5Miner:exec-a,exec-c"
    assert sorted(order[1:]) == ["create_task:exec-a", "create_task:exec-c"]
    # the run completed: the wave released its two claims and left the lane's executor alone
    assert service.in_flight == {"exec-a": CYCLE_DONE, "exec-b": EXPRESS_LANE, "exec-c": CYCLE_DONE}


@pytest.mark.asyncio
async def test_an_express_lane_run_reports_its_one_executor(monkeypatch):
    """The express lane asks for one executor; the report names that one, even when the miner
    answers with more."""
    order: list[str] = []
    service = _miner_service_recording(order)
    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", True)
    monkeypatch.setattr(settings, "USE_REST_API", False)
    monkeypatch.setattr(settings, "JOB_TIME_OUT", 300)
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: _wallet())

    with (
        patch(
            "services.miner_service.MinerClient",
            return_value=_FakeMinerClient(_accepted("exec-a", "exec-b")),
        ),
        patch("services.miner_service.measure_and_attach", AsyncMock()),
    ):
        await service.request_job_to_miner(
            _payload(), MagicMock(), RentedExecutorsResponse(executors={}), {}, executor_id="exec-b"
        )

    assert order == ["report:5Miner:exec-b", "create_task:exec-b"]
    # a lane run claims nothing for the cycle, so the release leaves in_flight untouched
    assert service.in_flight == {}


@pytest.mark.asyncio
async def test_create_task_no_longer_reports_per_executor(monkeypatch):
    """Regression guard for the per-executor shape of this PR's first draft (483 requests for a
    483-executor cycle); it passes on main too, where create_task never reported. Only the
    miner-level call above remains."""
    ctx = SimpleNamespace(pipeline_id="pipe-1", verified=None)

    class _Shell:
        def __init__(self, **_):
            self.ssh_client = object()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    ran: list[str] = []

    async def _run(_ctx):
        ran.append(_ctx.pipeline_id)
        return (
            True,
            [SimpleNamespace(event="ok", model_dump=lambda: {})],
            SimpleNamespace(success=True),
        )

    factory = SimpleNamespace(
        build_context=AsyncMock(return_value=ctx),
        build_checks=MagicMock(return_value=[]),
        build_pipeline=MagicMock(return_value=SimpleNamespace(run=_run)),
    )
    attestation = SimpleNamespace(
        prepare_host_policy=AsyncMock(
            return_value=SimpleNamespace(
                known_hosts=None,
                attestation_digest=None,
                tee_type=None,
                gpu_attestation_passed=None,
                attestation_passed=False,
            )
        )
    )
    handled = SimpleNamespace(attestation_digest=None, tee_type=None, gpu_attestation_passed=None)
    service = SimpleNamespace(
        ssh_service=SimpleNamespace(decrypt_payload=lambda *_: "key"),
        attestation_service=attestation,
        pipeline_factory=factory,
        redis_service=MagicMock(),
        report_verification_started=MagicMock(),
        backend_client=SimpleNamespace(report_verification_started=AsyncMock()),
    )
    module = TaskService.create_task.__globals__
    monkeypatch.setattr(
        module["InteractiveShellService"], "__init__", _Shell.__init__, raising=False
    )
    monkeypatch.setattr(
        module["InteractiveShellService"], "__aenter__", _Shell.__aenter__, raising=False
    )
    monkeypatch.setattr(
        module["InteractiveShellService"], "__aexit__", _Shell.__aexit__, raising=False
    )
    monkeypatch.setattr(module["settings"], "DRY_RUN", False, raising=False)
    with (
        patch.object(module["ResultHandler"], "handle_result", AsyncMock(return_value=handled)),
        patch.object(module["ResultHandler"], "__init__", lambda self, *a, **k: None),
    ):
        await TaskService.create_task(
            service,
            miner_info=_payload(),
            executor_info=_executor("exec-uuid-1"),
            keypair=SimpleNamespace(ss58_address="5Val"),
            private_key="enc",
            public_key="pub",
            encrypted_files=MagicMock(),
            rented_data=MagicMock(),
            default_docker_image_digests={},
        )

    assert ran == ["pipe-1"]  # the pipeline ran; only the report is gone
    service.report_verification_started.assert_not_called()
    service.backend_client.report_verification_started.assert_not_awaited()
