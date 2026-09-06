"""DAH-3019: the validator tells the backend when an executor's pipeline starts, once per run, off
the hot path and without ever failing the run."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.clients.backend_client import BackendClient
from neurons.validators.src.services.task.service import TaskService


@pytest.fixture
def client():
    keypair = MagicMock()
    keypair.ss58_address = "5FakeValidatorHotkey"
    keypair.sign = MagicMock(return_value=b"\x00" * 64)
    return BackendClient(base_url="https://api.example.com", keypair=keypair)


@pytest.mark.asyncio
async def test_report_verification_started_posts_the_run_to_the_validator_route(client):
    client.post = AsyncMock(return_value=SimpleNamespace(recorded=True))

    await client.report_verification_started(
        "exec-uuid-1", job_batch_id="2026-09-06 21:00:00", pipeline_id="pipe-1", miner_hotkey="5Miner"
    )

    client.post.assert_awaited_once()
    path = client.post.await_args.args[0]
    kwargs = client.post.await_args.kwargs
    assert path == "/validator/5FakeValidatorHotkey/executors/exec-uuid-1/verification-started"
    assert kwargs["json_data"] == {
        "job_batch_id": "2026-09-06 21:00:00",
        "pipeline_id": "pipe-1",
        "miner_hotkey": "5Miner",
    }
    assert kwargs["add_signature"] is True
    assert kwargs["timeout"] == 10


@pytest.mark.asyncio
async def test_report_verification_started_never_raises(client):
    client.post = AsyncMock(side_effect=RuntimeError("backend down"))

    # A failed report costs the provider a progress bar, never a verdict.
    await client.report_verification_started(
        "exec-uuid-1", job_batch_id="2026-09-06 21:00:00", pipeline_id="pipe-1", miner_hotkey="5Miner"
    )


@pytest.mark.asyncio
async def test_task_service_schedules_the_report_without_awaiting_it():
    backend = SimpleNamespace(report_verification_started=AsyncMock())
    service = SimpleNamespace(backend_client=backend, _start_reports=set())
    miner_info = SimpleNamespace(job_batch_id="2026-09-06 21:00:00", miner_hotkey="5Miner")
    executor_info = SimpleNamespace(uuid="exec-uuid-1")

    TaskService.report_verification_started(service, "pipe-1", miner_info, executor_info)

    # Scheduled, not run: nothing has been awaited yet when the pipeline goes on.
    assert len(service._start_reports) == 1
    backend.report_verification_started.assert_not_awaited()

    await asyncio.gather(*service._start_reports)

    backend.report_verification_started.assert_awaited_once_with(
        "exec-uuid-1", job_batch_id="2026-09-06 21:00:00", pipeline_id="pipe-1", miner_hotkey="5Miner"
    )
    assert service._start_reports == set()


@pytest.mark.asyncio
async def test_create_task_reports_once_before_the_pipeline_runs(monkeypatch):
    """The report is issued from inside create_task, after the SSH session and context exist and
    before the first check runs, so `elapsed` on the portal starts with the pipeline."""
    order: list[str] = []
    ctx = SimpleNamespace(pipeline_id="pipe-1", verified=None)

    class _Shell:
        def __init__(self, **_):
            self.ssh_client = object()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    factory = SimpleNamespace(
        build_context=AsyncMock(return_value=ctx),
        build_checks=MagicMock(return_value=[]),
        build_pipeline=MagicMock(),
    )

    async def _run(_ctx):
        order.append("pipeline.run")
        return True, [SimpleNamespace(event="ok", model_dump=lambda: {})], SimpleNamespace(success=True)

    factory.build_pipeline.return_value = SimpleNamespace(run=_run)

    attestation = SimpleNamespace(
        prepare_host_policy=AsyncMock(
            return_value=SimpleNamespace(
                known_hosts=None, attestation_digest=None, tee_type=None, gpu_attestation_passed=None, attestation_passed=False
            )
        )
    )
    handled = SimpleNamespace(attestation_digest=None, tee_type=None, gpu_attestation_passed=None)

    def _report(pipeline_id, miner_info, executor_info):
        order.append(f"report:{pipeline_id}:{executor_info.uuid}")

    service = SimpleNamespace(
        ssh_service=SimpleNamespace(decrypt_payload=lambda *_: "key"),
        attestation_service=attestation,
        pipeline_factory=factory,
        redis_service=MagicMock(),
        report_verification_started=_report,
    )
    module = TaskService.create_task.__globals__
    monkeypatch.setattr(module["InteractiveShellService"], "__init__", _Shell.__init__, raising=False)
    monkeypatch.setattr(module["InteractiveShellService"], "__aenter__", _Shell.__aenter__, raising=False)
    monkeypatch.setattr(module["InteractiveShellService"], "__aexit__", _Shell.__aexit__, raising=False)
    monkeypatch.setattr(module["settings"], "DRY_RUN", False, raising=False)
    with patch.object(module["ResultHandler"], "handle_result", AsyncMock(return_value=handled)), patch.object(
        module["ResultHandler"], "__init__", lambda self, *a, **k: None
    ):
        keypair = SimpleNamespace(ss58_address="5Val")
        await TaskService.create_task(
            service,
            miner_info=SimpleNamespace(job_batch_id="b", miner_hotkey="5Miner"),
            executor_info=SimpleNamespace(
                uuid="exec-uuid-1", address="1.2.3.4", ssh_username="root", ssh_port=22, port=8080, root_dir="/"
            ),
            keypair=keypair,
            private_key="enc",
            public_key="pub",
            encrypted_files=MagicMock(),
            rented_data=MagicMock(),
            default_docker_image_digests={},
        )

    assert order == ["report:pipe-1:exec-uuid-1", "pipeline.run"]
