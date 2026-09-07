"""DAH-3011: the fast first pass for a never-validated executor.

Only a caller that passes `first_pass=True` (the express lane, DAH-2958 — spec-only, never scored)
with FIRST_PASS_FAST_PATH_ENABLED on gets it: the capability matmul is sized from a VRAM budget,
VerifyX writes less RAM/disk, and a cold bandwidth sample is published but not enforced. The wave
never sets `first_pass`, so a scored verification is byte-for-byte today's.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import services.matrix_validation_service as mvs
from core.config import settings
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as CapMsg
from neurons.validators.src.services.task.messages import VerifyXMessages as VxMsg
from neurons.validators.src.services.task import pipeline_factory as pipeline_factory_module
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from neurons.validators.src.services.verifyx_validation_service import VerifyXValidationService
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse

from tests.helpers import build_context_config, build_services, build_state
from tests.test_capability_check import DummyValidationService
from tests.test_verifyx_check import MockVerifyXResponse

EXECUTOR = "executor-123"


@pytest.fixture
def fast_path_on(monkeypatch):
    monkeypatch.setattr(settings, "FIRST_PASS_FAST_PATH_ENABLED", True)


# --- plumbing: the flag AND the caller's first_pass are both needed ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,first_pass,expected",
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
async def test_context_first_pass_needs_the_flag_and_the_caller(monkeypatch, flag, first_pass, expected):
    monkeypatch.setattr(settings, "FIRST_PASS_FAST_PATH_ENABLED", flag)
    # Context validates a live asyncssh connection; what is under test is the ContextConfig
    # build_context assembles, so capture the kwargs instead of constructing the model.
    monkeypatch.setattr(pipeline_factory_module, "Context", lambda **kw: SimpleNamespace(**kw))
    redis = SimpleNamespace(
        get_verified_job_info=AsyncMock(return_value={}),
        is_elem_exists_in_set=AsyncMock(return_value=False),
    )
    factory = PipelineFactory.__new__(PipelineFactory)
    factory.redis_service = redis
    for name in (
        "ssh_service",
        "validation_service",
        "verifyx_validation_service",
        "inspector_validation_service",
        "collateral_contract_service",
        "executor_connectivity_service",
        "backend_client",
        "pod_recovery",
        "container_cleanup",
    ):
        setattr(factory, name, MagicMock())
    shell = SimpleNamespace(ssh_client=MagicMock())
    executor = SimpleNamespace(
        uuid=EXECUTOR, address="1.2.3.4", port=8000, ssh_username="root", ssh_port=22, root_dir="/root/app"
    )
    encrypted_files = SimpleNamespace(
        encrypt_key="k", machine_scrape_file_name="scrape", machine_scrape_source=None,
        all_keys={}, tmp_directory="/tmp/x",
    )
    miner_info = SimpleNamespace(
        job_batch_id="b", miner_hotkey="m", miner_coldkey="c", miner_address="1.1.1.1", miner_port=1
    )

    ctx = await factory.build_context(
        shell=shell,
        miner_info=miner_info,
        executor_info=executor,
        keypair=None,
        private_key="p",
        public_key="q",
        encrypted_files=encrypted_files,
        rented_data=None,
        default_docker_image_digests={},
        first_pass=first_pass,
    )

    assert ctx.config.first_pass is expected


def test_context_config_default_is_not_a_first_pass():
    assert build_context_config().first_pass is False


# --- capability matmul --------------------------------------------------------------------


class _SizingAwareValidationService(DummyValidationService):
    async def validate_gpu_model_and_process_job(self, *, ssh_client, executor_info, default_extra, machine_spec, **kw):
        self.sizing_kwargs = kw
        return await super().validate_gpu_model_and_process_job(
            ssh_client=ssh_client, executor_info=executor_info, default_extra=default_extra, machine_spec=machine_spec
        )


@pytest.mark.asyncio
async def test_scored_capability_call_passes_no_sizing(fast_path_on, context_factory):
    service = _SizingAwareValidationService(success=True)
    ctx = context_factory(
        services=build_services(validation=service),
        config=build_context_config(first_pass=False),
        state=build_state(specs={"gpu": {"count": 1}}),
    )

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert service.sizing_kwargs == {}
    assert "first_pass_vram_budget_mb" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_first_pass_capability_call_sizes_the_matmul_from_the_budget(fast_path_on, context_factory):
    service = _SizingAwareValidationService(success=True, metrics={"fp32_tflops": 1.0})
    ctx = context_factory(
        services=build_services(validation=service),
        config=build_context_config(first_pass=True),
        state=build_state(specs={"gpu": {"count": 1}}),
    )

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == CapMsg.VERIFY_OK.reason
    assert service.sizing_kwargs == {"vram_budget_mb": settings.FIRST_PASS_MATMUL_VRAM_MB}
    assert result.event.what_we_saw["first_pass_vram_budget_mb"] == 8192
    # A failed matmul still fails the first pass: the check itself is not skipped.
    failing = _SizingAwareValidationService(success=False)
    ctx = context_factory(
        services=build_services(validation=failing),
        config=build_context_config(first_pass=True),
        state=build_state(specs={"gpu": {"count": 1}}),
    )
    assert (await CapabilityCheck().run(ctx)).passed is False


@pytest.fixture
def matrix_service(monkeypatch):
    wrapper = MagicMock(name="DMCompVerifyWrapper")
    wrapper.DMCompVerify_new.return_value = "ptr"
    wrapper.getCipherText.return_value = "deadbeef"
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: wrapper)
    return mvs.ValidationService(), wrapper


def _h200_spec():
    return {"gpu": {"count": 1, "details": [{"name": "NVIDIA H200", "uuid": "GPU-1", "capacity": 143771}]}}


@pytest.mark.asyncio
async def test_vram_budget_shrinks_dim_k_and_default_is_full_card(matrix_service):
    svc, wrapper = matrix_service
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(stdout="UUID: nope", stderr="")))
    executor = SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python3")

    async def sized_for(machine_spec, **kw) -> tuple[int, int]:
        await svc.validate_gpu_model_and_process_job(
            ssh_client=ssh, executor_info=executor, default_extra={}, machine_spec=machine_spec, **kw
        )
        _ptr, dim_n, dim_k = wrapper.setDimension.call_args_list[-1].args
        return dim_n, dim_k

    dim_n, full_dim_k = await sized_for(_h200_spec())
    assert full_dim_k == int(svc.get_max_matrix_dimensions(143771, dim_n))
    dim_n, budget_dim_k = await sized_for(_h200_spec(), vram_budget_mb=8192)
    assert budget_dim_k == int(svc.get_max_matrix_dimensions(8192, dim_n))
    # (143771 - 4096) MB vs (8192 - 2048) MB of matrices: 20x less to generate and copy.
    assert full_dim_k > 15 * budget_dim_k
    assert budget_dim_k > 100_000  # still a real matmul, not a no-op

    # A card smaller than the budget is sized as today.
    small = {"gpu": {"count": 1, "details": [{"name": "RTX 3070", "uuid": "GPU-2", "capacity": 8192}]}}
    dim_n, small_dim_k = await sized_for(small, vram_budget_mb=16384)
    assert small_dim_k == int(svc.get_max_matrix_dimensions(8192, dim_n))


# --- VerifyX ------------------------------------------------------------------------------


class _ConfigAwareVerifyX:
    def __init__(self, response: MockVerifyXResponse):
        self._response = response
        self.kwargs: dict = {}

    async def validate_verifyx_and_process_job(self, *, shell, executor_info, default_extra, machine_spec, **kw):
        self.kwargs = kw
        return self._response


def _probe(download: float | None) -> MockVerifyXResponse:
    network = {} if download is None else {"download_speed": download, "upload_speed": 30.0}
    return MockVerifyXResponse(data={"success": True, "ram": {"total": 64}, "hard_disk": {"total": 900}, "network": network})


def _ctx(context_factory, service, *, first_pass: bool, rented_data=None):
    return context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(verifyx_enabled=True, first_pass=first_pass),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=rented_data),
    )


def _never_measured():
    return RentedExecutorsResponse(executors={}, banned_guids=[], network_ema={})


@pytest.mark.asyncio
async def test_scored_verifyx_call_passes_no_overrides_and_keeps_the_gate(fast_path_on, context_factory):
    service = _ConfigAwareVerifyX(_probe(59.9))

    result = await VerifyXCheck().run(_ctx(context_factory, service, first_pass=False, rented_data=_never_measured()))

    assert service.kwargs == {}
    assert result.passed is False
    assert result.event.reason_code == VxMsg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason


@pytest.mark.asyncio
async def test_first_pass_verifyx_uses_the_smaller_challenge_config(fast_path_on, context_factory):
    service = _ConfigAwareVerifyX(_probe(500.0))

    result = await VerifyXCheck().run(_ctx(context_factory, service, first_pass=True, rented_data=_never_measured()))

    assert service.kwargs == {
        "challenge_config_overrides": {"memory_max_test_gb": 16, "storage_throughput_test_gb": 1}
    }
    assert result.passed is True
    assert result.event.what_we_saw["first_pass_challenge_config"] == {
        "memory_max_test_gb": 16,
        "storage_throughput_test_gb": 1,
    }
    # A warm sample seeds the EMA exactly as today.
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_first_pass_cold_sample_is_published_but_the_gate_is_deferred(fast_path_on, context_factory):
    service = _ConfigAwareVerifyX(_probe(59.9))

    result = await VerifyXCheck().run(_ctx(context_factory, service, first_pass=True, rented_data=_never_measured()))

    assert result.passed is True
    assert result.event.reason_code == VxMsg.VERIFY_SUCCESS.reason
    assert result.event.what_we_saw["bandwidth_gate"] == "deferred_to_first_scored_cycle"
    assert result.event.what_we_saw["first_sample_download_speed_mbps"] == 59.9
    network = result.updates["state"].specs["network"]
    assert network["verifyx_download_speed"] == 59.9  # measured and published
    assert "ema_verifyx_download_speed" not in network  # not seeded: the first scored cycle bootstraps
    assert "ema_verifyx_upload_speed" not in network
    assert result.updates["state"].specs["ram"] == {"total": 64}


@pytest.mark.asyncio
async def test_first_pass_failed_network_probe_is_deferred_too(fast_path_on, context_factory):
    result = await VerifyXCheck().run(
        _ctx(context_factory, _ConfigAwareVerifyX(_probe(None)), first_pass=True, rented_data=_never_measured())
    )

    assert result.passed is True
    assert result.event.what_we_saw["bandwidth_gate"] == "deferred_to_first_scored_cycle"
    assert result.event.what_we_saw["first_sample_download_speed_mbps"] is None


@pytest.mark.asyncio
async def test_first_pass_on_a_host_with_an_ema_keeps_the_gate(fast_path_on, context_factory):
    known = RentedExecutorsResponse(
        executors={}, banned_guids=[],
        network_ema={EXECUTOR: NetworkEMA(ema_verifyx_download_speed=110.0, ema_verifyx_upload_speed=40.0)},
    )

    result = await VerifyXCheck().run(_ctx(context_factory, _ConfigAwareVerifyX(_probe(20.0)), first_pass=True, rented_data=known))

    assert result.passed is False
    assert result.event.reason_code == VxMsg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason


@pytest.mark.asyncio
async def test_first_pass_ram_or_disk_failure_still_fails(fast_path_on, context_factory):
    failed = MockVerifyXResponse(data={"success": False, "errors": ["Insufficient memory: 6 GB allocated, 8 GB required"]})

    result = await VerifyXCheck().run(_ctx(context_factory, _ConfigAwareVerifyX(failed), first_pass=True, rented_data=_never_measured()))

    assert result.passed is False
    assert result.event.reason_code == VxMsg.VERIFY_FAILED.reason


@pytest.mark.asyncio
async def test_flag_off_first_pass_is_todays_behaviour(monkeypatch, context_factory):
    monkeypatch.setattr(settings, "FIRST_PASS_FAST_PATH_ENABLED", False)
    # The flag is resolved in build_context, so ContextConfig.first_pass is False; here we assert
    # the checks themselves also do nothing special when it is False.
    verifyx = _ConfigAwareVerifyX(_probe(59.9))
    result = await VerifyXCheck().run(_ctx(context_factory, verifyx, first_pass=False, rented_data=_never_measured()))
    assert verifyx.kwargs == {}
    assert result.passed is False

    matmul = _SizingAwareValidationService(success=True)
    ctx = context_factory(
        services=build_services(validation=matmul),
        config=build_context_config(first_pass=False),
        state=build_state(specs={"gpu": {"count": 1}}),
    )
    await CapabilityCheck().run(ctx)
    assert matmul.sizing_kwargs == {}


@pytest.mark.asyncio
async def test_verifyx_service_applies_overrides_to_the_challenge_config_only():
    service = VerifyXValidationService()
    seen: dict = {}

    class _FakeValidator:
        def __init__(self, lib_name, seed):
            pass

        def generate_challenge(self, challenge_input):
            seen.update(challenge_input)
            return "deadbeef"

    with patch(
        "neurons.validators.src.services.verifyx_validation_service.sha256_from_path", return_value="s"
    ), patch(
        "neurons.validators.src.services.verifyx_validation_service.VerifyXValidator", _FakeValidator
    ):
        shell = MagicMock()
        shell.get_sha256_checksum_by_path = AsyncMock(return_value="s")
        shell.get_checksums_over_scp = AsyncMock(return_value="md5:s")
        shell.ssh_client = MagicMock()
        shell.ssh_client.run = AsyncMock(return_value=SimpleNamespace(stdout="", stderr="", exit_status=0))
        executor = SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python3")
        spec = {"gpu": {"count": 1, "details": [{"uuid": "u", "name": "H100"}]}}

        await service.validate_verifyx_and_process_job(shell=shell, executor_info=executor, default_extra={}, machine_spec=spec)
        default_config = dict(seen["config"])
        await service.validate_verifyx_and_process_job(
            shell=shell, executor_info=executor, default_extra={}, machine_spec=spec,
            challenge_config_overrides={"memory_max_test_gb": 16, "storage_throughput_test_gb": 1},
        )
        first_pass_config = dict(seen["config"])

    assert default_config["memory_max_test_gb"] == settings.verifyx.MEMORY_MAX_TEST_GB
    assert default_config["storage_throughput_test_gb"] == settings.verifyx.STORAGE_THROUGHPUT_TEST_GB
    assert first_pass_config == {**default_config, "memory_max_test_gb": 16, "storage_throughput_test_gb": 1}
    # Untouched: the RAM minimum the executor must still allocate, the disk minimum, the download.
    assert first_pass_config["memory_min_test_gb"] == default_config["memory_min_test_gb"]
    assert first_pass_config["storage_min_available_gb"] == default_config["storage_min_available_gb"]
    assert first_pass_config["network_timeout_seconds"] == default_config["network_timeout_seconds"]
