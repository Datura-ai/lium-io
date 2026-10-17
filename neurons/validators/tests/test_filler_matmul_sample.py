import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import build_services, build_state
from neurons.validators.src.services import matrix_validation_service as mvs
from neurons.validators.src.services.matrix_validation_service import ValidationResult
from neurons.validators.src.services.task.checks import capability
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse


class SamplingValidationService:
    def __init__(self, *, success: bool = True, raises: bool = False, result=None):
        self.success = success
        self.raises = raises
        self.result = result
        self.budgets: list[int | None] = []
        self.probe_all_cards: list[bool] = []
        self.gpu_uuids: list[str | None] = []

    async def validate_gpu_model_and_process_job(
        self, *, vram_budget_mb=None, probe_all_cards=True, gpu_uuid=None, **_
    ):
        self.budgets.append(vram_budget_mb)
        self.probe_all_cards.append(probe_all_cards)
        self.gpu_uuids.append(gpu_uuid)
        if self.raises:
            raise RuntimeError("ssh went away")
        if self.result is not None:
            return self.result
        return ValidationResult(
            success=self.success, error_message="" if self.success else "no card"
        )


@pytest.fixture(autouse=True)
def no_sample_count_carried_between_tests():
    capability._filler_cycles_since_sample.clear()
    capability._filler_samples_taken.clear()
    yield
    capability._filler_cycles_since_sample.clear()
    capability._filler_samples_taken.clear()


async def _filler_cycles(context_factory, service, cycles: int, specs=None) -> list:
    results = []
    for _ in range(cycles):
        state = build_state(
            specs=specs or {"gpu": {"count": 1}},
            rented_data=RentedExecutorsResponse(
                executors={},
                banned_guids=[],
                filler_containers_by_executor={"executor-123": "filler_active"},
            ),
        )
        ctx = context_factory(services=build_services(validation=service), state=state)
        results.append(await CapabilityCheck().run(ctx))
    return results


@pytest.mark.asyncio
async def test_no_sample_runs_by_default(context_factory):
    service = SamplingValidationService()

    results = await _filler_cycles(context_factory, service, 5)

    assert service.budgets == []
    assert all(r.event.what_we_saw == {"filler_container": "filler_active"} for r in results)


@pytest.mark.asyncio
async def test_every_nth_filler_cycle_runs_a_small_sample(context_factory, monkeypatch):
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 3)
    service = SamplingValidationService()

    results = await _filler_cycles(context_factory, service, 7)

    assert service.budgets == [capability.settings.FILLER_MATMUL_SAMPLE_VRAM_MB] * 2
    assert service.probe_all_cards == [False, False]
    assert ["matmul_sample" in r.event.what_we_saw for r in results] == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    assert results[2].event.what_we_saw["matmul_sample"]["passed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service", [SamplingValidationService(success=False), SamplingValidationService(raises=True)]
)
async def test_a_failed_sample_is_recorded_and_the_filler_skip_still_passes(
    context_factory, monkeypatch, service
):
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 1)

    (result,) = await _filler_cycles(context_factory, service, 1)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert result.event.what_we_saw["matmul_sample"]["passed"] is False
    assert result.event.what_we_saw["matmul_sample"]["error"]


def test_the_default_sample_budget_leaves_room_for_the_matrix(monkeypatch):
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: MagicMock())
    service = mvs.ValidationService()

    dim_k = service.get_max_matrix_dimensions(capability.settings.FILLER_MATMUL_SAMPLE_VRAM_MB, 2000)

    assert dim_k > 0


def _cards(count: int) -> dict:
    return {
        "gpu": {
            "count": count,
            "details": [
                {"name": "NVIDIA H100 80GB HBM3", "uuid": f"GPU-{i}", "capacity": 81920}
                for i in range(count)
            ],
        }
    }


@pytest.mark.asyncio
async def test_samples_rotate_over_every_claimed_card_within_the_budget(
    context_factory, monkeypatch
):
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 1)
    service = SamplingValidationService()

    results = await _filler_cycles(context_factory, service, 4, specs=_cards(3))

    assert service.gpu_uuids == ["GPU-0", "GPU-1", "GPU-2", "GPU-0"]
    assert service.budgets == [capability.settings.FILLER_MATMUL_SAMPLE_VRAM_MB] * 4
    assert service.probe_all_cards == [False] * 4
    assert [r.event.what_we_saw["matmul_sample"]["card_index"] for r in results] == [0, 1, 2, 0]


def _no_answer(stderr: str) -> ValidationResult:
    return ValidationResult(
        success=False,
        expected_uuid="nonce",
        returned_uuid="None",
        stderr=stderr,
        error_message=f"{mvs.UUID_MISMATCH_ERROR_PREFIX}: expected 'nonce', got 'None'",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "kind"),
    [
        (_no_answer("CUDA error: out of memory"), "no_memory"),
        (
            ValidationResult(
                success=False,
                expected_uuid="nonce",
                returned_uuid="other",
                error_message=f"{mvs.UUID_MISMATCH_ERROR_PREFIX}: expected 'nonce', got 'other'",
            ),
            "wrong_answer",
        ),
        (ValidationResult(success=False, timed_out=True, error_message="timed out"), "timeout"),
        (
            ValidationResult(success=False, error_message="ssh failed", validator_error=True),
            "validator_error",
        ),
    ],
)
async def test_a_failed_sample_names_why(context_factory, monkeypatch, result, kind):
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 1)

    (check,) = await _filler_cycles(
        context_factory, SamplingValidationService(result=result), 1, specs=_cards(2)
    )

    assert check.passed is True
    assert check.event.what_we_saw["matmul_sample"]["error_kind"] == kind


@pytest.mark.asyncio
async def test_a_pinned_run_targets_one_card_and_keys_on_it(monkeypatch):
    wrapper = MagicMock()
    wrapper.getCipherText.return_value = "deadbeef"
    wrapper._has_sealed = False
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: wrapper)
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(stdout="UUID: x", stderr="")))

    await mvs.ValidationService().validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=SimpleNamespace(root_dir="/root/app", python_path="python"),
        default_extra={},
        machine_spec=_cards(3),
        vram_budget_mb=4096,
        probe_all_cards=False,
        gpu_uuid="GPU-1",
    )

    assert ssh.run.call_args.args[0].startswith("CUDA_VISIBLE_DEVICES=GPU-1 python ")
    machine_info = json.loads(wrapper.generateChallenge.call_args.args[2])
    assert machine_info == {"uuids": "GPU-1", "gpu_count": 1, "gpu_model": "NVIDIA H100 80GB HBM3"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cipher", ["", "deadbeef"])
async def test_an_ssh_or_cipher_failure_is_flagged_as_ours(monkeypatch, cipher):
    wrapper = MagicMock()
    wrapper.getCipherText.return_value = cipher
    wrapper._has_sealed = False
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: wrapper)
    ssh = SimpleNamespace(run=AsyncMock(side_effect=OSError("connection lost")))

    result = await mvs.ValidationService().validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=SimpleNamespace(root_dir="/root/app", python_path="python"),
        default_extra={},
        machine_spec=_cards(3),
        vram_budget_mb=4096,
        probe_all_cards=False,
        gpu_uuid="GPU-1",
    )

    assert result.success is False
    assert result.validator_error is True
