from types import SimpleNamespace

import pytest
from core.config import Settings, settings
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from services.executor_image_policy import (
    ExpectedImage,
    ExpectedImageSnapshot,
    ImageVerdict,
)
from services.task.checks.executor_image import ExecutorImageCheck, observed_executor_digest
from services.task.checks.rented_machine import TenantEnforcementCheck
from services.task.pipeline_factory import PipelineFactory
from services.task.result_handler import ResultHandler
from services.task.score_calculator import calculate_scores

from tests.helpers import build_context_config, build_state, make_context

EXECUTOR_DIGEST = f"sha256:{'a' * 64}"
STALE_DIGEST = f"sha256:{'c' * 64}"


@pytest.fixture
def enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", True)


@pytest.fixture
def warn_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", False)


def policy() -> ExpectedImageSnapshot:
    return ExpectedImageSnapshot(
        executor=ExpectedImage("executor:latest", EXECUTOR_DIGEST),
        executor_ref="executor:latest",
    )


def rented_data() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="test-miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[8080, 8081])],
            )
        },
    )


def specs(*, executor_digest: str = EXECUTOR_DIGEST) -> dict:
    return {
        "docker": {
            "container_id": "executor-id",
            "containers": [
                {
                    "container_id": "executor-id",
                    "digest": executor_digest,
                    "name": "executor-executor-1",
                },
            ],
        }
    }


def test_observation_uses_container_id_then_name():
    assert observed_executor_digest(specs()) == EXECUTOR_DIGEST
    assert (
        observed_executor_digest(
            {
                "docker": {
                    "containers": [
                        {
                            "container_id": "other-id",
                            "digest": EXECUTOR_DIGEST,
                            "name": "provider-stack-executor-1",
                        }
                    ]
                }
            }
        )
        == EXECUTOR_DIGEST
    )


def test_observation_returns_none_when_ambiguous():
    assert (
        observed_executor_digest(
            {
                "docker": {
                    "containers": [
                        {"digest": EXECUTOR_DIGEST, "name": "executor-executor-1"},
                        {"digest": EXECUTOR_DIGEST, "name": "provider-executor-1"},
                    ]
                }
            }
        )
        is None
    )


def test_enforcement_is_off_by_default():
    # Regression: flipping the default back to True zeroes idle pay for every node still on
    # the old image before DAH-3419 restores auto-update (99 executors on 11 Sep).
    assert Settings.model_fields["EXECUTOR_IMAGE_CHECK_ENFORCE"].default is False


@pytest.mark.asyncio
async def test_unrented_outdated_executor_fails_validation(enforce):
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=SimpleNamespace(executors={}),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is False
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED
    assert "unavailable for rent" in result.event.impact
    assert result.event.severity == "error"
    assert result.event.context["executor_image_check_enforced"] is True
    assert "earns no incentive" in result.event.remediation


@pytest.mark.asyncio
async def test_warning_only_unrented_outdated_executor_passes_and_logs_digests(warn_only):
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=SimpleNamespace(executors={}),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED
    assert result.updates["default_extra"]["executor_image_status"] == "OUTDATED"
    assert result.event.reason_code == "EXECUTOR_IMAGE_OUTDATED"
    assert result.event.severity == "warning"
    assert result.event.what_we_saw["observed_digest"] == STALE_DIGEST
    assert result.event.what_we_saw["expected_digest"] == EXECUTOR_DIGEST
    assert "EXECUTOR_IMAGE_CHECK_ENFORCE" in result.event.impact
    assert "EXECUTOR_IMAGE_CHECK_ENFORCE" in result.event.remediation
    assert "earns no incentive" not in result.event.remediation
    assert result.event.context["executor_image_check_enforced"] is False


@pytest.mark.asyncio
async def test_rented_outdated_executor_passes_validation(enforce):
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=rented_data(),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED


@pytest.mark.asyncio
async def test_current_executor_passes_validation():
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=EXECUTOR_DIGEST),
            rented_data=SimpleNamespace(executors={}),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates["state"].executor_image_report.status is ImageVerdict.CURRENT


@pytest.mark.asyncio
async def test_unobservable_executor_digest_is_outdated(enforce):
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs={"docker": {"containers": []}},
            rented_data=rented_data(),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED
    assert result.event.reason_code == "EXECUTOR_IMAGE_OUTDATED"


@pytest.mark.asyncio
async def test_cvm_outdated_executor_skips_image_check():
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=SimpleNamespace(executors={}),
        ),
        tdx_attestation_passed=True,
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates == {}
    assert result.event.reason_code == "EXECUTOR_IMAGE_SKIPPED"
    assert result.event.context.get("skip_reason") == "cvm"


@pytest.mark.asyncio
async def test_missing_expected_state_skips_without_penalty():
    unknown_policy = ExpectedImageSnapshot(
        executor=None,
        executor_ref="executor:latest",
    )
    context = make_context(
        config=build_context_config(executor_image_snapshot=unknown_policy),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=SimpleNamespace(executors={}),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates == {}
    assert result.event.reason_code == "EXECUTOR_IMAGE_SKIPPED"


def test_image_check_is_fatal():
    assert ExecutorImageCheck.fatal is True


def test_image_check_precedes_tenant_enforcement_in_both_pipelines():
    for checks in (PipelineFactory.build_checks(), PipelineFactory.build_dry_run_checks()):
        image_index = next(
            index for index, check in enumerate(checks) if isinstance(check, ExecutorImageCheck)
        )
        tenant_index = next(
            index for index, check in enumerate(checks) if isinstance(check, TenantEnforcementCheck)
        )
        assert image_index < tenant_index


@pytest.mark.parametrize(
    ("rented", "expected_job_score"),
    [(False, 0.0), (True, 1.0)],
)
def test_score_calculator_zeroes_outdated_executor(
    enforce,
    rented: bool,
    expected_job_score: float,
):
    report = policy().report(STALE_DIGEST)
    context = make_context(
        state=build_state(
            gpu_model="NVIDIA H200",
            executor_image_report=report,
        ),
        collateral_deposited=True,
    )

    actual_score, job_score, warning = calculate_scores(context, rented)

    assert actual_score == 0.0
    assert job_score == expected_job_score
    assert "Required executor image is outdated" in warning


@pytest.mark.parametrize("rented", [False, True])
def test_warning_only_score_calculator_leaves_outdated_executor_scored(warn_only, rented: bool):
    report = policy().report(STALE_DIGEST)
    context = make_context(
        state=build_state(
            gpu_model="NVIDIA H200",
            executor_image_report=report,
            specs={"network": {"ema_verifyx_download_speed": 100.0}},
        ),
        collateral_deposited=True,
    )

    actual_score, job_score, warning = calculate_scores(context, rented)

    assert actual_score == 1.0
    assert job_score == 1.0
    assert "outdated" not in warning


@pytest.mark.asyncio
async def test_result_handler_publishes_report_on_job_result_and_specs():
    report = policy().report(EXECUTOR_DIGEST)
    context = make_context(
        state=build_state(specs={"gpu": {}}, executor_image_report=report),
    )

    result = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=context,
        miner_info=SimpleNamespace(miner_hotkey="miner", job_batch_id="batch"),
        executor_info=context.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert result.executor_image_report == report.as_dict()
    assert result.spec["executor_image"] == report.as_dict()
