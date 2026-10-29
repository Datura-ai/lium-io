from types import SimpleNamespace

import pytest
from core.config import settings
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
from services.task.checks.executor_image import ExecutorImageCheck
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


