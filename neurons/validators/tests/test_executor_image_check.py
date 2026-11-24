from dataclasses import replace
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
from services.default_docker_image_digest_service import executor_build_published as real_executor_build_published
from services.task.checks.executor_image import ExecutorImageCheck
from services.task.score_calculator import calculate_scores

from tests.helpers import build_context_config, build_state, make_context

EXECUTOR_DIGEST = f"sha256:{'a' * 64}"
STALE_DIGEST = f"sha256:{'c' * 64}"


@pytest.fixture
def unofficial_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_UNOFFICIAL_ENFORCE", True)


def policy() -> ExpectedImageSnapshot:
    return ExpectedImageSnapshot(
        executor=ExpectedImage("executor:latest", EXECUTOR_DIGEST),
        executor_ref="executor:latest",
    )


def rented_data(miner_hotkey: str = "test-miner", port: str = "22") -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey=miner_hotkey,
                executor_ip_address="127.0.0.1",
                executor_ip_port=port,
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
async def test_unrented_outdated_executor_fails_validation():
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
    assert "off the rent list" in result.event.remediation
    assert "a rented node keeps its rental and incentive" in result.event.remediation


@pytest.mark.asyncio
async def test_rented_outdated_executor_passes_and_keeps_its_incentive():
    # DAH-6845: the renter's pod is untouched; the node leaves the rent list only once the rental ends
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(specs=specs(executor_digest=STALE_DIGEST), rented_data=rented_data()),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED
    assert result.event.impact == "Rental and incentive continue; update the image before the rental ends"


@pytest.mark.asyncio
async def test_unobservable_executor_digest_is_unofficial(unofficial_enforced, no_registry_lookup_for_executor_build):
    no_registry_lookup_for_executor_build.side_effect = real_executor_build_published
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs={"docker": {"containers": []}},
            rented_data=rented_data(),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True  # rented: the rental stays, the score path zeroes the incentive
    assert result.updates["state"].executor_image_report.status is ImageVerdict.UNOFFICIAL
    assert result.event.reason_code == "EXECUTOR_IMAGE_UNOFFICIAL"


def test_score_calculator_zeroes_an_outdated_free_node():
    report = policy().report(STALE_DIGEST)
    context = make_context(
        state=build_state(gpu_model="NVIDIA H200", executor_image_report=report), collateral_deposited=True
    )

    actual_score, job_score, warning = calculate_scores(context, False)

    assert (actual_score, job_score) == (0.0, 0.0)
    assert "Required executor image is outdated" in warning


def test_score_calculator_leaves_an_outdated_rented_node_its_score():
    report = replace(policy().report(STALE_DIGEST), rented_when_read=True)
    outdated_read_while_rented = make_context(
        state=build_state(gpu_model="NVIDIA H200", executor_image_report=report), collateral_deposited=True
    )
    current = make_context(state=build_state(gpu_model="NVIDIA H200"), collateral_deposited=True)

    assert calculate_scores(outdated_read_while_rented, True) == calculate_scores(current, True)
    assert calculate_scores(outdated_read_while_rented, True)[0] > 0


@pytest.mark.asyncio
async def test_outdated_executor_under_another_miners_rental_on_a_different_endpoint_fails():
    # DAH-4263: another miner's rental on a different endpoint is not this node's rental.
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=rented_data(miner_hotkey="other-miner-hotkey", port="2222"),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is False


@pytest.mark.asyncio
async def test_unrented_executor_on_a_build_we_never_published_fails(
    unofficial_enforced, no_registry_lookup_for_executor_build
):
    no_registry_lookup_for_executor_build.return_value = False
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(specs=specs(executor_digest=STALE_DIGEST), rented_data=SimpleNamespace(executors={})),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is False
    assert result.updates["state"].executor_image_report.status is ImageVerdict.UNOFFICIAL
    assert result.event.reason_code == "EXECUTOR_IMAGE_UNOFFICIAL"
    no_registry_lookup_for_executor_build.assert_awaited_once_with(STALE_DIGEST)


@pytest.mark.asyncio
@pytest.mark.parametrize(("published", "build"), [(False, "unofficial"), (True, "official"), (None, "unknown")])
async def test_with_unofficial_enforcement_off_a_build_is_judged_as_outdated_and_logged(
    no_registry_lookup_for_executor_build, caplog, published, build
):
    no_registry_lookup_for_executor_build.return_value = published
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(specs=specs(executor_digest=STALE_DIGEST), rented_data=SimpleNamespace(executors={})),
    )

    with caplog.at_level("INFO"):
        result = await ExecutorImageCheck().run(context)

    assert result.passed is False  # a free node on an old image
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED
    logged = [r for r in caplog.records if "EXECUTOR_BUILD" in r.getMessage()]
    assert len(logged) == 1 and logged[0].levelname == "INFO"
    assert logged[0].msg.extra["executor_build"] == build


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [False, True])
async def test_an_older_published_build_fails_only_a_free_node(no_registry_lookup_for_executor_build, rented):
    no_registry_lookup_for_executor_build.return_value = True
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(
            specs=specs(executor_digest=STALE_DIGEST),
            rented_data=rented_data() if rented else SimpleNamespace(executors={}),
        ),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is rented
    assert result.updates["state"].executor_image_report.status is ImageVerdict.OUTDATED


@pytest.mark.asyncio
async def test_current_build_needs_no_registry_lookup(no_registry_lookup_for_executor_build):
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(specs=specs(), rented_data=SimpleNamespace(executors={})),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is True
    no_registry_lookup_for_executor_build.assert_not_awaited()


@pytest.mark.parametrize("rented", [False, True])
def test_score_calculator_zeroes_an_unofficial_build_rented_or_not(rented: bool):
    report = replace(policy().report(STALE_DIGEST), status=ImageVerdict.UNOFFICIAL)
    context = make_context(
        state=build_state(gpu_model="NVIDIA H200", executor_image_report=report),
        collateral_deposited=True,
    )

    actual_score, _, _ = calculate_scores(context, rented)

    assert actual_score == 0.0


@pytest.mark.asyncio
async def test_the_previous_release_in_its_grace_passes_and_keeps_incentive():
    # DAH-6560: CURRENT feeds the image check and the incentive, not only the portal card
    in_grace = replace(policy(), rollout_previous_digest=STALE_DIGEST)
    context = make_context(
        config=build_context_config(executor_image_snapshot=in_grace),
        state=build_state(specs=specs(executor_digest=STALE_DIGEST), rented_data=SimpleNamespace(executors={})),
    )

    result = await ExecutorImageCheck().run(context)
    report = result.updates["state"].executor_image_report
    scored = make_context(
        state=build_state(gpu_model="NVIDIA H200", executor_image_report=report), collateral_deposited=True
    )
    _, _, warning = calculate_scores(scored, False)

    assert result.passed is True
    assert report.status is ImageVerdict.CURRENT
    assert "outdated" not in (warning or "")
    # after the grace the same node is OUTDATED again
    assert policy().report(STALE_DIGEST).status is ImageVerdict.OUTDATED


@pytest.mark.asyncio
async def test_a_rental_ending_after_the_image_was_read_keeps_the_node_its_score():
    # DAH-6845 review: the outdated reading was taken while rented; the rental ending before the cycle's score must
    # not turn it into a free-node exclusion, the next cycle reads the image again on a free node
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(specs=specs(executor_digest=STALE_DIGEST), rented_data=rented_data()),
    )

    result = await ExecutorImageCheck().run(context)
    report = result.updates["state"].executor_image_report
    rental_ended = make_context(
        state=build_state(gpu_model="NVIDIA H200", executor_image_report=report), collateral_deposited=True
    )
    current = make_context(state=build_state(gpu_model="NVIDIA H200"), collateral_deposited=True)

    assert result.passed is True
    assert report.status is ImageVerdict.OUTDATED and report.rented_when_read is True
    assert report.as_dict()["rented_when_read"] is True
    assert calculate_scores(rental_ended, False) == calculate_scores(current, False)
    assert "outdated" not in (calculate_scores(rental_ended, False)[2] or "")


@pytest.mark.asyncio
async def test_an_outdated_image_read_on_a_free_node_records_it_free():
    context = make_context(
        config=build_context_config(executor_image_snapshot=policy()),
        state=build_state(specs=specs(executor_digest=STALE_DIGEST), rented_data=SimpleNamespace(executors={})),
    )

    result = await ExecutorImageCheck().run(context)

    assert result.passed is False
    assert result.updates["state"].executor_image_report.rented_when_read is False


def test_only_an_outdated_image_text_says_a_rented_node_keeps_its_incentive():
    # an UNOFFICIAL build earns nothing, rented or not: its no-pay line must not promise the incentive
    from services.executor_image_policy import outdated_image_remediation

    assert "keeps its rental and incentive" in outdated_image_remediation("executor:latest", "OUTDATED")
    unofficial = outdated_image_remediation("executor:latest", "UNOFFICIAL")
    assert "incentive" not in unofficial and unofficial.endswith("stopped-updating.")
