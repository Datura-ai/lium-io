"""Tests for MinerService._build_manual_rental_results — the synthetic forced-pass path for
special manual (bare-metal) rentals.

See .omc/plans/special-manual-rental.md sections T3, T4 and 5 for the full rationale:

- T3: a manually-rented node hands root to the renter, so the miner can no longer install the
  validator's SSH key and silently drops that executor from AcceptSSHKeyRequest.executors. The
  validator therefore never calls create_task for it and it would score 0 unless the forced pass
  is synthesised out-of-band, from the backend's manual_rental_executors list.
- T4: forcing score=1.0 is necessary but not sufficient for emissions. The incentive layer keys
  its total-count denominator off gpu_model and multiplies by gpu_count (default.py:167-174), so a
  synthetic result missing either is worth zero despite a perfect score. is_rented/sysbox_runtime
  avoid gate haircuts, and is_spot/provider_discord_connected must be read from rented_data rather
  than hardcoded, because forcing score=1.0 buys a place in the mining pool -- it does not exempt
  the node from the spot-tier or Discord-gate exclusions.
- 5 (A7/A7b/A7c/A7d/A7e/A8/A12): the acceptance criteria this file maps to, one test per row.

`_build_manual_rental_results` is a pure, non-async method (services/miner_service.py), so it is
exercised directly here rather than by driving the full websocket/REST flow.
"""

from unittest.mock import MagicMock, Mock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import BASE_GPU_MAP
from payload_models.payloads import MinerJobRequestPayload
from pydantic import BaseModel

from protocol.vc_protocol.compute_requests import (
    ManualRentalInfo,
    NetworkEMA,
    RentedExecutor,
    RentedExecutorsResponse,
)
from services.miner_service import MinerService
from services.task.models import JobResult

MINER_HOTKEY = "5MinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
OTHER_MINER_HOTKEY = "5OtherMinerHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
EXECUTOR_UUID = "executor-manual-rental-123"


def make_payload(*, miner_hotkey: str = MINER_HOTKEY) -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="batch-manual-1",
        miner_hotkey=miner_hotkey,
        miner_coldkey="5ColdkeyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        miner_address="10.0.0.5",
        miner_port=8000,
    )


def make_rented_executor(*, miner_hotkey: str = MINER_HOTKEY) -> RentedExecutor:
    return RentedExecutor(
        miner_hotkey=miner_hotkey,
        executor_ip_address="10.0.0.5",
        executor_ip_port="8080",
        pods=[],
    )


def make_real_job_result(*, uuid: str, score: float = 0.42) -> JobResult:
    """A JobResult as produced by a real, reachable TaskService.create_task run -- used to prove
    a real result beats a synthesized one (A8)."""
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
            address="10.0.0.5",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/root/app",
        ),
        score=score,
        job_score=score,
        job_batch_id="batch-manual-1",
        log_status="success",
        log_text="real validation result",
    )


def make_miner_service() -> MinerService:
    """MinerService.__init__ only assigns its four dependencies -- no I/O, no heavy setup -- so
    plain Mocks are enough; _build_manual_rental_results touches none of them."""
    return MinerService(
        ssh_service=Mock(),
        task_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        backend_client=MagicMock(),
        file_encrypt_service=MagicMock(),
    )


def test_manual_rental_forced_pass_synthesizes_full_scoring_result():
    """A7: an executor absent from msg.executors (existing=[]) but present in both
    manual_rental_executors and rented_data.executors for this miner gets exactly one
    synthesized JobResult carrying every field the incentive layer needs (T4)."""
    # Arrange
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor()},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- exactly one synthesized result for the one flagged executor
    assert len(results) == 1
    result = results[0]
    # score/job_score=1.0 is the forced pass itself (D1)
    assert result.score == 1.0
    assert result.job_score == 1.0
    # is_rented exempts the min-driver gate; sysbox_runtime avoids the sysbox haircut (T4)
    assert result.is_rented is True
    assert result.sysbox_runtime is True
    # spec=None keeps save_executor_into_db a no-op on the backend (T4/V4) -- a synthetic spec
    # that validated but disagreed with the stored row would flip the executor inactive.
    assert result.spec is None
    # gpu_model/gpu_count are the incentive-layer denominator and numerator (default.py:167-174);
    # missing either means zero emission despite score=1.0.
    assert result.gpu_model == "NVIDIA H200"
    assert result.gpu_count == 8
    assert result.log_status == "success"
    assert result.executor_info.uuid == EXECUTOR_UUID


def test_manual_rental_forced_pass_yields_no_duplicate_when_real_result_exists():
    """A8: if the node answered after all and `existing` already contains a real JobResult for
    it, the synthesized pass must return [] for that executor -- a real result always wins over a
    synthetic one, and there must be exactly one JobResult per executor id overall."""
    # Arrange
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor()},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
    )
    real_result = make_real_job_result(uuid=EXECUTOR_UUID)
    service = make_miner_service()

    # Act
    synthetic_results = service._build_manual_rental_results(
        payload, rented_data, existing=[real_result]
    )

    # Assert -- nothing new is synthesized for an executor already covered by a real result
    assert synthetic_results == []
    # The combined result set (as miner_service.py builds it via results.extend(...)) still has
    # exactly one JobResult for this executor id, and it is the real one.
    combined = [real_result] + synthetic_results
    assert len(combined) == 1
    assert combined[0] is real_result


def test_manual_rental_forced_pass_skips_executor_owned_by_different_miner():
    """A7b: an executor in manual_rental_executors whose RentedExecutor.miner_hotkey belongs to a
    different miner than payload.miner_hotkey is not synthesized for this miner's job request."""
    # Arrange
    payload = make_payload(miner_hotkey=MINER_HOTKEY)
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor(miner_hotkey=OTHER_MINER_HOTKEY)},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- this job request is for MINER_HOTKEY; the executor belongs to a different miner
    assert results == []


def test_manual_rental_forced_pass_skips_executor_missing_from_rented_executors():
    """A7c: an executor id flagged in manual_rental_executors but absent from
    rented_data.executors is skipped without raising -- there is no address/hotkey to build a
    result from."""
    # Arrange
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- no exception, and nothing to score without an address
    assert results == []


def test_manual_rental_forced_pass_skips_gpu_model_absent_from_base_gpu_map():
    """A synthetic result must never carry a gpu_model that is missing from BASE_GPU_MAP.

    A real result cannot reach the incentive layer with an unknown model -- GpuModelValidCheck is
    fatal and halts the pipeline first. A synthetic one skips the pipeline entirely, so it has to
    make that check itself: RentalPriceIncentive.get_base_model_for_gpu does a RAISING
    BASE_GPU_MAP[...] subscript (rental_price.py:176), reached from calculate_mining_scores, which
    has no per-result try/except. A manual pod whose gpu_name was frozen at rental creation and
    later retired from the map would therefore abort weight-setting for EVERY miner on the subnet
    for that cycle. Dropping the entry costs one node its emission; raising costs everyone's.
    """
    # Arrange -- a model that is deliberately not a BASE_GPU_MAP key
    assert "NVIDIA RETIRED-SKU 9000" not in BASE_GPU_MAP
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor()},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA RETIRED-SKU 9000", gpu_count=8)
        },
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- skipped rather than synthesised
    assert results == []


def test_manual_rental_forced_pass_gpu_model_is_a_real_base_gpu_map_key():
    """Guard against the BASE_GPU_MAP check being so strict it rejects legitimate models --
    a silently-empty forced pass would look identical to the feature being off."""
    assert "NVIDIA H200" in BASE_GPU_MAP


@pytest.mark.parametrize(
    "rented_data",
    [
        pytest.param(
            RentedExecutorsResponse(executors={EXECUTOR_UUID: make_rented_executor()}),
            id="manual_rental_executors_empty",
        ),
        pytest.param(None, id="rented_data_is_none"),
    ],
)
def test_manual_rental_forced_pass_returns_empty_when_nothing_flagged(rented_data):
    """A7d: an empty/absent manual_rental_executors dict returns []; rented_data=None (e.g. the
    backend call failed this cycle) also returns [] rather than raising."""
    # Arrange
    payload = make_payload()
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- nothing flagged, nothing synthesized
    assert results == []


def test_manual_rental_forced_pass_marks_spot_when_executor_in_spot_ids():
    """A7e: the spot-tier gate is read from rented_data.spot_executor_ids, not hardcoded --
    forcing score=1.0 puts the node in the mining pool, it does not exempt it from the spot-tier
    exclusion applied later in the incentive layer."""
    # Arrange
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor()},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
        spot_executor_ids=[EXECUTOR_UUID],
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- executor id is in spot_executor_ids, so the synthetic result is marked spot
    assert len(results) == 1
    assert results[0].is_spot is True


def test_manual_rental_forced_pass_discord_disconnected_when_excluded_from_list():
    """A7e: when provider_discord_connected_executor_ids is a list that does NOT contain this
    executor id, provider_discord_connected is False -- a provider without Discord connected
    earns zero even when rented (R4), and the synthetic result must not paper over that."""
    # Arrange
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor()},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
        provider_discord_connected_executor_ids=["some-other-executor-id"],
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- executor id is absent from the connected-ids list, so the gate excludes it
    assert len(results) == 1
    assert results[0].provider_discord_connected is False


def test_manual_rental_forced_pass_discord_connected_defaults_true_when_list_is_none():
    """A7e: provider_discord_connected_executor_ids=None means the field was not populated this
    cycle, so the synthetic result defaults provider_discord_connected to True."""
    # Arrange
    payload = make_payload()
    rented_data = RentedExecutorsResponse(
        executors={EXECUTOR_UUID: make_rented_executor()},
        manual_rental_executors={
            EXECUTOR_UUID: ManualRentalInfo(gpu_model="NVIDIA H200", gpu_count=8)
        },
        provider_discord_connected_executor_ids=None,
    )
    service = make_miner_service()

    # Act
    results = service._build_manual_rental_results(payload, rented_data, existing=[])

    # Assert -- None means "gate not populated"; default is connected=True
    assert len(results) == 1
    assert results[0].provider_discord_connected is True


class _PreManualRentalRentedExecutorsResponse(BaseModel):
    """Mirrors RentedExecutorsResponse exactly as it looked before this feature -- every field
    except manual_rental_executors -- to simulate an un-upgraded validator parsing a payload the
    upgraded backend now sends."""

    executors: dict[str, RentedExecutor]
    filler_containers_by_executor: dict[str, str] = {}
    banned_guids: list[str] = []
    gpu_splitting_config: dict[str, int] = {}
    network_ema: dict[str, NetworkEMA] = {}
    spot_executor_ids: list[str] = []
    new_rentals_paused_executor_ids: list[str] = []
    provider_discord_connected_executor_ids: list[str] | None = None
    default_job_owner_by_executor: dict[str, str] = {}


def test_old_validator_model_ignores_new_manual_rental_key():
    """A12: a payload dict containing the new manual_rental_executors key must parse without
    raising against a model that does not declare it. RentedExecutorsResponse declares no
    model_config, so it inherits Pydantic v2's default extra="ignore" -- an un-upgraded validator
    silently ignores the field instead of raising a ValidationError that would skip the entire
    validation cycle for every miner (R7/R8)."""
    # Arrange -- payload shaped like the upgraded backend's response
    payload_with_new_key = {
        "executors": {},
        "manual_rental_executors": {
            EXECUTOR_UUID: {"gpu_model": "NVIDIA H200", "gpu_count": 8}
        },
    }

    # Act -- parse against the OLD (pre-feature) model shape
    old = _PreManualRentalRentedExecutorsResponse.model_validate(payload_with_new_key)

    # Assert -- parses without raising, and the old model has no knowledge of the new field
    assert not hasattr(old, "manual_rental_executors")


def test_current_model_defaults_manual_rental_executors_to_empty_when_key_absent():
    """A12: the CURRENT model defaults manual_rental_executors to {} when the key is absent from
    the payload -- an old backend that has not shipped the field yet force-passes nobody
    (fail-closed), never everybody."""
    # Arrange -- payload shaped like an old (pre-feature) backend's response
    payload_without_key = {"executors": {}}

    # Act
    current = RentedExecutorsResponse.model_validate(payload_without_key)

    # Assert
    assert current.manual_rental_executors == {}
