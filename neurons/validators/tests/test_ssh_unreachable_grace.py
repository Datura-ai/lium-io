"""DAH-2748: the first cycle that cannot reach a node over SSH must not zero its score."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from services.task.ssh_unreachable_grace import (
    SSH_UNREACHABLE_REASON_CODE,
    SshUnreachableGrace,
)


def _redis() -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    return redis


@pytest.mark.asyncio
async def test_first_failure_keeps_the_last_good_score() -> None:
    # Arrange: the node scored 4.5 last cycle, and this is its first unreachable cycle.
    redis = _redis()
    redis.get = AsyncMock(side_effect=lambda key: "4.5" if key.endswith(":last_score") else None)
    grace = SshUnreachableGrace(redis)

    # Act
    verdict = await grace.score_for_unreachable_cycle("exec-1")

    # Assert
    assert verdict.score == 4.5
    assert verdict.forgiven is True
    assert verdict.streak == 1


@pytest.mark.asyncio
async def test_second_failure_in_a_row_zeroes_the_score() -> None:
    # Arrange
    redis = _redis()
    redis.get = AsyncMock(
        side_effect=lambda key: "4.5" if key.endswith(":last_score") else "1"
    )
    grace = SshUnreachableGrace(redis)

    # Act
    verdict = await grace.score_for_unreachable_cycle("exec-1")

    # Assert
    assert verdict.score == 0.0
    assert verdict.forgiven is False
    assert verdict.streak == 2


@pytest.mark.asyncio
async def test_first_failure_without_a_stored_score_scores_zero() -> None:
    # Arrange: nothing to carry forward — a node that never validated cannot be forgiven.
    grace = SshUnreachableGrace(_redis())

    # Act
    verdict = await grace.score_for_unreachable_cycle("exec-1")

    # Assert
    assert verdict.score == 0.0
    assert verdict.forgiven is False
    assert verdict.streak == 1


@pytest.mark.asyncio
async def test_a_good_cycle_clears_the_streak_and_stores_the_score() -> None:
    # Arrange
    redis = _redis()
    grace = SshUnreachableGrace(redis)

    # Act
    await grace.record_successful_cycle("exec-1", 3.25)

    # Assert
    redis.delete.assert_awaited_once()
    assert redis.delete.await_args.args[0].endswith(":ssh_unreachable_streak")
    stored_key, stored_value = redis.set.await_args.args[0], redis.set.await_args.args[1]
    assert stored_key.endswith(":last_score")
    assert stored_value == "3.25"


@pytest.mark.asyncio
async def test_a_zero_score_cycle_does_not_become_the_carried_score() -> None:
    # Arrange: carrying a 0 forward would forgive nothing and hide a real failure.
    redis = _redis()
    grace = SshUnreachableGrace(redis)

    # Act
    await grace.record_successful_cycle("exec-1", 0.0)

    # Assert
    redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_broken_redis_forgives_rather_than_punishes() -> None:
    # Arrange: our own outage must never cost the provider its score.
    redis = _redis()
    redis.get = AsyncMock(side_effect=RuntimeError("redis down"))
    grace = SshUnreachableGrace(redis)

    # Act
    verdict = await grace.score_for_unreachable_cycle("exec-1")

    # Assert
    assert verdict.forgiven is True
    assert verdict.streak == 1


def test_the_event_carries_a_reason_code_and_a_remedy() -> None:
    # Arrange / Act
    event = SshUnreachableGrace.build_event(
        executor_uuid="exec-1",
        host="1.2.3.4",
        port=2200,
        error="Not allowed at this time",
        streak=2,
        forgiven=False,
    )

    # Assert
    assert event.reason_code == SSH_UNREACHABLE_REASON_CODE
    assert event.remediation
    assert event.what_we_saw["ssh_port"] == 2200
    assert event.what_we_saw["consecutive_failures"] == 2


@pytest.mark.asyncio
async def test_the_service_forgives_the_first_unreachable_cycle_then_zeroes() -> None:
    """The except branch in TaskService.create_task must run the grace, not a plain zero."""
    import asyncssh
    from datura.requests.miner_requests import ExecutorSSHInfo
    from payload_models.payloads import MinerJobRequestPayload
    from services.task.service import TaskService, _is_ssh_transport_failure

    assert _is_ssh_transport_failure(asyncssh.Error(code=1, reason="refused")) is True
    assert _is_ssh_transport_failure(ValueError("a failed check")) is False

    service = TaskService.__new__(TaskService)
    redis = _redis()
    stored: dict[str, str] = {"executor:node-9:last_score": "2.5"}
    redis.get = AsyncMock(side_effect=lambda key: stored.get(key))
    redis.set = AsyncMock(side_effect=lambda key, value, **kw: stored.__setitem__(key, value))
    service.ssh_unreachable_grace = SshUnreachableGrace(redis)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(side_effect=asyncssh.Error(code=1, reason="refused"))

    executor = ExecutorSSHInfo(
        uuid="node-9",
        address="10.0.0.5",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1",
        miner_hotkey="5Miner",
        miner_coldkey="5Cold",
        miner_address="10.0.0.5",
        miner_port=8080,
        executors=[],
    )

    async def run_one_cycle():
        return await service.create_task(
            miner_info=miner,
            executor_info=executor,
            keypair=MagicMock(ss58_address="5Val"),
            private_key="key",
            public_key="pub",
            encrypted_files=MagicMock(),
            rented_data=MagicMock(),
            default_docker_image_digests={},
        )

    first = await run_one_cycle()
    assert first.score == 2.5, "the first unreachable cycle must keep the last good score"

    second = await run_one_cycle()
    assert second.score == 0, "the second unreachable cycle in a row must score zero"
    assert "EXECUTOR_SSH_UNREACHABLE" in second.log_text
