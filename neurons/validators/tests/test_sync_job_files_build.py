"""The cycle's job files are built before anything counts the cycle, and a lost forced-request clear does not stop it."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest_plugins = ["fixtures.incentive_fixtures"]


def _digests():
    return (
        patch("core.validator.fetch_default_image_digests", new=AsyncMock(return_value={})),
        patch("core.validator.fetch_executor_image_digest", new=AsyncMock(return_value=None)),
    )


@pytest.mark.asyncio
async def test_a_failed_job_files_build_records_no_rollout_attempt(validator_with_mocks):
    validator = validator_with_mocks
    validator.file_encrypt_service.ecrypt_miner_job_files.side_effect = RuntimeError("obfuscator exit 1")
    validator.observe_executor_rollout = AsyncMock(return_value=None)
    first, second = _digests()
    with first, second:
        await validator.sync()

    validator.observe_executor_rollout.assert_not_awaited()
    assert validator.last_job_run_blocks == 0


@pytest.mark.asyncio
async def test_a_failed_forced_request_clear_still_publishes_the_new_job_files(validator_with_mocks):
    validator = validator_with_mocks
    validator.file_encrypt_service.ecrypt_miner_job_files.return_value = {"dir": "new"}
    validator.an_operator_asked_for_a_cycle_now = AsyncMock(return_value=True)
    validator.redis_service.clear_forced_validation_cycle_request = AsyncMock(side_effect=ConnectionError("reply lost"))
    first, second = _digests()
    with first, second:
        await validator.sync()

    assert validator.cycle_inputs.encrypted_files == {"dir": "new"}


@pytest.mark.asyncio
async def test_the_build_keeps_the_published_job_files_while_no_express_verification_holds_them(validator_with_mocks):
    validator = validator_with_mocks
    validator.cycle_inputs = MagicMock()
    validator.cycle_inputs.encrypted_files.tmp_directory = "/jobs/previous"
    validator.express_lane.directories_in_use = MagicMock(return_value=set())
    validator.observe_executor_rollout = AsyncMock(return_value=None)
    first, second = _digests()
    with first, second:
        await validator.sync()

    kept = validator.file_encrypt_service.ecrypt_miner_job_files.call_args.kwargs["keep_directories"]
    assert "/jobs/previous" in kept
