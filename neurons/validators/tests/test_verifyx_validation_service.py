"""Service-level tests for VerifyXValidationService failure diagnostics paths.

Covers:
- The outer catch-all (failure before SSH) now emits a structured log line and populates
  `diagnostics` with `failure_class=UNKNOWN`.
- A non-zero exit with a non-empty stdout is classified `EXECUTOR_CRASH`, not
  `CIPHER_REJECTED`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurons.validators.src.services.verifyx_validation_service import (
    MIN_CIPHER_LEN,
    VerifyXFailureClass,
    VerifyXValidationService,
)


def _executor_info() -> SimpleNamespace:
    return SimpleNamespace(python_path="/usr/bin/python3", root_dir="/root/app", uuid="exec-1")


@pytest.mark.asyncio
async def test_outer_catch_all_populates_diagnostics_and_logs(caplog):
    service = VerifyXValidationService()

    # Force local sha256 lookup to blow up so the outer except catches it — this is
    # the exact pre-SSH failure surface the previous catch-all swallowed without a log.
    with patch(
        "neurons.validators.src.services.verifyx_validation_service.sha256_from_path",
        side_effect=RuntimeError("boom"),
    ):
        shell = MagicMock()
        shell.get_checksums_over_scp = AsyncMock(return_value="md5:sha256")

        with caplog.at_level(logging.ERROR):
            result = await service.validate_verifyx_and_process_job(
                shell=shell,
                executor_info=_executor_info(),
                default_extra={"executor_uuid": "exec-1", "miner_hotkey": "hk"},
                machine_spec={"gpu": {"count": 1, "details": [{"uuid": "u", "name": "H100"}]}},
            )

    assert result.error is not None
    assert result.diagnostics is not None
    assert result.diagnostics["failure_class"] == VerifyXFailureClass.UNKNOWN.value
    # Pre-SSH failures record the exception under `internal_error` — not `transport_error` —
    # so the classifier can't mislabel an internal exception as an SSH transport error.
    assert "RuntimeError: boom" in (result.diagnostics.get("internal_error") or "")
    # One structured ERROR log line.
    assert any(
        "VerifyX validation failed" in rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.ERROR
    )


@pytest.mark.asyncio
async def test_nonzero_exit_with_nonempty_stdout_classified_as_executor_crash(caplog):
    service = VerifyXValidationService()

    # Skip the checksum check (equal) and the ctypes-based challenge generation. We only
    # care about the SSH capture → classification path.
    with patch(
        "neurons.validators.src.services.verifyx_validation_service.sha256_from_path",
        return_value="s",
    ), patch(
        "neurons.validators.src.services.verifyx_validation_service.VerifyXValidator"
    ) as fake_validator_cls:
        fake_validator_cls.return_value.generate_challenge.return_value = "deadbeef"

        partial_stdout = "a" * (MIN_CIPHER_LEN + 20)
        ssh_result = SimpleNamespace(
            stdout=partial_stdout,
            stderr="Traceback: boom",
            exit_status=1,
        )
        shell = MagicMock()
        shell.get_sha256_checksum_by_path = AsyncMock(return_value="s")
        shell.get_checksums_over_scp = AsyncMock(return_value="md5:s")
        shell.ssh_client = MagicMock()
        shell.ssh_client.run = AsyncMock(return_value=ssh_result)

        with caplog.at_level(logging.ERROR):
            result = await service.validate_verifyx_and_process_job(
                shell=shell,
                executor_info=_executor_info(),
                default_extra={"executor_uuid": "exec-1", "miner_hotkey": "hk"},
                machine_spec={"gpu": {"count": 1, "details": [{"uuid": "u", "name": "H100"}]}},
            )

    assert result.diagnostics is not None
    assert result.diagnostics["failure_class"] == VerifyXFailureClass.EXECUTOR_CRASH.value
    assert result.diagnostics["exit_status"] == 1
    # stdout_len is reported against the raw stdout bytes/chars, not the stripped length.
    assert result.diagnostics["stdout_len"] == len(partial_stdout)
