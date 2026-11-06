"""G18: a seal-aware verifier that gets no sealed blob fails a matching plaintext answer.

The plaintext is printed by the executor's own wrapper, so a matching answer is logged and failed.
A sealed answer, or a probe that never answered (out of memory), is judged as before.
"""

import json
import logging

from services import matrix_validation_service as mvs
from services.task.checks import capability
from services.task.checks.capability import Msg
from tests.test_local_verify import SPECS, matmul_service


def _unsealed_stdout(uuid: str) -> str:
    return (
        "UUID:  "
        + uuid
        + "\nRESULT_JSON: "
        + json.dumps({"uuid": uuid, "metrics": {"tflops": 9999.0}})
    )


def _sealed_stdout(uuid: str) -> str:
    return "RESULT_JSON: " + json.dumps(
        {"uuid": uuid, "metrics": {"tflops": 9999.0}, "sealed": "cafe"}
    )


def _unsealed_logs(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if "MATMUL_RESULT_UNSEALED" in record.getMessage()]


def test_unsealed_answer_is_logged_and_fails(monkeypatch, caplog):
    validation = matmul_service(monkeypatch)
    challenge = validation.prepare_matmul_challenge(SPECS, {}, vram_budget_mb=8192)

    with caplog.at_level(logging.WARNING, logger=mvs.logger.name):
        result = validation.evaluate_matmul_output(
            challenge, stdout=_unsealed_stdout(challenge.params.uuid)
        )
    challenge.close()

    assert result.success is False
    assert result.metrics is None
    assert len(_unsealed_logs(caplog)) == 1


def test_sealed_answer_passes_and_logs_nothing(monkeypatch, caplog):
    validation = matmul_service(monkeypatch)
    challenge = validation.prepare_matmul_challenge(SPECS, {}, vram_budget_mb=8192)

    with caplog.at_level(logging.WARNING, logger=mvs.logger.name):
        result = validation.evaluate_matmul_output(challenge, stdout=_sealed_stdout("ignored"))
    challenge.close()

    assert result.success is True
    assert result.metrics == {"tflops": 42.0}
    assert _unsealed_logs(caplog) == []


def test_out_of_memory_probe_keeps_the_busy_gpu_waiver(monkeypatch, caplog):
    # a probe that never allocated prints no uuid and `sealed: null`; it is not a skipped seal
    validation = matmul_service(monkeypatch)
    challenge = validation.prepare_matmul_challenge(SPECS, {}, vram_budget_mb=8192)
    stdout = "UUID:  None\nRESULT_JSON: " + json.dumps(
        {"uuid": "None", "metrics": None, "sealed": None}
    )

    with caplog.at_level(logging.WARNING, logger=mvs.logger.name):
        result = validation.evaluate_matmul_output(
            challenge, stdout=stdout, stderr="cudaMalloc failed: out of memory"
        )
    challenge.close()

    assert result.success is False
    assert result.error_message.startswith(mvs.UUID_MISMATCH_ERROR_PREFIX)
    assert capability._probe_gave_no_answer(result) is True
    assert capability._failure_template(result) is Msg.VERIFY_FAILED_VRAM_UNAVAILABLE
    assert _unsealed_logs(caplog) == []


def test_unsealed_wrong_uuid_is_still_a_mismatch(monkeypatch):
    validation = matmul_service(monkeypatch)
    challenge = validation.prepare_matmul_challenge(SPECS, {}, vram_budget_mb=8192)

    result = validation.evaluate_matmul_output(challenge, stdout=_unsealed_stdout("forged"))
    challenge.close()

    assert result.success is False
    assert result.error_message.startswith(mvs.UUID_MISMATCH_ERROR_PREFIX)
    assert capability._probe_gave_no_answer(result) is False
