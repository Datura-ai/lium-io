from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationResponse,
)
from neurons.validators.src.services.redis_service import STREAMING_LOG_CHANNEL
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from neurons.validators.src.services.task.inspector_verdict import (
    ACTION_NONE,
    ACTION_QUARANTINE,
    SENSOR_ATTESTED,
    SENSOR_UNATTESTED,
    build_verdict,
    canonical_sha256,
    exec_payload,
    is_platform_origin,
)
from neurons.validators.src.services.task.messages import InspectorMessages as Msg
from neurons.validators.src.services.task.score_calculator import calculate_scores
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from core.config import settings

POD = "0f6a1c2e-1111-4222-8333-444455556666"


def _finding(command: str, *, host: bool = False, kind: str = "DockerExec", nested: bool = True) -> dict:
    return {
        "category": "runtime_interference",
        "kind": kind,
        "severity": "high",
        "time": "2026-09-08T00:11:00Z",
        "command": command,
        "binary": "/usr/bin/docker",
        "pid": 4242,
        "cwd": "/root",
        "container": f"pod_{POD}",
        "parent": {"pid": 4000, "binary": "/bin/bash"},
        "host": host,
        "policy": "tamper-docker-cli",
        "tags": ["tamper:docker-cli", "severity:high"] + (["nested_from:executor-executor-1"] if nested else []),
        "details": {},
    }


VALIDATOR_LIVENESS = f'/usr/bin/docker exec -u 0 -i pod_{POD} sh -c "cat /root/.ssh/authorized_keys"'
EXECUTOR_DF = f"/usr/bin/docker exec -u root pod_{POD} df -k /lium-cipher"
EXECUTOR_TAR = f"/usr/bin/docker exec -u 0 -i pod_{POD} tar --xattrs --acls -xzpf - -C /root/restored"
HUMAN_SHELL = f"/usr/bin/docker exec -it pod_{POD} bash"
HUMAN_KEY_READ = f"/usr/bin/docker exec -u 0 -i pod_{POD} sh -c 'cat /root/.ssh/id_ed25519'"
CHAINED = f"/usr/bin/docker exec -u 0 -i pod_{POD} sh -c 'cat /root/.ssh/authorized_keys; cat /root/.ssh/id_rsa'"


@pytest.mark.parametrize(
    ("command", "payload"),
    [
        (VALIDATOR_LIVENESS, "cat /root/.ssh/authorized_keys"),
        (f"/usr/bin/docker exec -u 0 -i pod_{POD} sh -c cat /root/.ssh/authorized_keys", "cat /root/.ssh/authorized_keys"),
        (EXECUTOR_DF, "df -k /lium-cipher"),
        (EXECUTOR_TAR, "tar --xattrs --acls -xzpf - -C /root/restored"),
        (HUMAN_SHELL, "bash"),
        (f"/usr/bin/docker exec --user=root pod_{POD} df -k /", "df -k /"),
        ("/usr/bin/docker ps", None),
        (f"/usr/bin/docker exec -u 0 -i executor-executor-1 sh -c 'cat x'", None),
    ],
)
def test_exec_payload_strips_options_and_shell_wrapper(command, payload):
    assert exec_payload(command) == payload


def test_platform_origin_needs_both_the_executor_source_and_a_known_payload():
    assert is_platform_origin(_finding(VALIDATOR_LIVENESS))
    assert is_platform_origin(_finding(EXECUTOR_DF))
    assert is_platform_origin(_finding(EXECUTOR_TAR))
    # the same command from the host is a human at the keyboard
    assert not is_platform_origin(_finding(VALIDATOR_LIVENESS, host=True, nested=False))
    # a chain that starts with our command is not our command
    assert not is_platform_origin(_finding(CHAINED))
    assert not is_platform_origin(_finding(HUMAN_KEY_READ))
    assert not is_platform_origin(_finding(HUMAN_SHELL))
    # only execs can be ours; an nsenter never is
    assert not is_platform_origin(_finding(VALIDATOR_LIVENESS, kind="NamespaceEnter"))


def test_verdict_hashes_only_the_provider_findings_and_names_the_pod():
    report = {"findings": [_finding(VALIDATOR_LIVENESS), _finding(HUMAN_KEY_READ, host=True, nested=False)]}
    verdict = build_verdict(
        report,
        report["findings"],
        rented_pod_ids=[POD, "other-pod"],
        sensor_attested=False,
        enforce=False,
    )

    assert len(verdict.platform_findings) == 1
    assert len(verdict.provider_findings) == 1
    assert verdict.evidence == [canonical_sha256(report["findings"][1])]
    assert verdict.report_sha256 == canonical_sha256(report)
    assert verdict.affected_pod_ids == [POD]
    assert verdict.classes == ["DockerExec"]
    assert verdict.sensor == SENSOR_UNATTESTED
    assert verdict.action == ACTION_NONE
    assert verdict.as_payload()["ban_source"] is None


def test_verdict_without_a_named_pod_tells_every_renter_on_the_host():
    finding = _finding(HUMAN_SHELL, host=True, nested=False)
    finding["container"] = None
    verdict = build_verdict({}, [finding], rented_pod_ids=["b", "a"], sensor_attested=True, enforce=True)

    assert verdict.affected_pod_ids == ["a", "b"]
    assert verdict.sensor == SENSOR_ATTESTED
    assert verdict.action == ACTION_QUARANTINE
    assert verdict.as_payload()["ban_source"] == "inspector_auto"


# --- the check -----------------------------------------------------------------------------


class DummyInspectorService:
    def __init__(self, response: InspectorValidationResponse) -> None:
        self.response = response
        self.sensor_attested = None

    async def validate_rented_executor(self, shell, ssh, executor, default_extra, *, sensor_attested=False):
        self.sensor_attested = sensor_attested
        return self.response


def _rented(executor_uuid: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey="miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id=POD, container_name=f"pod_{POD}")],
            )
        },
        banned_guids=[],
    )


def _ctx(context_factory, findings: list[dict], *, redis=None, **overrides):
    service = DummyInspectorService(
        InspectorValidationResponse(
            report={
                "canary_ok": True,
                "health": {"collector_started_unix": 1},
                "findings": findings,
                "summary": {"findings": len(findings)},
            },
            diagnostics={"sensor_integrity": "shell_sha256_unattested"},
        )
    )
    ctx = context_factory(
        config=build_context_config(inspector_enabled=True),
        services=build_services(inspector=service, redis=redis),
        state=build_state(rented_data=_rented("executor-123")),
        **overrides,
    )
    return ctx, service


@pytest.mark.asyncio
async def test_platform_only_findings_are_clean_and_no_renter_is_told(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "INSPECTOR_ENFORCE_ENABLED", True)
    redis = AsyncMock()
    ctx, _ = _ctx(context_factory, [_finding(VALIDATOR_LIVENESS), _finding(EXECUTOR_DF)], redis=redis)

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PLATFORM_ORIGIN_ONLY.reason
    assert result.event.severity == "info"
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "CLEAN"
    assert event["context"]["verdict"]["platform_findings"] == 2
    assert event["context"]["verdict"]["provider_findings"] == 0
    assert "inspector_passed" not in result.updates
    redis.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_finding_in_shadow_records_the_verdict_and_tells_the_renter(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "INSPECTOR_ENFORCE_ENABLED", False)
    redis = AsyncMock()
    provider = _finding(HUMAN_KEY_READ, host=True, nested=False)
    ctx, _ = _ctx(context_factory, [_finding(VALIDATOR_LIVENESS), provider], redis=redis)

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.MALICIOUS_FINDINGS.reason
    assert result.event.severity == "warning"
    assert result.event.what_we_saw["findings"] == [provider]
    assert result.event.what_we_saw["platform_findings"] == 1
    assert "inspector_passed" not in result.updates
    verdict = result.updates["state"].inspector_event["context"]["verdict"]
    assert verdict["evidence_sha256"] == [canonical_sha256(provider)]
    assert verdict["action"] == ACTION_NONE
    assert verdict["enforce"] is False
    assert verdict["sensor"] == SENSOR_UNATTESTED
    assert verdict["affected_pod_ids"] == [POD]
    # the sensor's own diagnostics stay next to the verdict
    assert result.updates["state"].inspector_event["context"]["sensor_integrity"] == "shell_sha256_unattested"

    redis.publish.assert_awaited_once()
    channel, message = redis.publish.await_args.args
    assert channel == STREAMING_LOG_CHANNEL
    assert message["pod_id"] == POD
    assert message["executor_uuid"] == ctx.executor.uuid
    (log,) = message["logs"]
    assert log["log_tag"] == "provider_access_detected"
    assert log["log_status"] == "error"
    assert log["evidence_sha256"] == verdict["evidence_sha256"]
    assert "removed the host" not in log["log_text"]


@pytest.mark.asyncio
async def test_provider_finding_under_enforcement_fails_the_check_and_requests_quarantine(
    context_factory, monkeypatch
):
    monkeypatch.setattr(settings, "INSPECTOR_ENFORCE_ENABLED", True)
    redis = AsyncMock()
    ctx, _ = _ctx(context_factory, [_finding(HUMAN_SHELL, host=True, nested=False)], redis=redis)

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is False
    assert result.halt is False  # non-fatal: the cycle finishes, the score gate does the rest
    assert result.event.severity == "error"
    assert result.updates["inspector_passed"] is False
    verdict = result.updates["state"].inspector_event["context"]["verdict"]
    assert verdict["action"] == ACTION_QUARANTINE
    assert verdict["ban_source"] == "inspector_auto"
    (log,) = redis.publish.await_args.args[1]["logs"]
    assert "removed the host from the marketplace" in log["log_text"]


@pytest.mark.asyncio
async def test_tdx_attested_host_marks_the_sensor_attested_and_skips_the_shell_checksum(context_factory):
    ctx, service = _ctx(context_factory, [], tdx_attestation_passed=True)

    result = await InspectorRentedCheck().run(ctx)

    assert service.sensor_attested is True
    assert result.updates["state"].inspector_event["context"]["verdict"]["sensor"] == SENSOR_ATTESTED


@pytest.mark.asyncio
async def test_renter_event_failure_does_not_lose_the_verdict(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "INSPECTOR_ENFORCE_ENABLED", False)
    redis = AsyncMock()
    redis.publish.side_effect = ConnectionError("redis down")
    ctx, _ = _ctx(context_factory, [_finding(HUMAN_SHELL, host=True, nested=False)], redis=redis)

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.MALICIOUS_FINDINGS.reason
    assert result.updates["state"].inspector_event["context"]["verdict"]["provider_findings"] == 1


def test_score_gate_zeroes_on_a_failed_inspector_verdict():
    def ctx(inspector_passed: bool):
        return SimpleNamespace(
            state=SimpleNamespace(gpu_model="", specs={"network": {"ema_verifyx_download_speed": 500.0}}),
            collateral_deposited=True,
            collateral_error_message=None,
            contract_version=None,
            executor=SimpleNamespace(price_per_gpu=None, tdx_quote=None),
            tdx_attestation_passed=False,
            cpu_truth_passed=True,
            provider_side_load_passed=True,
            inspector_passed=inspector_passed,
        )

    actual, job, warning = calculate_scores(ctx(False), rented=False)
    assert (actual, job) == (0.0, 0.0)
    assert "Inspector" in warning

    actual, _, _ = calculate_scores(ctx(True), rented=False)
    assert actual > 0.0
