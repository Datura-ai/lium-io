from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationResponse,
)
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from neurons.validators.src.services.task.inspector_verdict import (
    build_verdict,
    is_platform_origin,
)
from neurons.validators.src.services.task.score_calculator import calculate_scores
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)


POD = "0f6a1c2e-1111-4222-8333-444455556666"


def _finding(
    command: str, *, host: bool = False, kind: str = "DockerExec", nested: bool = True, nested_from: str = "executor-executor-1"
) -> dict:
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
        "tags": ["tamper:docker-cli", "severity:high"] + ([f"nested_from:{nested_from}"] if nested else []),
        "details": {},
    }


VALIDATOR_LIVENESS = f'/usr/bin/docker exec -u 0 -i pod_{POD} sh -c "cat /root/.ssh/authorized_keys"'
HUMAN_SHELL = f"/usr/bin/docker exec -it pod_{POD} bash"


def test_a_container_named_like_the_storage_helper_cannot_borrow_the_platforms_origin():
    # taiberium (11 Sep) asked to exempt the encrypted backup's nsenter (restic.py runs it from a
    # `lium-storage-<id>` helper). The sensor already judges that `nsenter -t <pid> -U` benign by
    # nstype (CLONE_NEWUSER only, nsenter.rs) and the unencrypted flow's volume bind as a runtime
    # mount (mount.rs), so neither reaches the verdict; a NamespaceEnter or DockerVolumeMount that
    # does arrive is real and a helper-shaped container name must not turn it into the platform's
    nsenter = "nsenter -t 4242 -m -p -- /bin/sh"
    mount = f"/var/lib/docker/volumes/volume_{POD}/_data"
    for kind, command in (("NamespaceEnter", nsenter), ("DockerVolumeMount", mount)):
        assert not is_platform_origin(_finding(command, kind=kind, nested_from="lium-storage-0123456789ab"))
        assert not is_platform_origin(_finding(command, kind=kind))  # the executor container itself
        assert not is_platform_origin(_finding(command, kind=kind, nested=False))
    # the shadow digest still sees what the platform did produce, by kind
    verdict = build_verdict(
        {}, [_finding(VALIDATOR_LIVENESS), _finding(f"docker rm -f pod_{POD}", kind="DockerRm"), _finding(VALIDATOR_LIVENESS)],
        rented_pod_ids=[POD], sensor_attested=False,
    )
    assert verdict.provider_findings == []
    assert verdict.as_payload().platform_kind_counts == {"DockerExec": 2, "DockerRm": 1}


def test_a_pod_outside_the_rented_list_is_recorded_but_no_renter_is_told():
    finding = _finding(HUMAN_SHELL, host=True, nested=False)
    finding["container"] = "pod_gone-since-the-list-was-fetched"
    verdict = build_verdict({}, [finding], rented_pod_ids=[POD], sensor_attested=False)

    assert verdict.affected_pod_ids == []
    assert verdict.unmatched_containers == ["pod_gone-since-the-list-was-fetched"]
    assert verdict.unmatched_containers_count == 1
    assert verdict.as_payload().unmatched_containers == ["pod_gone-since-the-list-was-fetched"]
    assert len(verdict.provider_findings) == 1


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
                "summary": {"findings": len(findings) if isinstance(findings, list) else 0},
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


@pytest.mark.parametrize("rented", [True, False])
@pytest.mark.asyncio
async def test_a_malicious_finding_on_a_rented_pod_changes_no_score_and_triggers_no_enforcement(context_factory, rented):
    """No enforcement on the validator side and no scoring based on Inspector findings: the context
    the MALICIOUS run hands on scores exactly like the CLEAN run's, the check passes and nothing is
    published to a renter's pod stream."""

    async def run(findings):
        redis = AsyncMock()
        ctx, _ = _ctx(context_factory, findings, redis=redis, collateral_deposited=True)
        ctx = ctx.model_copy(
            update={
                # a context every other gate lets through, so a zeroing gate cannot hide in 0 == 0
                "state": replace(ctx.state, specs={"network": {"ema_verifyx_download_speed": 500.0}}),
                "executor": ctx.executor.model_copy(update={"price_per_gpu": None}),
            }
        )
        result = await InspectorRentedCheck().run(ctx)
        return result, ctx.model_copy(update=result.updates), redis

    malicious, after_malicious, redis = await run([_finding(HUMAN_SHELL, host=True, nested=False)])
    clean, after_clean, _ = await run([])

    assert malicious.updates["state"].inspector_event["outcome"] == "MALICIOUS"
    assert malicious.updates["state"].inspector_event["context"]["verdict"]["affected_pod_ids"] == [POD]
    assert clean.updates["state"].inspector_event["outcome"] == "CLEAN"
    assert malicious.passed is True and malicious.halt is False
    assert malicious.event.severity == "warning"
    assert set(malicious.updates) == set(clean.updates) == {"default_extra", "state"}
    score = calculate_scores(after_malicious, rented=rented)
    assert score == calculate_scores(after_clean, rented=rented)
    assert score == (1.0, 1.0, "")
    redis.publish.assert_not_awaited()


