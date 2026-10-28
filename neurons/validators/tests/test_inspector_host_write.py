"""DAH-4548: the validator reads the Inspector sensor's rental-interference verdict. A host write
into a rented pod fails the node; a read, a sensor without the verdict and a non-rented pod never do."""

import logging

import pytest
from helpers import FakeRedis, build_context_config, build_services, build_state
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationResponse,
)
from neurons.validators.src.services.task.checks import inspector
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from neurons.validators.src.services.task.inspector_verdict import host_write_findings
from test_inspector_check import DummyInspectorService, _healthy_report, _rented_data

POD = "pod_pod-1"
CTR_ID = "0338e0ffef4ba92700d9c838d14f35d24d68a9433d4a3583ae43fdbdcc1b23de"

HOST_WRITE = {
    "category": "RuntimeInterference",
    "kind": "DockerCp",
    "severity": "critical",
    "command": f"cp /usr/local/share/dmi-cloak-inner.sh {CTR_ID}:/tmp/.dmi_cloak.sh",
    "binary": "/usr/bin/docker",
    "pid": 404793,
    "container": POD,
    "parent": {"binary": "/usr/local/bin/dmi-cloak-daemon.sh", "pid": 404131},
    "host": True,
    "policy": "tamper-docker-cli",
    "tags": ["tamper:docker-cli", "severity:high"],
    "docker": {"resolved_name": POD, "state": "running"},
    "rental_interference": True,
    "direction": "write",
}
HOST_READ = {
    **HOST_WRITE,
    "command": f"cp {POD}:/root/data.bin /tmp/data.bin",
    "direction": "read",
}
OTHER_POD_WRITE = {
    **HOST_WRITE,
    "container": "pod_other",
    "docker": {"resolved_name": "pod_other"},
}


async def _run(context_factory, findings, **report):
    report = _healthy_report(findings=findings, **({"interference_rules_version": 1} | report))
    ctx = context_factory(
        config=build_context_config(inspector_enabled=True),
        services=build_services(
            inspector=DummyInspectorService(InspectorValidationResponse(report=report)),
            redis=FakeRedis(),
        ),
        state=build_state(rented_data=_rented_data("executor-123")),
    )
    return await InspectorRentedCheck().run(ctx)


def _logs(caplog, name):
    return [r.msg.extra for r in caplog.records if r.getMessage() == name]


@pytest.mark.asyncio
async def test_write_verdict_fails_the_node(context_factory, caplog):
    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        result = await _run(context_factory, [HOST_WRITE])

    assert result.passed is False
    assert result.updates["inspector_host_write_passed"] is False
    assert "node failed" in result.event.impact
    assert "score stays zero" in result.event.remediation
    logs = _logs(caplog, "INSPECTOR_HOST_WRITE")
    assert [log["enforced"] for log in logs] == [True]
    assert "dmi-cloak-inner.sh" in logs[0]["commands"][0]


@pytest.mark.asyncio
async def test_read_verdict_is_logged_never_enforced(context_factory, caplog):
    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        result = await _run(context_factory, [HOST_READ])

    assert result.passed is True
    assert "inspector_host_write_passed" not in result.updates
    logs = _logs(caplog, "INSPECTOR_HOST_READ")
    assert len(logs) == 1 and logs[0]["host_reads"] == 1 and logs[0]["enforced"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version", [{"interference_rules_version": None}, {"interference_rules_version": 0}]
)
async def test_sensor_without_the_verdict_does_not_fail(context_factory, caplog, version):
    with caplog.at_level(logging.INFO, logger=inspector.logger.name):
        result = await _run(context_factory, [HOST_WRITE], **version)

    assert result.passed is True
    assert "inspector_host_write_passed" not in result.updates
    assert len(_logs(caplog, "INSPECTOR_INTERFERENCE_UNSUPPORTED")) == 1


@pytest.mark.asyncio
async def test_write_into_a_pod_that_is_not_rented_is_ignored(context_factory, caplog):
    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        result = await _run(context_factory, [OTHER_POD_WRITE])

    assert result.passed is True
    assert _logs(caplog, "INSPECTOR_HOST_WRITE") == []


def test_host_write_findings_reads_only_the_sensor_verdict():
    not_interference = {**HOST_WRITE, "rental_interference": False}
    findings = [HOST_WRITE, HOST_READ, OTHER_POD_WRITE, not_interference]
    report = {"interference_rules_version": 1}

    assert host_write_findings(report, findings, ["pod-1"]) == ([HOST_WRITE], [HOST_READ])
    assert host_write_findings({}, findings, ["pod-1"]) is None
