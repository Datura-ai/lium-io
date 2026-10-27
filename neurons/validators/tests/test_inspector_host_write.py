"""DAH-4548: the validator reads the Inspector sensor's rental-interference verdict. A host write
into a rented pod fails the node; a read, a sensor without the verdict and a non-rented pod never do."""

import logging
from datetime import datetime, timedelta

import pytest
from helpers import FakeRedis, build_context_config, build_services, build_state
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationResponse,
)
from neurons.validators.src.services.task.checks import inspector
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from neurons.validators.src.services.task.inspector_verdict import (
    host_write_findings,
    is_volume_copy_up,
)
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


async def _run(context_factory, findings, pod_created_at=None, **report):
    report = _healthy_report(findings=findings, **({"interference_rules_version": 1} | report))
    rented_data = _rented_data("executor-123")
    for pod in rented_data.executors["executor-123"].pods:
        pod.created_at = pod_created_at
    ctx = context_factory(
        config=build_context_config(inspector_enabled=True),
        services=build_services(
            inspector=DummyInspectorService(InspectorValidationResponse(report=report)),
            redis=FakeRedis(),
        ),
        state=build_state(rented_data=rented_data),
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


# DAH-5234: the live shape of Docker's copy of the image home into a new pod volume
CREATED = datetime(2026, 10, 26, 14, 41, 0)  # backend pod created_at, naive UTC
VOLUME = "/var/lib/docker/plugins/20f9d24d/propagated-mount/volume_pod-1"
COPY_UP = {
    "category": "RuntimeInterference",
    "kind": "OverlayFsWrite",
    "severity": "high",
    "time": "2026-10-26T14:41:05Z",
    "command": f"{VOLUME}/.bashrc",
    "binary": "",
    "pid": 0,
    "container": None,
    "parent": {"pid": None, "binary": ""},
    "host": True,
    "policy": "tamper-fs",
    "tags": ["rental_volume"],
    "rental_interference": True,
    "direction": "write",
}


def _at(seconds):
    return (CREATED + timedelta(seconds=seconds)).isoformat() + "Z"


@pytest.mark.parametrize(
    ("change", "exempt"),
    [
        ({}, True),  # copy-up at +5 s
        ({"command": f"{VOLUME}/.profile"}, True),
        ({"parent": None}, True),
        ({"command": "/var/lib/docker/volumes/volume_pod-1/_data/.bashrc"}, True),
        ({"time": _at(-3)}, True),  # host clock a little behind the backend
        ({"time": _at(600)}, False),  # the same file 10 min later is a host write
        ({"time": "not a time"}, False),
        ({"binary": "/usr/bin/bash", "parent": {"pid": 1200, "binary": "/usr/bin/sshd"}}, False),
        ({"parent": {"pid": 1200, "binary": ""}}, False),
        ({"command": f"{VOLUME}/.ssh/authorized_keys"}, False),
        ({"command": f"{VOLUME}/root/.bashrc"}, False),  # nested
        ({"kind": "OverlayFsRead"}, False),
    ],
)
def test_volume_copy_up_is_exempt_only_in_its_exact_shape(change, exempt):
    assert is_volume_copy_up({**COPY_UP, **change}, CREATED) is exempt


def test_volume_copy_up_needs_the_pods_creation_time():
    assert is_volume_copy_up(COPY_UP, None) is False
    report = {"interference_rules_version": 1}
    assert host_write_findings(report, [COPY_UP], ["pod-1"]) == ([COPY_UP], [])
    assert host_write_findings(report, [COPY_UP], ["pod-1"], {"pod-1": CREATED}) == ([], [])
    # a pod_ container named by the finding never lends its time to another volume's path
    other_volume = "/var/lib/docker/volumes/volume_x/_data/.bashrc"
    named = {**COPY_UP, "container": "pod_pod-1", "command": other_volume}
    assert host_write_findings(report, [named], ["pod-1"], {"pod-1": CREATED}) == ([named], [])


@pytest.mark.asyncio
async def test_copy_up_into_a_new_pod_volume_does_not_fail_the_node(context_factory, caplog):
    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        result = await _run(context_factory, [COPY_UP], pod_created_at=CREATED)
        later = await _run(context_factory, [{**COPY_UP, "time": _at(600)}], pod_created_at=CREATED)

    assert result.passed is True
    assert "inspector_host_write_passed" not in result.updates
    assert later.passed is False
    assert [log["enforced"] for log in _logs(caplog, "INSPECTOR_HOST_WRITE")] == [True]


class _InspectSsh:
    def __init__(self, stdout="", error=None):
        self.stdout, self.error, self.commands = stdout, error, []

    async def run(self, command, timeout=None):
        self.commands.append(command)
        if self.error:
            raise self.error
        return type("Res", (), {"stdout": self.stdout, "exit_status": 0})()


@pytest.mark.parametrize(
    ("ssh", "passed"),
    [
        # a 3-minute pull: Docker created the container, and copied the volume, at +180 s
        (_InspectSsh('{"name":"/pod_pod-1","created":"2026-10-26T14:44:00.123456789Z"}\n'), True),
        (None, False),  # no container read: the pod's created_at is the anchor, +185 s fails
        (_InspectSsh(error=OSError("ssh down")), False),
        (_InspectSsh('{"name":"/pod_other","created":"2026-10-26T14:44:00Z"}\n'), False),
    ],
)
@pytest.mark.asyncio
async def test_copy_up_after_a_slow_pull_is_anchored_on_the_container(context_factory, ssh, passed):
    report = _healthy_report(findings=[{**COPY_UP, "time": _at(185)}], interference_rules_version=1)
    rented_data = _rented_data("executor-123")
    for pod in rented_data.executors["executor-123"].pods:
        pod.created_at = CREATED
    ctx = context_factory(
        config=build_context_config(inspector_enabled=True),
        services=build_services(
            inspector=DummyInspectorService(InspectorValidationResponse(report=report)),
            redis=FakeRedis(),
        ),
        state=build_state(rented_data=rented_data),
        ssh=ssh,
    )
    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is passed
    if ssh is not None:
        assert "docker inspect" in ssh.commands[-1] and "pod_pod-1" in ssh.commands[-1]
