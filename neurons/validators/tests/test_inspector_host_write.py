"""DAH-4548: a host process writing into a rented pod fails the node under
INSPECTOR_HOST_WRITE_ENFORCEMENT_ENABLED, and is only logged without it."""

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

# shaped like the 16 Oct 5E5GpE1b findings (validator Loki)
DMI_CLOAK_CP = {
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
}
ARCHIVE_PUT = {
    **DMI_CLOAK_CP,
    "kind": "DockerSocketWrite",
    "command": f"PUT /v1.56/containers/{CTR_ID}/archive?noOverwriteDirNonDir=true&path=%2Ftmp",
    "parent": {"binary": "/usr/bin/bash", "pid": 480545},
    "policy": "tamper-docker-sock-write",
}
EXECUTOR_RESTORE_CP = {
    **DMI_CLOAK_CP,
    "command": f"cp /tmp/restore.tar {POD}:/root/restore.tar",
    "parent": {"binary": "/usr/bin/sh", "pid": 7},
    "host": False,
    "tags": ["tamper:docker-cli", "nested_from:executor-executor-1"],
}
COPY_OUT = {**DMI_CLOAK_CP, "command": f"cp {POD}:/root/data.bin /tmp/data.bin"}
VOLUME = "/var/lib/docker/plugins/9f1c/propagated-mount/volume_pod-1"


def _overlay_write(path):
    # the writer the 16 Oct copy-ups carried: the host's init, parent <kernel>
    return {
        "category": "RuntimeInterference",
        "kind": "OverlayFsWrite",
        "severity": "critical",
        "command": path,
        "binary": "/usr/bin/bash",
        "pid": 1,
        "cwd": "/root/executor",
        "container": None,
        "parent": {"binary": "<kernel>", "pid": 0},
        "host": True,
        "policy": "tamper-fs",
        "tags": ["tamper:fs", "rental_volume"],
    }


async def _run(context_factory, findings):
    report = _healthy_report(findings=findings)
    ctx = context_factory(
        config=build_context_config(inspector_enabled=True),
        services=build_services(
            inspector=DummyInspectorService(InspectorValidationResponse(report=report)),
            redis=FakeRedis(),
        ),
        state=build_state(rented_data=_rented_data("executor-123")),
    )
    return await InspectorRentedCheck().run(ctx)


def _host_write_logs(caplog):
    return [r.msg.extra for r in caplog.records if r.getMessage() == "INSPECTOR_HOST_WRITE"]


@pytest.mark.asyncio
@pytest.mark.parametrize("finding", [DMI_CLOAK_CP, ARCHIVE_PUT])
async def test_host_copy_into_a_rented_pod_fails_the_node_when_enforced(context_factory, monkeypatch, finding):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_HOST_WRITE_ENFORCEMENT_ENABLED", True)

    result = await _run(context_factory, [finding])

    assert result.passed is False
    assert result.updates["inspector_host_write_passed"] is False
    assert "node failed" in result.event.impact


@pytest.mark.asyncio
async def test_host_copy_with_the_flag_off_is_only_logged(context_factory, caplog):
    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        result = await _run(context_factory, [DMI_CLOAK_CP])

    assert result.passed is True
    assert "inspector_host_write_passed" not in result.updates
    logs = _host_write_logs(caplog)
    assert len(logs) == 1 and logs[0]["enforced"] is False
    assert "dmi-cloak-inner.sh" in logs[0]["commands"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finding",
    [
        EXECUTOR_RESTORE_CP,
        COPY_OUT,
        _overlay_write(f"{VOLUME}/.bashrc"),
        _overlay_write(f"{VOLUME}/.profile"),
        _overlay_write("/var/lib/docker/volumes/volume_pod-1/_data/.bashrc"),
        {**_overlay_write(f"{VOLUME}/.profile"), "binary": "", "pid": 16377, "parent": None},
    ],
)
async def test_platform_copy_copy_out_and_volume_copy_up_pass(context_factory, monkeypatch, caplog, finding):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_HOST_WRITE_ENFORCEMENT_ENABLED", True)

    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        result = await _run(context_factory, [finding])

    assert result.passed is True
    assert "inspector_host_write_passed" not in result.updates
    assert _host_write_logs(caplog) == []


def test_other_volume_writes_and_memory_writes_count():
    findings = [
        _overlay_write(f"{VOLUME}/workspace/.bashrc"),
        _overlay_write(f"{VOLUME}/train.py"),
        # a host shell editing the renter's .bashrc is not Docker's copy-up
        {
            **_overlay_write(f"{VOLUME}/.bashrc"),
            "pid": 4242,
            "parent": {"binary": "/usr/bin/bash", "pid": 4200},
        },
        # an empty binary is only the unresolved copy-up when the parent is absent too
        {
            **_overlay_write(f"{VOLUME}/.profile"),
            "binary": "",
            "pid": 4243,
            "parent": {"binary": "/usr/bin/bash", "pid": 4200},
        },
        {**DMI_CLOAK_CP, "kind": "ProcFsWrite", "command": "procfs mem"},
        {**DMI_CLOAK_CP, "kind": "ProcessMemoryWrite", "command": "process_vm_writev"},
        # a write into a pod that is not rented here is not this rental's
        {**DMI_CLOAK_CP, "container": "pod_other", "docker": {"resolved_name": "pod_other"}},
    ]
    assert host_write_findings(findings, ["pod-1"]) == findings[:6]


# shaped like provider 5ED8Byct's host bash loop (fleet sweep, 15,817 findings since 8 Oct)
HOST_DOCKER_UPDATE = {
    **ARCHIVE_PUT,
    "kind": "DockerUpdate",
    "command": f"POST /v1.56/containers/{CTR_ID}/update",
}
EXECUTOR_DOCKER_UPDATE = {
    **HOST_DOCKER_UPDATE,
    "host": False,
    "tags": ["tamper:docker-sock-write", "nested_from:executor-executor-1"],
}


@pytest.mark.asyncio
async def test_host_docker_update_of_a_rented_pod_fails_when_on_logs_when_off(context_factory, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        off = await _run(context_factory, [HOST_DOCKER_UPDATE])
    assert off.passed is True and "inspector_host_write_passed" not in off.updates
    assert [log["enforced"] for log in _host_write_logs(caplog)] == [False]

    monkeypatch.setattr(inspector.settings, "INSPECTOR_HOST_WRITE_ENFORCEMENT_ENABLED", True)
    on = await _run(context_factory, [HOST_DOCKER_UPDATE])
    assert on.passed is False and on.updates["inspector_host_write_passed"] is False

    executor = await _run(context_factory, [EXECUTOR_DOCKER_UPDATE])
    assert executor.passed is True and "inspector_host_write_passed" not in executor.updates
