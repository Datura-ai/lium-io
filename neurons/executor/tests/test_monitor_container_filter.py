"""DAH-2395: the container monitor must watch the containers the validator actually creates.

The monitor recorded only the legacy "container_" prefix, which today matches
nothing but the validator's short-lived port-check probes — every rental pod
(pod_*) and idle-node filler (filler_*) death went unrecorded (verified on the
staging executor: 32k podlog rows, zero pod_/filler_ entries). These tests pin
the corrected filter.
"""

from unittest.mock import patch

import monitor


def _docker_event(name: str, action: str, exit_code: int | None = None) -> dict:
    attributes: dict = {"name": name}
    if exit_code is not None:
        attributes["exitCode"] = str(exit_code)
    return {
        "Type": "container",
        "Action": action,
        "id": "abc123",
        "Actor": {"ID": "abc123", "Attributes": attributes},
    }


def test_monitor_records_filler_container_death():
    with patch.object(monitor, "log_event") as log_event:
        monitor.handle_event(_docker_event("filler_63409a62-e024-414d-99aa-ba6f30be1bfd", "die", 137))

    log_event.assert_called_once()
    logged = log_event.call_args.args[0]
    assert logged.container_name == "filler_63409a62-e024-414d-99aa-ba6f30be1bfd"
    assert logged.event == "die"
    assert logged.exit_code == 137


def test_monitor_records_rental_pod_events():
    with patch.object(monitor, "log_event") as log_event:
        monitor.handle_event(_docker_event("pod_9b522424-5275-46ed-b283-7afd9f025b8d", "oom"))

    log_event.assert_called_once()
    logged = log_event.call_args.args[0]
    assert logged.container_name == "pod_9b522424-5275-46ed-b283-7afd9f025b8d"
    assert logged.event == "oom"


def test_monitor_ignores_port_check_probe_containers():
    probe_name = "container_5DfbmGgkgYoYKJk7rA9UFgeJFacfkBMhaeFZgJk7c4mCv7DT_40000"
    with patch.object(monitor, "log_event") as log_event:
        monitor.handle_event(_docker_event(probe_name, "die", 137))

    log_event.assert_not_called()


def test_monitor_ignores_infra_containers():
    with patch.object(monitor, "log_event") as log_event:
        monitor.handle_event(_docker_event("executor-db-1", "die", 1))

    log_event.assert_not_called()
