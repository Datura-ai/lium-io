"""DAH-6379: a pod die right after an oom event is logged as oom_killed; Docker's OOMKilled is reset on restart."""

from unittest.mock import patch

import monitor


def _event(action, exit_code=None):
    attrs = {"name": "pod_1"}
    if exit_code is not None:
        attrs["exitCode"] = str(exit_code)
    return {"Type": "container", "Action": action, "id": "abc", "Actor": {"ID": "abc", "Attributes": attrs}}


def _die_reason(events):
    logged = []
    with patch.object(monitor, "log_event", logged.append), patch.object(monitor, "classify_stop", lambda c: "other"):
        for e in events:
            monitor.handle_event(e)
    return [p.reason for p in logged if p.event == "die"][0]


def test_a_die_after_an_oom_event_is_oom_killed():
    assert _die_reason([_event("oom"), _event("die", 137)]) == "oom_killed"


def test_a_die_without_an_oom_event_keeps_its_exit_code_reason():
    assert _die_reason([_event("die", 137)]) == "other"


def test_an_old_oom_event_does_not_label_a_later_die():
    with patch.object(monitor.time, "monotonic", side_effect=[0.0, 1000.0]):
        assert _die_reason([_event("oom"), _event("die", 137)]) == "other"
