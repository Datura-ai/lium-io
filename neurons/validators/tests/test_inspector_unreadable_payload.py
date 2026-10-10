"""Two behaviours of the validator's read of the inspector executor's stdout.

asyncssh's ``SSHReader.readline()`` returns a PARTIAL line when one line is longer than the channel
receive window (2 MiB by default), and on 20 Sep 2026 that left the integrity check unrun on every
host whose response crossed it. A line is now reassembled across partial reads; a payload that still
does not parse is recorded as ``INSPECTOR_UNREADABLE`` with typed diagnostics (the executor id, the
byte length, the first 200 chars repr-escaped, the decoder's position) and the other executors are
unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from neurons.validators.src.services import inspector_validation_service as ivs
from neurons.validators.src.services.task.messages import InspectorMessages as Msg
from test_inspector_validation_service import (  # noqa: F401 — the autouse fixtures ride along
    FakeShell,
    FakeSSH,
    FakeStderr,
    FakeStdout,
    FakeValidator,
    _interactive_stdout,
    enable_collector_ensure,
    fake_validator,
    matching_lib_checksums,
)


def _executor(uuid: str = "exec-1") -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid, python_path="/usr/bin/python3", root_dir="/root/app")


# --- the shape found on 252f0496: one response line longer than the SSH receive window ---------


# --- the other ways a stdout line stops being one JSON document ---------------------------------


# --- what the run does with it ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreadable_node_is_an_inspector_unreadable_event_not_a_pass(context_factory):
    # the check is non-fatal and the run goes on; the event carries the distinct reason so the
    # node counts as unreadable — not as INSPECTOR_VALIDATION_ERROR and not as clean
    from helpers import build_context_config, build_services, build_state
    from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
    from test_inspector_check import DummyInspectorService, _rented_data

    partial = '{"ok": true, "result": "' + "A" * 40
    service = DummyInspectorService(
        ivs.InspectorValidationResponse(
            error="inspector executor wrote an unreadable response to 'cmd': Unterminated string",
            diagnostics={
                "executor_uuid": "executor-123",
                "reason": Msg.UNREADABLE.reason,
                "error_type": "InspectorUnreadableError",
                "payload_cmd": "cmd",
                "payload_bytes": len(partial),
                "payload_head": repr(partial[:200]),
                "payload_terminated": False,
                "json_error": "Unterminated string starting at",
                "json_error_pos": 23,
                "json_error_lineno": 1,
                "json_error_colno": 24,
            },
            message=Msg.UNREADABLE,
        )
    )
    ctx = context_factory(
        config=build_context_config(inspector_enabled=True),
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.halt is False
    assert result.event.reason_code == "INSPECTOR_UNREADABLE"
    assert result.event.severity == "warning"
    assert result.event.what_we_saw["payload_bytes"] == len(partial)
    assert result.event.what_we_saw["json_error_colno"] == 24
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "ERROR"
    assert event["reason_code"] == "INSPECTOR_UNREADABLE"
    assert event["report"] is None
    assert event["error"]["payload_head"] == repr(partial[:200])
    assert event["error"]["json_error_pos"] == 23
