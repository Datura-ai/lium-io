"""The inspector's interactive protocol is one JSON line per command on the executor's stdout.

On 20 Sep 2026 every inspector run on one rented executor died with
``JSONDecodeError: Unterminated string starting at line 1 column 24`` — column 24 is the opening
quote of the ``result`` value in ``{"ok": true, "result": "…``. asyncssh's ``SSHReader.readline()`` returns
a PARTIAL line when a single line is longer than the channel receive window (2 MiB by default):
the session pauses reading at the window and ``readuntil`` gives back what it has. The response
cipher for a host with a large collector report crosses that window, so the validator parsed the
first 2 MiB of the line and the integrity check never ran on that node.

These tests pin both halves of the fix: a line is reassembled across partial reads, and a payload
that still does not parse (a non-JSON line on stdout, a line cut at EOF, an unescaped quote) is
recorded as ``INSPECTOR_UNREADABLE`` with typed diagnostics — the executor id, the byte length, the
first 200 chars repr-escaped and the decoder's position — and the other executors are unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from neurons.validators.src.services import inspector_validation_service as ivs
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationService,
)
from neurons.validators.src.services.task.messages import InspectorMessages as Msg
from test_inspector_validation_service import (  # noqa: F401 — the autouse fixtures ride along
    FakeShell,
    FakeSSH,
    FakeStderr,
    FakeStdout,
    FakeValidator,
    enable_collector_ensure,
    fake_validator,
    matching_lib_checksums,
)

HANDSHAKE_OK = json.dumps({"ok": True, "result": '{"hello": "executor"}'}) + "\n"
EXECUTE_OK = json.dumps({"ok": True, "result": "response-cipher"}) + "\n"
EMPTY_OK = json.dumps({"ok": True, "result": ""}) + "\n"


def _executor(uuid: str = "exec-1") -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid, python_path="/usr/bin/python3", root_dir="/root/app")


def _stdout(*execute_lines: str) -> list[str]:
    # start-collector ok, handshake ok, then the execute response as given, then quit ok
    return [EMPTY_OK, HANDSHAKE_OK, *execute_lines, EMPTY_OK]


async def _validate(ssh: FakeSSH, uuid: str = "exec-1") -> ivs.InspectorValidationResponse:
    service = InspectorValidationService()
    return await service.validate_rented_executor(
        FakeShell(), ssh, _executor(uuid), {"executor_uuid": uuid}
    )


def _assert_unreadable(result: ivs.InspectorValidationResponse, *, cmd: str = "execute") -> dict:
    assert result.report is None
    assert result.message is not None
    assert result.message.reason == "INSPECTOR_UNREADABLE"
    assert result.message.reason == Msg.UNREADABLE.reason
    diagnostics = result.diagnostics or {}
    assert diagnostics["reason"] == "INSPECTOR_UNREADABLE"
    assert diagnostics["error_type"] == "InspectorUnreadableError"
    assert diagnostics["executor_uuid"]
    assert diagnostics["payload_cmd"] == cmd
    assert isinstance(diagnostics["payload_bytes"], int)
    assert isinstance(diagnostics["payload_head"], str)
    assert len(diagnostics["payload_head"]) <= 200 + 2  # repr quotes
    assert isinstance(diagnostics["payload_terminated"], bool)
    assert isinstance(diagnostics["json_error"], str)
    return diagnostics


# --- the shape found on 252f0496: one response line longer than the SSH receive window ---------


@pytest.mark.asyncio
async def test_response_line_cut_by_the_receive_window_is_reassembled(monkeypatch):
    # asyncssh returns the first `window` chars without '\n', the rest on the next readline()
    verified: list[str] = []

    class RecordingValidator(FakeValidator):
        def verify(self, response_cipher: str) -> dict:
            verified.append(response_cipher)
            return super().verify("response-cipher")

    monkeypatch.setattr(ivs, "InspectorValidator", RecordingValidator)
    cipher = "A" * 3000
    full = json.dumps({"ok": True, "result": cipher}) + "\n"
    cut = 2048
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout(full[:cut], full[cut:]))

    result = await _validate(ssh)

    assert result.error is None
    assert result.report is not None
    assert verified == [cipher]


@pytest.mark.asyncio
async def test_response_line_cut_at_eof_is_unreadable_with_position():
    # the executor died (or the channel closed) mid-line: the partial line then EOF
    partial = '{"ok": true, "result": "' + "A" * 500
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout(partial, ""))

    result = await _validate(ssh)

    diagnostics = _assert_unreadable(result)
    assert diagnostics["payload_terminated"] is False
    assert diagnostics["payload_bytes"] == len(partial.encode())
    assert diagnostics["json_error_lineno"] == 1
    assert diagnostics["json_error_colno"] == 24
    assert diagnostics["json_error_pos"] == 23
    assert diagnostics["payload_head"] == repr(partial[:200])
    assert "Unterminated string" in diagnostics["json_error"]
    assert "Unterminated string" in (result.error or "")


@pytest.mark.asyncio
async def test_response_line_over_the_hard_cap_is_unreadable():
    service = InspectorValidationService(response_max_bytes=1024)
    line = json.dumps({"ok": True, "result": "B" * 4096}) + "\n"
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout(line[:1024], line[1024:]))

    result = await service.validate_rented_executor(
        FakeShell(), ssh, _executor(), {"executor_uuid": "exec-1"}
    )

    diagnostics = _assert_unreadable(result)
    assert diagnostics["payload_terminated"] is False
    assert diagnostics["payload_bytes"] >= 1024
    assert "cap" in diagnostics["json_error"]


@pytest.mark.asyncio
async def test_response_line_ending_in_a_newline_past_the_cap_is_still_unreadable():
    # the cap is checked before the newline: a terminating chunk does not smuggle a bigger line in
    service = InspectorValidationService(response_max_bytes=1024)
    line = json.dumps({"ok": True, "result": "C" * 2048}) + "\n"
    cut = 1000  # first chunk under the cap, the second ends the line and crosses it
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout(line[:cut], line[cut:]))

    result = await service.validate_rented_executor(
        FakeShell(), ssh, _executor(), {"executor_uuid": "exec-1"}
    )

    diagnostics = _assert_unreadable(result)
    assert diagnostics["payload_terminated"] is False
    assert diagnostics["payload_bytes"] == len(line.encode())
    assert diagnostics["payload_head"] == repr(line[:200])
    assert "cap" in diagnostics["json_error"]


@pytest.mark.asyncio
async def test_error_text_of_a_failed_reply_is_bounded():
    # the `error` of an `ok: false` reply is the executor's text: kept a string, cut at the cap
    long_error = "E" * (ivs.INSPECTOR_ERROR_TEXT_MAX_CHARS * 4)
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(
        _stdout(json.dumps({"ok": False, "error": long_error}) + "\n")
    )

    result = await _validate(ssh)

    assert result.report is None
    assert result.message is not None
    assert result.message.reason == Msg.FAILED_INTERACTIVE.reason
    assert result.error == "E" * ivs.INSPECTOR_ERROR_TEXT_MAX_CHARS
    assert (result.diagnostics or {})["error"] == "E" * ivs.INSPECTOR_ERROR_TEXT_MAX_CHARS


@pytest.mark.asyncio
async def test_error_of_a_failed_reply_that_is_not_a_string_is_made_one():
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(
        _stdout(json.dumps({"ok": False, "error": {"code": 7, "why": ["x"]}}) + "\n")
    )

    result = await _validate(ssh)

    assert result.message is not None
    assert result.message.reason == Msg.FAILED_INTERACTIVE.reason
    assert isinstance(result.error, str)
    assert result.error == str({"code": 7, "why": ["x"]})


# --- the other ways a stdout line stops being one JSON document ---------------------------------


@pytest.mark.asyncio
async def test_interleaved_non_json_line_is_unreadable():
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout("WARNING: collector buffer at 80%\n", EXECUTE_OK))

    result = await _validate(ssh)

    diagnostics = _assert_unreadable(result)
    assert diagnostics["payload_terminated"] is True
    assert diagnostics["payload_head"] == repr("WARNING: collector buffer at 80%\n")
    assert diagnostics["json_error_colno"] == 1


@pytest.mark.asyncio
async def test_unescaped_quote_in_a_name_is_unreadable():
    line = '{"ok": true, "result": "GPU "NVIDIA "Foo" Edition" busy"}\n'
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout(line))

    result = await _validate(ssh)

    with pytest.raises(json.JSONDecodeError) as decoder:
        json.loads(line)

    diagnostics = _assert_unreadable(result)
    assert diagnostics["payload_terminated"] is True
    assert diagnostics["json_error_pos"] == decoder.value.pos > 23
    assert diagnostics["json_error_lineno"] == 1
    assert diagnostics["json_error"] == decoder.value.msg


@pytest.mark.asyncio
async def test_json_that_is_not_an_object_is_unreadable():
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout("42\n"))

    result = await _validate(ssh)

    diagnostics = _assert_unreadable(result)
    assert "not a JSON object" in diagnostics["json_error"]
    assert diagnostics["payload_head"] == repr("42\n")


@pytest.mark.asyncio
async def test_unreadable_handshake_reply_names_its_command():
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout([EMPTY_OK, "not json\n", EMPTY_OK])

    result = await _validate(ssh)

    _assert_unreadable(result, cmd="handshake-reply")


# --- what the run does with it ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreadable_payload_is_logged_with_typed_fields(caplog):
    partial = '{"ok": true, "result": "' + "Z" * 300
    ssh = FakeSSH()
    ssh.process.stdout = FakeStdout(_stdout(partial, ""))
    ssh.process.stderr = FakeStderr(["Killed\n"])

    with caplog.at_level(logging.ERROR, logger=ivs.__name__):
        await _validate(ssh, uuid="exec-252f0496")

    records = [r for r in caplog.records if str(r.msg) == "Inspector validation failed"]
    assert len(records) == 1
    extra = records[0].msg.extra
    assert extra["executor_uuid"] == "exec-252f0496"
    assert extra["reason"] == "INSPECTOR_UNREADABLE"
    assert extra["error_type"] == "InspectorUnreadableError"
    assert extra["payload_cmd"] == "execute"
    assert extra["payload_bytes"] == len(partial.encode())
    assert extra["payload_head"] == repr(partial[:200])
    assert extra["payload_terminated"] is False
    assert extra["json_error_pos"] == 23
    assert extra["json_error_lineno"] == 1
    assert extra["json_error_colno"] == 24
    assert extra["executor_stderr"] == "Killed"


@pytest.mark.asyncio
async def test_an_unreadable_node_does_not_stop_the_other_nodes():
    bad = FakeSSH()
    bad.process.stdout = FakeStdout(_stdout('{"ok": true, "result": "' + "Q" * 100, ""))
    good_before = FakeSSH()
    good_after = FakeSSH()

    results = await asyncio.gather(
        _validate(good_before, uuid="exec-good-1"),
        _validate(bad, uuid="exec-bad"),
        _validate(good_after, uuid="exec-good-2"),
    )

    assert results[0].error is None and results[0].report is not None
    assert results[2].error is None and results[2].report is not None
    diagnostics = _assert_unreadable(results[1])
    assert diagnostics["executor_uuid"] == "exec-bad"


@pytest.mark.asyncio
async def test_a_clean_payload_still_parses():
    ssh = FakeSSH()

    result = await _validate(ssh)

    assert result.error is None
    assert result.report is not None
    assert result.report["canary_ok"] is True


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
            error="inspector executor wrote an unreadable response to 'execute': Unterminated string",
            diagnostics={
                "executor_uuid": "executor-123",
                "reason": Msg.UNREADABLE.reason,
                "error_type": "InspectorUnreadableError",
                "payload_cmd": "execute",
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
