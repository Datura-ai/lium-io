"""`inspector_executor.py --interactive` writes exactly one JSON document per command to stdout.

The validator parses that stream line by line (`json.loads` per line). Anything else on fd 1 — a
dependency's printf, a stray print, the collector's status line — turns into an unreadable payload
on the validator (INSPECTOR_UNREADABLE) and the node's integrity check is skipped. So the
interactive protocol takes fd 1 for itself and points fd 1 at stderr for everyone else, and a
result string with a quote, a newline or a non-ASCII name still comes out as one escaped line.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import inspector_executor

SCRIPT = Path(inspector_executor.__file__).resolve()

RESULT_WITH_HAZARDS = 'GPU "NVIDIA \u00dcber" busy\nsecond line\ttab\u2028ls \\ backslash'


class FakeLibExecutor:
    """Stands in for the ctypes-backed InspectorExecutor; every result carries hazards."""

    def __init__(self) -> None:
        self.started = 0

    def handshake_reply(self, open_json: str) -> str:
        # what a C dependency does when it is chatty: printf to fd 1, not through sys.stdout
        os.write(1, b"libinspector: handshake ok\n")
        return json.dumps({"hello": "executor", "open": open_json})

    def execute(self, request_cipher: str) -> str:
        print("stray print from a helper")
        return RESULT_WITH_HAZARDS

    def start_collector(self) -> None:
        self.started += 1
        sys.stdout.write("lib says: collector thread up\n")


def _only_json_lines(text: str) -> list[dict]:
    lines = text.split("\n")
    assert lines[-1] == "", "stdout ends with the protocol line's newline"
    return [json.loads(line) for line in lines[:-1]]


def test_run_interactive_emits_one_escaped_json_line_per_command():
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"cmd": "start-collector"}),
                json.dumps({"cmd": "handshake-reply", "open_json": '{"hello": "validator"}'}),
                "not json at all",
                "42",
                json.dumps({"cmd": "execute", "request_cipher": "cipher"}),
                json.dumps({"cmd": "quit"}),
                json.dumps({"cmd": "execute", "request_cipher": "after quit, never read"}),
            ]
        )
        + "\n"
    )
    protocol_out = io.StringIO()

    inspector_executor.run_interactive(FakeLibExecutor(), stdin=stdin, out=protocol_out)

    responses = _only_json_lines(protocol_out.getvalue())
    assert responses == [
        {"ok": True, "result": ""},
        {"ok": True, "result": json.dumps({"hello": "executor", "open": '{"hello": "validator"}'})},
        {"ok": False, "error": "invalid json: Expecting value: line 1 column 1 (char 0)"},
        {"ok": False, "error": "invalid request: expected a JSON object, got int"},
        {"ok": True, "result": RESULT_WITH_HAZARDS},
        {"ok": True, "result": ""},
    ]
    raw_lines = protocol_out.getvalue().split("\n")[:-1]
    assert len(raw_lines) == 6, "one physical line per command, whatever the result contains"
    assert all(line.isascii() for line in raw_lines), "the wire is ASCII: every non-ASCII char is escaped"


def test_interactive_stdout_carries_only_the_protocol(tmp_path):
    # the real entry point in a subprocess: fd 1 is the validator's pipe, fd 2 the executor's log
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(SCRIPT.parent)!r})
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            import inspector_executor
            from test_inspector_executor_protocol import FakeLibExecutor
            inspector_executor.InspectorExecutor = FakeLibExecutor
            sys.argv = ["inspector_executor.py", "--interactive"]
            inspector_executor.main()
            """
        )
    )
    commands = "\n".join(
        [
            json.dumps({"cmd": "start-collector"}),
            json.dumps({"cmd": "handshake-reply", "open_json": "{}"}),
            json.dumps({"cmd": "execute", "request_cipher": "cipher"}),
            json.dumps({"cmd": "quit"}),
        ]
    )
    proc = subprocess.run(
        [sys.executable, str(driver)],
        input=commands + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    assert proc.returncode == 0, proc.stderr
    responses = _only_json_lines(proc.stdout)
    assert [r["ok"] for r in responses] == [True, True, True, True]
    assert responses[2]["result"] == RESULT_WITH_HAZARDS
    assert "libinspector: handshake ok" in proc.stderr
    assert "stray print from a helper" in proc.stderr
    assert "lib says: collector thread up" in proc.stderr
    assert "libinspector" not in proc.stdout
    assert "stray" not in proc.stdout
    assert "lib says" not in proc.stdout


def test_collector_status_lines_go_to_stderr(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(SCRIPT.parent)!r})
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            import inspector_executor
            from test_inspector_executor_protocol import FakeLibExecutor
            inspector_executor.InspectorExecutor = FakeLibExecutor
            sys.argv = ["inspector_executor.py", "--start-collector"]
            inspector_executor.main()
            """
        )
    )
    proc = subprocess.run(
        [sys.executable, str(driver)], capture_output=True, text=True, timeout=30
    )

    assert proc.returncode == 0, proc.stderr
    assert "collector running" not in proc.stdout
    assert "collector running" in proc.stderr
