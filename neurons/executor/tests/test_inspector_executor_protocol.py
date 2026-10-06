"""`inspector_executor.py --interactive` owns its stdout: the validator reads it one line at a time.

Anything else that reaches fd 1 while the executor runs — a dependency's printf, a stray print, a
status line — has to land on stderr, and every reply has to be a single line whatever the result
string contains. These tests pin those two properties; the command set itself is not spelled out.
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
    """Stands in for the ctypes-backed InspectorExecutor so no .so is loaded."""

    def __init__(self) -> None:
        self.started = 0

    def start_collector(self) -> None:
        self.started += 1
        sys.stdout.write("lib says: collector thread up\n")


def _driver(tmp_path, body: str) -> Path:
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
            """
        )
        + textwrap.dedent(body)
    )
    return driver


def test_emit_writes_exactly_one_ascii_line_whatever_the_result_contains():
    out = io.StringIO()

    inspector_executor._emit(out, True, result=RESULT_WITH_HAZARDS)
    inspector_executor._emit(out, False, error=RESULT_WITH_HAZARDS)

    text = out.getvalue()
    assert text.endswith("\n")
    lines = text.split("\n")[:-1]
    assert len(lines) == 2, "one physical line per reply: newline, U+2028 and quotes are escaped"
    assert all(line.isascii() for line in lines)
    for line in lines:
        decoded = json.loads(line)
        assert isinstance(decoded, dict)
        assert RESULT_WITH_HAZARDS in decoded.values(), "the text round-trips unchanged"


def test_claimed_protocol_stream_is_the_only_writer_to_the_validators_pipe(tmp_path):
    # the real fd plumbing in a subprocess: fd 1 is the validator's pipe, fd 2 the executor's log
    driver = _driver(
        tmp_path,
        """
        import os
        out = inspector_executor._claim_protocol_stream()
        os.write(1, b"libinspector: handshake ok\\n")   # a C dependency's printf
        print("stray print from a helper")             # a stray print
        sys.stdout.write("lib says: collector thread up\\n")
        sys.stdout.flush()
        out.write("protocol line\\n")
        out.flush()
        """,
    )
    proc = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "protocol line\n"
    assert "libinspector: handshake ok" in proc.stderr
    assert "stray print from a helper" in proc.stderr
    assert "lib says: collector thread up" in proc.stderr


def test_malformed_request_lines_get_a_reply_and_do_not_end_the_session():
    # neither a non-JSON line nor a JSON scalar reaches the library or raises out of the loop
    out = io.StringIO()

    inspector_executor.run_interactive(
        FakeLibExecutor(), stdin=io.StringIO("not json at all\n42\n\n"), out=out
    )

    lines = out.getvalue().split("\n")[:-1]
    assert len(lines) == 2, "one reply per malformed line, none for the blank one"
    for line in lines:
        assert line.isascii()
        assert isinstance(json.loads(line), dict)


def test_collector_status_lines_go_to_stderr(tmp_path):
    driver = _driver(
        tmp_path,
        """
        sys.argv = ["inspector_executor.py", "--start-collector"]
        inspector_executor.main()
        """,
    )
    proc = subprocess.run(
        [sys.executable, str(driver)], capture_output=True, text=True, timeout=30
    )

    assert proc.returncode == 0, proc.stderr
    assert "collector running" not in proc.stdout
    assert "collector running" in proc.stderr
