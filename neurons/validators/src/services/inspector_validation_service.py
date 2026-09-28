from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import asyncssh
from core.checksums import sha256_from_executor, sha256_from_path
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from core.utils import _m, get_extra_info

if TYPE_CHECKING:
    from services.task.messages import MessageTemplate

logger = logging.getLogger(__name__)

INSPECTOR_LIB_PATH = "/usr/lib/libinspector.so"
INSPECTOR_COMMAND_TIMEOUT_SECONDS = 30
INSPECTOR_STDERR_CAPTURE_TIMEOUT_SECONDS = 10
INSPECTOR_STDERR_CAPTURE_MAX_BYTES = 8192
# One protocol response is one line. asyncssh's readline() hands back a PARTIAL line once a single
# line outgrows the channel receive window (2 MiB by default: the session pauses reading and
# readuntil() returns what it has), so a line is read until its '\n' and this is the ceiling on
# how much of one response the validator will hold before calling the payload unreadable.
INSPECTOR_RESPONSE_MAX_BYTES = 64 * 1024 * 1024
INSPECTOR_PAYLOAD_HEAD_CHARS = 200
# the `error` text of an `ok: false` reply is the executor's to write; this is how much of it is kept
INSPECTOR_ERROR_TEXT_MAX_CHARS = 2048
# the sensor is inside the executor image measured by the CVM's TDX quote
SENSOR_INTEGRITY_MEASURED = "tdx_measured_image"
# a sha256sum run through the provider's own shell — unattested
SENSOR_INTEGRITY_SHELL = "shell_sha256_unattested"


class InspectionFailed(Exception):
    pass


class InspectorInteractiveError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InspectorExecutorExitError(Exception):
    pass


class InspectorUnreadableError(Exception):
    """A stdout line of the interactive protocol that is not one JSON object.

    Carries what a reader needs to tell the cases apart without the raw payload: which command
    was answered, how many bytes came back, whether the line ended in '\\n' (False = cut at EOF
    or at the size cap), the first 200 chars repr-escaped, and the decoder's position.
    """

    def __init__(
        self,
        *,
        cmd: str,
        payload: str,
        terminated: bool,
        json_error: str,
        pos: int | None = None,
        lineno: int | None = None,
        colno: int | None = None,
        payload_bytes: int | None = None,
    ) -> None:
        self.cmd = cmd
        # the reader already counted the bytes; only re-encode when nobody did
        self.payload_bytes = (
            payload_bytes
            if payload_bytes is not None
            else len(payload.encode("utf-8", errors="replace"))
        )
        self.payload_head = repr(payload[:INSPECTOR_PAYLOAD_HEAD_CHARS])
        self.terminated = terminated
        self.json_error = json_error
        self.pos = pos
        self.lineno = lineno
        self.colno = colno
        super().__init__(
            f"inspector executor wrote an unreadable response to {cmd!r}: {json_error}"
            f" ({self.payload_bytes} bytes, {'newline-terminated' if terminated else 'cut'})"
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "payload_cmd": self.cmd,
            "payload_bytes": self.payload_bytes,
            "payload_head": self.payload_head,
            "payload_terminated": self.terminated,
            "json_error": self.json_error,
            "json_error_pos": self.pos,
            "json_error_lineno": self.lineno,
            "json_error_colno": self.colno,
        }


class InspectorValidator:
    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        self.session = None

    @classmethod
    def load_library(cls, lib_path: str) -> ctypes.CDLL:
        lib = ctypes.CDLL(lib_path)
        cls._bind_signatures(lib)
        return lib

    @staticmethod
    def _bind_signatures(lib: ctypes.CDLL) -> None:
        lib.session_new.restype = ctypes.c_void_p
        lib.session_handshake_start.argtypes = [ctypes.c_void_p]
        lib.session_handshake_start.restype = ctypes.POINTER(ctypes.c_char)
        lib.session_handshake_finish.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.session_handshake_finish.restype = ctypes.c_int
        lib.inspector_generate.argtypes = [ctypes.c_void_p]
        lib.inspector_generate.restype = ctypes.POINTER(ctypes.c_char)
        lib.inspector_verify.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.inspector_verify.restype = ctypes.POINTER(ctypes.c_char)
        lib.session_del.argtypes = [ctypes.c_void_p]
        lib.str_del.argtypes = [ctypes.POINTER(ctypes.c_char)]

    def start_session(self) -> None:
        self.session = self.lib.session_new()
        if not self.session:
            raise RuntimeError("session_new failed")

    def close_session(self) -> None:
        if self.session:
            self.lib.session_del(self.session)
            self.session = None

    def _take(self, ptr) -> str:
        if not ptr:
            raise RuntimeError("libinspector call failed")
        try:
            return ctypes.string_at(ptr).decode("utf-8")
        finally:
            self.lib.str_del(ptr)

    def handshake_start(self) -> str:
        return self._take(self.lib.session_handshake_start(self.session))

    def handshake_finish(self, reply_json: str) -> None:
        if self.lib.session_handshake_finish(self.session, reply_json.encode("utf-8")) != 0:
            raise RuntimeError("session_handshake_finish failed")

    def generate(self) -> str:
        return self._take(self.lib.inspector_generate(self.session))

    def verify(self, response_cipher: str) -> dict[str, Any]:
        try:
            body = self._take(
                self.lib.inspector_verify(self.session, response_cipher.encode("utf-8"))
            )
        except RuntimeError as exc:
            raise InspectionFailed("inspector_verify failed") from exc
        return json.loads(body)


@dataclass(frozen=True)
class InspectorValidationResponse:
    report: dict[str, Any] | None = None
    error: str | None = None
    diagnostics: dict[str, Any] | None = None
    message: MessageTemplate | None = None


class InspectorValidationService:
    def __init__(
        self,
        *,
        lib_path: str = INSPECTOR_LIB_PATH,
        command_timeout: int = INSPECTOR_COMMAND_TIMEOUT_SECONDS,
        stderr_capture_timeout: float = INSPECTOR_STDERR_CAPTURE_TIMEOUT_SECONDS,
        stderr_capture_max_bytes: int = INSPECTOR_STDERR_CAPTURE_MAX_BYTES,
        response_max_bytes: int = INSPECTOR_RESPONSE_MAX_BYTES,
    ) -> None:
        self.lib_path = lib_path
        self.command_timeout = command_timeout
        self.stderr_capture_timeout = stderr_capture_timeout
        self.stderr_capture_max_bytes = stderr_capture_max_bytes
        self.response_max_bytes = response_max_bytes
        self.local_checksum = sha256_from_path(self.lib_path)
        self.inspector_lib = InspectorValidator.load_library(self.lib_path)

    async def validate_rented_executor(
        self,
        shell,
        ssh: asyncssh.SSHClientConnection,
        executor: ExecutorSSHInfo,
        default_extra: dict[str, Any],
        *,
        sensor_attested: bool = False,
    ) -> InspectorValidationResponse:
        from services.task.messages import InspectorMessages as Msg

        process = None
        validator = None
        command = self._interactive_command(executor)
        diagnostics: dict[str, Any] = {
            "command": command,
            "executor_uuid": executor.uuid,
            # DAH-3275: what vouches for the sensor binary this report came from. On a dstack
            # CVM the executor image (and the .so in it) is part of the stack the validator
            # measured against TDX_WHITELIST; asking the provider's shell for a sha256sum on
            # top proves nothing (the shell is theirs), so the shell read is skipped there and
            # every other host is marked as what it is.
            "sensor_integrity": SENSOR_INTEGRITY_MEASURED if sensor_attested else SENSOR_INTEGRITY_SHELL,
        }

        try:
            if not sensor_attested:
                executor_checksum = await sha256_from_executor(shell, self.lib_path)
                if self.local_checksum != executor_checksum:
                    return self._failure_response(
                        error=(
                            "Executor using outdated libinspector library. "
                            "Run docker compose restart to update to the latest executor image"
                        ),
                        message=Msg.FAILED_LIB_MISMATCH,
                        diagnostics={
                            **diagnostics,
                            "local_sha256": self.local_checksum,
                            "executor_sha256": executor_checksum or None,
                        },
                        default_extra=default_extra,
                    )

            validator = InspectorValidator(self.inspector_lib)
            validator.start_session()
            process = await ssh.create_process(command)

            if settings.INSPECTOR_ENSURE_COLLECTOR_ON_RENTED_CHECK:
                try:
                    await self._send_message(process, {"cmd": "start-collector"})
                except Exception as exc:
                    diagnostics["collector_ensure_error"] = str(exc)

            open_json = validator.handshake_start()
            handshake_reply = await self._send_message(
                process,
                {"cmd": "handshake-reply", "open_json": open_json},
            )
            validator.handshake_finish(handshake_reply)

            request_cipher = validator.generate()
            response_cipher = await self._send_message(
                process,
                {"cmd": "execute", "request_cipher": request_cipher},
            )
            report = validator.verify(response_cipher)
            return InspectorValidationResponse(
                report=self._normalize_report(report),
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return await self._validation_failure(
                process,
                exc,
                Msg=Msg,
                diagnostics=diagnostics,
                default_extra=default_extra,
            )
        finally:
            if validator is not None:
                validator.close_session()
            if process is not None:
                await self._close_process(process)

    async def _validation_failure(
        self,
        process,
        exc: Exception,
        *,
        Msg,
        diagnostics: dict[str, Any],
        default_extra: dict[str, Any],
    ) -> InspectorValidationResponse:
        if isinstance(exc, asyncio.TimeoutError):
            error = f"Inspector interactive command timed out after {self.command_timeout}s"
            message = Msg.FAILED_TIMEOUT
        elif isinstance(exc, InspectionFailed):
            error = str(exc)
            message = Msg.FAILED_CIPHER_REJECTED
        elif isinstance(exc, InspectorExecutorExitError):
            error = str(exc)
            message = Msg.FAILED_EXECUTOR_CRASH
        elif isinstance(exc, InspectorInteractiveError):
            error = exc.message
            message = Msg.FAILED_INTERACTIVE
        elif isinstance(exc, InspectorUnreadableError):
            # the node is recorded as unreadable — a distinct reason, counted apart from the
            # generic error and never a silent pass; the other executors' checks are their own
            error = str(exc)
            message = Msg.UNREADABLE
            diagnostics = {**diagnostics, **exc.diagnostics()}
        else:
            error = str(exc)
            message = Msg.FAILED_SSH_TRANSPORT if process is None else Msg.VALIDATION_ERROR

        executor_stderr = await self._capture_stderr(process)
        return self._failure_response(
            error=error,
            message=message,
            diagnostics=diagnostics,
            default_extra=default_extra,
            executor_stderr=executor_stderr,
            error_type=type(exc).__name__,
            # DAH-3593: the unclassified branch covers our own library too (InspectorValidator
            # before create_process); only a transport error there is the node's.
            validator_fault=message is Msg.VALIDATION_ERROR
            or (message is Msg.FAILED_SSH_TRANSPORT and not isinstance(exc, (asyncssh.Error, OSError))),
        )

    def _failure_response(
        self,
        *,
        error: str,
        message: MessageTemplate,
        diagnostics: dict[str, Any],
        default_extra: dict[str, Any],
        executor_stderr: str | None = None,
        error_type: str | None = None,
        validator_fault: bool = False,
    ) -> InspectorValidationResponse:
        payload = {
            **diagnostics,
            "reason": message.reason,
            "error": error,
        }
        if error_type:
            payload["error_type"] = error_type
        if executor_stderr:
            payload["executor_stderr"] = executor_stderr
        # DAH-3593: the node failing the check is a verdict (INSPECTOR_FAILED_*), recorded and
        # scored — WARNING. An exception the validator did not classify is still ERROR: that one
        # may be ours.
        if validator_fault:
            logger.error(
                _m(
                    "Inspector validation failed",
                    extra=get_extra_info({**default_extra, **payload}),
                ),
            )
        else:
            logger.warning(
                _m(
                    "Inspector validation failed",
                    extra=get_extra_info({**default_extra, **payload, "reason_class": "node_verdict"}),
                ),
            )
        return InspectorValidationResponse(
            error=error,
            diagnostics=payload,
            message=message,
        )

    def _interactive_command(self, executor: ExecutorSSHInfo) -> str:
        script = f"{executor.root_dir.rstrip('/')}/src/inspector_executor.py"
        return f"{shlex.quote(executor.python_path)} {shlex.quote(script)} --interactive"

    async def _send_message(self, process, payload: dict[str, Any]) -> str:
        cmd = str(payload.get("cmd", ""))
        process.stdin.write(json.dumps(payload) + "\n")
        line = await self._read_response_line(process, cmd)

        if not line:
            raise InspectorExecutorExitError("inspector executor exited without a response")

        response = self._parse_response(line, cmd)
        if not response.get("ok"):
            # the executor's text, kept a string and bounded before it reaches the log or the row
            error_text = str(response.get("error") or "inspector executor command failed")
            raise InspectorInteractiveError(error_text[:INSPECTOR_ERROR_TEXT_MAX_CHARS])
        return response.get("result", "")

    async def _read_response_line(self, process, cmd: str) -> str:
        """One protocol line, whole: reassembled across the partial reads asyncssh returns when a
        line is longer than the channel window, under one deadline for the whole line and the
        `response_max_bytes` cap. An empty string is EOF before any byte of the response."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.command_timeout
        chunks: list[str] = []
        total_bytes = 0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError()
            chunk = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            if not chunk:
                break  # EOF: whatever was buffered is the whole payload
            chunks.append(chunk)
            total_bytes += len(chunk.encode("utf-8", errors="replace"))
            if total_bytes >= self.response_max_bytes:
                # the cap is checked before the newline so no line above it reaches the decoder;
                # only the head is kept — never a copy of everything that was buffered
                raise InspectorUnreadableError(
                    cmd=cmd,
                    payload=chunks[0][:INSPECTOR_PAYLOAD_HEAD_CHARS],
                    payload_bytes=total_bytes,
                    terminated=False,
                    json_error=f"response line exceeds the {self.response_max_bytes}-byte cap",
                )
            if chunk.endswith("\n"):
                break
        return "".join(chunks)

    @staticmethod
    def _parse_response(line: str, cmd: str) -> dict[str, Any]:
        terminated = line.endswith("\n")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InspectorUnreadableError(
                cmd=cmd,
                payload=line,
                terminated=terminated,
                json_error=exc.msg,
                pos=exc.pos,
                lineno=exc.lineno,
                colno=exc.colno,
            ) from exc
        if not isinstance(response, dict):
            raise InspectorUnreadableError(
                cmd=cmd,
                payload=line,
                terminated=terminated,
                json_error=f"response is not a JSON object ({type(response).__name__})",
            )
        return response

    async def _capture_stderr(self, process) -> str | None:
        if process is None:
            return None

        stderr = getattr(process, "stderr", None)
        readline = getattr(stderr, "readline", None)
        if readline is None:
            return None

        captured: list[str] = []
        captured_bytes = 0
        while captured_bytes < self.stderr_capture_max_bytes:
            try:
                line = await asyncio.wait_for(
                    readline(),
                    timeout=self.stderr_capture_timeout,
                )
            except asyncio.TimeoutError:
                break
            except Exception:
                break
            if not line:
                break

            if isinstance(line, bytes):
                text = line.decode("utf-8", errors="replace")
            else:
                text = str(line)

            encoded = text.encode("utf-8", errors="replace")
            remaining = self.stderr_capture_max_bytes - captured_bytes
            if len(encoded) > remaining:
                captured.append(encoded[:remaining].decode("utf-8", errors="replace"))
                captured_bytes = self.stderr_capture_max_bytes
                break

            captured.append(text)
            captured_bytes += len(encoded)

        stderr_text = "".join(captured).strip()
        return stderr_text or None

    async def _close_process(self, process) -> None:
        try:
            await self._send_message(process, {"cmd": "quit"})
        except Exception:
            terminate = getattr(process, "terminate", None)
            if terminate:
                terminate()
        wait = getattr(process, "wait", None)
        if wait:
            try:
                await asyncio.wait_for(wait(), timeout=5)
            except Exception:
                kill = getattr(process, "kill", None)
                if kill:
                    kill()

    def _normalize_report(self, report: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(report)
        normalized.setdefault("canary_ok", None)
        normalized.setdefault("findings", [])
        normalized.setdefault("summary", {})
        normalized.setdefault("health", {})
        return normalized
