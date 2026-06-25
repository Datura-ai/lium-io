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


class InspectionFailed(Exception):
    pass


class InspectorInteractiveError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InspectorExecutorExitError(Exception):
    pass


class InspectorValidator:
    def __init__(self, lib_path: str) -> None:
        self.lib = ctypes.CDLL(lib_path)
        self._bind_signatures()
        self.session = None

    def _bind_signatures(self) -> None:
        self.lib.session_new.restype = ctypes.c_void_p
        self.lib.session_handshake_start.argtypes = [ctypes.c_void_p]
        self.lib.session_handshake_start.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.session_handshake_finish.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.session_handshake_finish.restype = ctypes.c_int
        self.lib.inspector_generate.argtypes = [ctypes.c_void_p]
        self.lib.inspector_generate.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.inspector_verify.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.inspector_verify.restype = ctypes.POINTER(ctypes.c_char)
        self.lib.session_del.argtypes = [ctypes.c_void_p]
        self.lib.str_del.argtypes = [ctypes.POINTER(ctypes.c_char)]

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
    ) -> None:
        self.lib_path = lib_path
        self.command_timeout = command_timeout
        self.stderr_capture_timeout = stderr_capture_timeout
        self.stderr_capture_max_bytes = stderr_capture_max_bytes

    async def validate_rented_executor(
        self,
        shell,
        ssh: asyncssh.SSHClientConnection,
        executor: ExecutorSSHInfo,
        default_extra: dict[str, Any],
    ) -> InspectorValidationResponse:
        from services.task.messages import InspectorMessages as Msg

        process = None
        validator = None
        command = self._interactive_command(executor)
        diagnostics: dict[str, Any] = {
            "command": command,
            "executor_uuid": executor.uuid,
        }

        try:
            local_checksum = sha256_from_path(self.lib_path)
            executor_checksum = await sha256_from_executor(shell, self.lib_path)
            if local_checksum != executor_checksum:
                return self._failure_response(
                    error=(
                        "Executor using outdated libinspector library. "
                        "Run docker compose restart to update to the latest executor image"
                    ),
                    message=Msg.FAILED_LIB_MISMATCH,
                    diagnostics={
                        **diagnostics,
                        "local_sha256": local_checksum,
                        "executor_sha256": executor_checksum or None,
                    },
                    default_extra=default_extra,
                )

            validator = InspectorValidator(self.lib_path)
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
        logger.error(
            _m(
                "Inspector validation failed",
                extra=get_extra_info({**default_extra, **payload}),
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
        process.stdin.write(json.dumps(payload) + "\n")
        try:
            line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self.command_timeout,
            )
        except asyncio.TimeoutError:
            raise

        if not line:
            raise InspectorExecutorExitError("inspector executor exited without a response")

        response = json.loads(line)
        if not response.get("ok"):
            raise InspectorInteractiveError(
                response.get("error", "inspector executor command failed")
            )
        return response.get("result", "")

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
