import json
import subprocess
import time
from typing import Any

from core.config import settings
from core.logger import _m, get_logger

logger = get_logger(__name__)


class ChutesRelayDisabledError(RuntimeError):
    pass


class ChutesBridgeConfigError(RuntimeError):
    pass


class ChutesTransportError(RuntimeError):
    pass


class ChutesMalformedResponseError(RuntimeError):
    pass


class ChutesExecutorError(RuntimeError):
    pass


class ChutesRelayService:
    INSTALL_TIMEOUT_SEC = 3600
    TRANSPORT_ERROR_MARKERS = (
        "Permission denied",
        "Connection refused",
        "No route to host",
        "Connection timed out",
        "Host key verification failed",
        "Could not resolve hostname",
        "Operation timed out",
        "Network is unreachable",
    )

    def build_bridge_command(self, verb: str, args: list[str] | None = None) -> list[str]:
        if not settings.CHUTES_BRIDGE_ENABLED:
            raise ChutesRelayDisabledError("Chutes bridge relay is disabled")
        if not settings.CHUTES_BRIDGE_SSH_HOST:
            raise ChutesBridgeConfigError(
                "CHUTES_BRIDGE_SSH_HOST is required when bridge relay is enabled"
            )

        command = [
            "ssh",
            "-i",
            settings.CHUTES_BRIDGE_SSH_KEY_PATH,
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"ConnectTimeout={settings.CHUTES_BRIDGE_CONNECT_TIMEOUT_SEC}",
            "-p",
            str(settings.CHUTES_BRIDGE_SSH_PORT),
            f"{settings.CHUTES_BRIDGE_SSH_USER}@{settings.CHUTES_BRIDGE_SSH_HOST}",
            "bridgectl",
            verb,
        ]
        if args:
            command.extend(args)
        return command

    def install(
        self,
        validator_hotkey: str,
        hotkey_ss58: str,
        hotkey_seed: str,
        node_name: str,
    ) -> dict[str, Any]:
        return self._run_bridge_command(
            "setup",
            [
                "--validator-hotkey",
                validator_hotkey,
                "--hotkey-ss58",
                hotkey_ss58,
                "--hotkey-seed",
                hotkey_seed,
                "--node-name",
                node_name,
            ],
            timeout=self.INSTALL_TIMEOUT_SEC,
        )

    def start(self) -> dict[str, Any]:
        return self._run_bridge_command(
            "start",
            timeout=settings.CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC,
        )

    def stop(self) -> dict[str, Any]:
        return self._run_bridge_command(
            "stop",
            timeout=settings.CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC,
        )

    def status(self) -> dict[str, Any]:
        return self._run_bridge_command(
            "status",
            timeout=settings.CHUTES_BRIDGE_COMMAND_TIMEOUT_SEC,
        )

    def get_status_summary(self) -> dict[str, Any]:
        if not settings.CHUTES_BRIDGE_ENABLED:
            return {
                "ok": True,
                "capability": {
                    "bridge_enabled": False,
                    "bridge_accessible": False,
                    "can_install": False,
                },
                "bridge": None,
                "error": "Chutes bridge relay is disabled",
            }

        try:
            bridge = self.status()
            return {
                "ok": True,
                "capability": {
                    "bridge_enabled": True,
                    "bridge_accessible": True,
                    "can_install": True,
                },
                "bridge": bridge,
                "error": None,
            }
        except (
            ChutesTransportError,
            ChutesRelayDisabledError,
            ChutesBridgeConfigError,
            ChutesMalformedResponseError,
            ChutesExecutorError,
        ) as exc:
            return {
                "ok": True,
                "capability": {
                    "bridge_enabled": True,
                    "bridge_accessible": False,
                    "can_install": False,
                },
                "bridge": None,
                "error": str(exc),
            }

    def _run_bridge_command(
        self,
        verb: str,
        args: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        command = self.build_bridge_command(verb, args)
        started_at = time.monotonic()
        logger.info(_m("Running Chutes bridge command", extra={"verb": verb}))

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ChutesExecutorError(
                f"Chutes bridge command '{verb}' timed out after {timeout}s"
            ) from None
        except OSError as exc:
            raise ChutesExecutorError(f"Failed to execute ssh for '{verb}': {exc}") from None

        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()

        logger.info(
            _m(
                "Chutes bridge command finished",
                extra={
                    "verb": verb,
                    "returncode": completed.returncode,
                    "elapsed_ms": elapsed_ms,
                },
            )
        )

        if self._is_transport_error(completed.returncode, stderr):
            raise ChutesTransportError(self._summarize_stderr(stderr))

        if not stdout:
            if completed.returncode == 0:
                raise ChutesMalformedResponseError(
                    f"Chutes bridge command '{verb}' returned empty output"
                )
            raise ChutesExecutorError(
                f"Chutes bridge command '{verb}' failed: {self._summarize_stderr(stderr)}"
            )

        payload = self._extract_json(stdout)
        if payload is None:
            raise ChutesMalformedResponseError(
                f"Chutes bridge command '{verb}' returned malformed JSON"
            )

        if not isinstance(payload, dict):
            raise ChutesMalformedResponseError(
                f"Chutes bridge command '{verb}' returned non-object JSON"
            )

        if payload.get("ok") is False:
            raise ChutesExecutorError(
                f"Chutes bridge command '{verb}' failed: {self._summarize_bridge_error(payload, stderr)}"
            )

        return payload

    @staticmethod
    def _extract_json(stdout: str) -> dict[str, Any] | None:
        """Extract JSON object from stdout that may contain non-JSON prefix (helm, k3s, etc.)."""
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass
        # Find the last '{' that starts a line — bridge scripts output JSON at the end
        for i in range(len(stdout) - 1, -1, -1):
            if stdout[i] == "{" and (i == 0 or stdout[i - 1] == "\n"):
                try:
                    return json.loads(stdout[i:])
                except json.JSONDecodeError:
                    continue
        return None

    def _is_transport_error(self, returncode: int, stderr: str) -> bool:
        if returncode == 255:
            return True
        return any(marker in stderr for marker in self.TRANSPORT_ERROR_MARKERS)

    def _summarize_stderr(self, stderr: str) -> str:
        if stderr:
            return stderr.splitlines()[-1]
        return "Unknown SSH transport error"

    def _summarize_bridge_error(self, payload: dict[str, Any], stderr: str) -> str:
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        if stderr:
            return self._summarize_stderr(stderr)
        return "Bridge command returned ok=false"
