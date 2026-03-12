import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch


os.environ.setdefault(
    "MINER_HOTKEY_SS58_ADDRESS",
    "5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVU",
)
os.environ.setdefault("DB_URI", "sqlite:///test.db")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.config import settings
from services.chutes_relay_service import (
    ChutesExecutorError,
    ChutesMalformedResponseError,
    ChutesRelayService,
    ChutesTransportError,
)


class TestChutesRelayService(unittest.TestCase):
    def setUp(self):
        self.service = ChutesRelayService()

    def test_build_bridge_command(self):
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch.object(settings, "CHUTES_BRIDGE_SSH_PORT", 2222),
            patch.object(settings, "CHUTES_BRIDGE_SSH_USER", "lium-bridge"),
            patch.object(settings, "CHUTES_BRIDGE_SSH_KEY_PATH", "/tmp/key"),
            patch.object(settings, "CHUTES_BRIDGE_CONNECT_TIMEOUT_SEC", 7),
        ):
            command = self.service.build_bridge_command("start")
        self.assertEqual(
            command,
            [
                "ssh",
                "-i",
                "/tmp/key",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=7",
                "-p",
                "2222",
                "lium-bridge@127.0.0.1",
                "bridgectl",
                "start",
            ],
        )

    def test_status_parses_valid_json(self):
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout=json.dumps({"ok": True, "state": "not_installed"}),
            stderr="",
        )
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch("services.chutes_relay_service.subprocess.run", return_value=completed),
        ):
            payload = self.service.status()
        self.assertEqual(payload["state"], "not_installed")

    def test_status_preserves_escaped_last_error(self):
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "state": "error",
                    "last_error": 'bad "quote" and slash \\\\ content',
                }
            ),
            stderr="",
        )
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch("services.chutes_relay_service.subprocess.run", return_value=completed),
        ):
            payload = self.service.status()
        self.assertEqual(payload["last_error"], 'bad "quote" and slash \\\\ content')

    def test_transport_failure_is_raised(self):
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="ssh: connect to host 127.0.0.1 port 22: Connection refused",
        )
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch("services.chutes_relay_service.subprocess.run", return_value=completed),
        ):
            with self.assertRaises(ChutesTransportError):
                self.service.status()

    def test_malformed_json_is_rejected(self):
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch("services.chutes_relay_service.subprocess.run", return_value=completed),
        ):
            with self.assertRaises(ChutesMalformedResponseError):
                self.service.status()

    def test_ok_false_payload_is_raised_as_executor_error(self):
        completed = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout=json.dumps({"ok": False, "state": "error", "error": "agent unhealthy"}),
            stderr="",
        )
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch("services.chutes_relay_service.subprocess.run", return_value=completed),
        ):
            with self.assertRaisesRegex(ChutesExecutorError, "agent unhealthy"):
                self.service.status()

    def test_install_timeout_is_sanitized_and_has_no_cause(self):
        timeout_error = subprocess.TimeoutExpired(
            cmd=["ssh", "bridgectl", "setup", "--hotkey-seed", "super-secret-seed"],
            timeout=self.service.INSTALL_TIMEOUT_SEC,
        )
        with (
            patch.object(settings, "CHUTES_BRIDGE_ENABLED", True),
            patch.object(settings, "CHUTES_BRIDGE_SSH_HOST", "127.0.0.1"),
            patch("services.chutes_relay_service.subprocess.run", side_effect=timeout_error),
        ):
            with self.assertRaises(ChutesExecutorError) as exc_info:
                self.service.install(
                    validator_hotkey="validator",
                    hotkey_ss58="miner",
                    hotkey_seed="super-secret-seed",
                    node_name="gpu-node-01",
                )

        self.assertEqual(
            str(exc_info.exception),
            "Chutes bridge command 'setup' timed out after 3600s",
        )
        self.assertIsNone(exc_info.exception.__cause__)
        self.assertNotIn("super-secret-seed", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
