import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ.setdefault(
    "MINER_HOTKEY_SS58_ADDRESS",
    "5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVU",
)
os.environ.setdefault("DB_URI", "sqlite:///test.db")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../datura")))

from core.config import settings
from datura.requests.validator_requests import AuthenticationPayload
from executor import app
from services.chutes_relay_service import ChutesExecutorError


VALID_VALIDATOR_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
VALID_HOTKEY_SS58 = "5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVU"
VALID_SEED = "0x" + ("ab" * 32)
NORMALIZED_SEED = "ab" * 32
VALID_NODE_NAME = "gpu-node-01"
FIXED_NOW = 1_773_100_000


def _signature_for(ss58_address: str, message: str) -> str:
    return f"0xsig::{ss58_address}::{message}"


class _SignedHeaderKeypair:
    def __init__(self, ss58_address=None):
        self.ss58_address = ss58_address

    def verify(self, message, signature):
        normalized = signature if signature.startswith("0x") else f"0x{signature}"
        return normalized == _signature_for(self.ss58_address, message)


class TestChutesApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _auth_headers(
        self,
        *,
        validator_hotkey: str = VALID_VALIDATOR_HOTKEY,
        miner_hotkey: str = VALID_HOTKEY_SS58,
        timestamp: int = FIXED_NOW,
        signature: str | None = None,
    ) -> dict[str, str]:
        blob = AuthenticationPayload(
            validator_hotkey=validator_hotkey,
            miner_hotkey=miner_hotkey,
            timestamp=timestamp,
        ).blob_for_signing()
        return {
            "X-Validator-Hotkey": validator_hotkey,
            "X-Miner-Hotkey": miner_hotkey,
            "X-Timestamp": str(timestamp),
            "X-Signature": signature or _signature_for(validator_hotkey, blob),
        }

    @contextmanager
    def _patched_auth(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(settings, "MINER_HOTKEY_SS58_ADDRESS", VALID_HOTKEY_SS58)
            )
            stack.enter_context(
                patch("dependencies.auth.VALIDATOR_HOTKEY_SS58", VALID_VALIDATOR_HOTKEY)
            )
            stack.enter_context(
                patch("dependencies.auth.bittensor.Keypair", _SignedHeaderKeypair)
            )
            stack.enter_context(
                patch("dependencies.auth.time.time", return_value=FIXED_NOW)
            )
            yield

    def test_status_when_bridge_disabled(self):
        with patch.object(settings, "CHUTES_BRIDGE_ENABLED", False):
            response = self.client.get("/chutes/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["capability"]["bridge_enabled"])
        self.assertFalse(payload["capability"]["bridge_accessible"])

    def test_status_when_bridge_reachable(self):
        status_payload = {
            "ok": True,
            "capability": {
                "bridge_enabled": True,
                "bridge_accessible": True,
                "can_install": True,
            },
            "bridge": {"ok": True, "state": "not_installed"},
            "error": None,
        }
        with patch("routes.apis.ChutesRelayService.get_status_summary", return_value=status_payload):
            response = self.client.get("/chutes/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["bridge"]["state"], "not_installed")

    def test_install_happy_path(self):
        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.install",
                return_value={"ok": True, "state": "installed_stopped"},
            ) as install_mock,
        ):
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "installed_stopped")
        install_mock.assert_called_once_with(
            validator_hotkey=VALID_VALIDATOR_HOTKEY,
            hotkey_ss58=VALID_HOTKEY_SS58,
            hotkey_seed=NORMALIZED_SEED,
            node_name=VALID_NODE_NAME,
        )

    def test_install_requires_auth_headers(self):
        response = self.client.post(
            "/chutes/install",
            json={
                "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                "hotkey_ss58": VALID_HOTKEY_SS58,
                "hotkey_seed": VALID_SEED,
                "node_name": VALID_NODE_NAME,
            },
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing auth headers", response.json()["detail"])

    def test_install_rejects_wrong_validator_header(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(
                    validator_hotkey="5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVV"
                ),
            )
        self.assertEqual(response.status_code, 403)

    def test_install_rejects_wrong_miner_header(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(
                    miner_hotkey="5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVV"
                ),
            )
        self.assertEqual(response.status_code, 403)

    def test_install_rejects_stale_timestamp(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(timestamp=FIXED_NOW - 60),
            )
        self.assertEqual(response.status_code, 401)
        self.assertIn("too old", response.json()["detail"])

    def test_install_rejects_invalid_signature(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(signature="0xdeadbeef"),
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid signature")

    def test_install_rejects_body_validator_mismatch(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_HOTKEY_SS58,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("validator_hotkey", response.json()["detail"])

    def test_install_rejects_body_miner_mismatch(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_VALIDATOR_HOTKEY,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("hotkey_ss58", response.json()["detail"])

    def test_install_rejects_invalid_seed(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": "seed",
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 422)

    def test_install_rejects_invalid_validator_hotkey(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": "invalid-ss58",
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 422)

    def test_install_rejects_invalid_hotkey_ss58(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": "invalid-ss58",
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 422)

    def test_install_rejects_invalid_node_name(self):
        with self._patched_auth():
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": "-bad-node-name",
                },
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 422)

    def test_start_happy_path(self):
        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.start",
                return_value={"ok": True, "state": "running"},
            ),
        ):
            response = self.client.post(
                "/chutes/start",
                json={},
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "running")

    def test_start_bridge_failure_returns_500(self):
        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.start",
                side_effect=ChutesExecutorError("Agent pod did not become healthy within 3 min"),
            ),
        ):
            response = self.client.post(
                "/chutes/start",
                json={},
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Agent pod did not become healthy within 3 min",
        )

    def test_install_known_error_logs_without_traceback_or_secret(self):
        try:
            try:
                raise subprocess.TimeoutExpired(
                    cmd=["ssh", "bridgectl", "setup", "--hotkey-seed", "super-secret-seed"],
                    timeout=3600,
                )
            except subprocess.TimeoutExpired as exc:
                raise ChutesExecutorError(
                    "Chutes bridge command 'setup' timed out after 3600s"
                ) from exc
        except ChutesExecutorError as exc:
            chained_error = exc

        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.install",
                side_effect=chained_error,
            ),
            patch("routes.apis.logger.error") as logger_error,
        ):
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": VALID_VALIDATOR_HOTKEY,
                    "hotkey_ss58": VALID_HOTKEY_SS58,
                    "hotkey_seed": VALID_SEED,
                    "node_name": VALID_NODE_NAME,
                },
                headers=self._auth_headers(),
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Chutes bridge command 'setup' timed out after 3600s",
        )
        self.assertNotIn("exc_info", logger_error.call_args.kwargs)
        self.assertNotIn("super-secret-seed", str(logger_error.call_args))

    def test_start_unexpected_error_bubbles_to_fastapi(self):
        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.start",
                side_effect=TypeError("programming bug"),
            ),
        ):
            with self.assertRaisesRegex(TypeError, "programming bug"):
                self.client.post(
                    "/chutes/start",
                    json={},
                    headers=self._auth_headers(),
                )

    def test_stop_happy_path(self):
        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.stop",
                return_value={"ok": True, "state": "stopped"},
            ),
        ):
            response = self.client.post(
                "/chutes/stop",
                json={},
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "stopped")

    def test_chutes_routes_bypass_global_miner_middleware(self):
        with (
            self._patched_auth(),
            patch(
                "routes.apis.ChutesRelayService.start",
                return_value={"ok": True, "state": "running"},
            ),
        ):
            response = self.client.post(
                "/chutes/start",
                json={},
                headers=self._auth_headers(),
            )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
