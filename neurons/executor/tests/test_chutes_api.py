import os
import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ.setdefault(
    "MINER_HOTKEY_SS58_ADDRESS",
    "5D4jX4TqUkZwNwKAjjYrbk2FHFNN2U1TgFF6ZMuNPnjnKJVU",
)
os.environ.setdefault("DB_URI", "sqlite:///test.db")


class _DummyKeypair:
    def __init__(self, ss58_address=None):
        self.ss58_address = ss58_address

    def verify(self, *_args, **_kwargs):
        return False


sys.modules.setdefault("bittensor", types.SimpleNamespace(Keypair=_DummyKeypair))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.config import settings
from executor import app


class TestChutesApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

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
            patch.object(settings, "CHUTES_STAGING_AUTH_BYPASS", True),
            patch(
                "routes.apis.ChutesRelayService.install",
                return_value={"ok": True, "state": "installed_stopped"},
            ),
        ):
            response = self.client.post(
                "/chutes/install",
                json={
                    "validator_hotkey": "validator",
                    "hotkey_ss58": "hotkey",
                    "hotkey_seed": "seed",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "installed_stopped")

    def test_start_happy_path(self):
        with (
            patch.object(settings, "CHUTES_STAGING_AUTH_BYPASS", True),
            patch(
                "routes.apis.ChutesRelayService.start",
                return_value={"ok": True, "state": "running"},
            ),
        ):
            response = self.client.post("/chutes/start", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "running")

    def test_stop_happy_path(self):
        with (
            patch.object(settings, "CHUTES_STAGING_AUTH_BYPASS", True),
            patch(
                "routes.apis.ChutesRelayService.stop",
                return_value={"ok": True, "state": "stopped"},
            ),
        ):
            response = self.client.post("/chutes/stop", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "stopped")

    def test_auth_bypass_flag_controls_post_routes(self):
        with patch.object(settings, "CHUTES_STAGING_AUTH_BYPASS", False):
            response = self.client.post("/chutes/start", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
