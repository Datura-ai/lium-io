import json
import logging
import subprocess

import requests

from vast_api.config import VastSettings
from vast_api.errors import ApiFailure

logger = logging.getLogger(__name__)

VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastClient:
    """Vast.ai control plane: REST for reads, the vastai CLI for market ops.

    The account key is read from VAST_ACCOUNT_KEY_FILE and must never be logged.
    It travels only in the Authorization header (never in the URL), and requests'
    own exception text (which embeds the full request URL) never propagates.
    """

    def __init__(self, settings: VastSettings):
        self.settings = settings

    def _account_key(self) -> str:
        with open(self.settings.VAST_ACCOUNT_KEY_FILE) as f:
            return f.read().strip()

    def _get(self, path: str) -> requests.Response:
        # key-safe GET: key in header only; network errors reduced to the exception type
        try:
            return requests.get(
                f"{VAST_API_BASE}{path}",
                headers={"Authorization": f"Bearer {self._account_key()}"},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ApiFailure("vast_api_failed", f"GET {path} failed: {type(exc).__name__}")

    def _cli(self, args: list[str], timeout: int = 600) -> str:
        # run one vastai CLI command; the key never reaches logs or error text
        cmd = ["vastai"] + args + ["--api-key", self._account_key()]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ApiFailure("vast_cli_failed", f"vastai {' '.join(args)} timed out after {timeout}s")
        if result.returncode != 0:
            raise ApiFailure(
                "vast_cli_failed",
                f"vastai {' '.join(args)} rc={result.returncode}: {result.stderr.strip()[:500]}",
            )
        return result.stdout

    def get_machines(self) -> list[dict]:
        response = self._get("/machines/")
        if not response.ok:
            raise ApiFailure("vast_api_failed", f"GET /machines/ HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise ApiFailure("vast_api_failed", "GET /machines/ returned non-JSON")
        return data.get("machines", data) if isinstance(data, dict) else data

    def mint_machine_key(self) -> str:
        # MK rotates on every call — mint right before use, never cache
        response = self._get("/auth/machine_key/")
        if not response.ok:
            raise ApiFailure("vast_api_failed", f"GET /auth/machine_key/ HTTP {response.status_code}")
        return response.text.strip().strip('"')

    def identify(self, machine_key: str, machine_api_key_hex: str) -> dict:
        # direct daemon identify; the response may say success:false with the id still created
        try:
            response = requests.post(
                f"{VAST_API_BASE}/daemon/identify/",
                headers={"Authorization": f"Bearer {machine_key}"},
                json={"machine_api_key": machine_api_key_hex},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ApiFailure("identify_rejected", f"identify request failed: {type(exc).__name__}")
        try:
            return response.json()
        except ValueError:
            raise ApiFailure(
                "identify_rejected", f"identify returned {response.status_code}: {response.text[:200]}"
            )

    def search_offers(self, machine_id: int) -> list[dict]:
        # the three any-flags are load-bearing: a bare search hides unverified machines
        output = self._cli(
            ["search", "offers", f"machine_id={machine_id} rentable=any rented=any verified=any", "--raw"]
        )
        try:
            return json.loads(output)
        except ValueError:
            raise ApiFailure("vast_cli_failed", f"unparseable search offers output: {output[:200]}")

    def unlist_machine(self, machine_id: int) -> None:
        self._cli(["unlist", "machine", str(machine_id)])

    def list_machine(self, machine_id: int, price_gpu_usd: float, duration: str) -> None:
        # NO --vol_size: disproven cargo-cult (SUCCESS-PATH step 9)
        self._cli([
            "list", "machine", str(machine_id),
            "--price_gpu", str(price_gpu_usd),
            "--price_disk", "0.02",
            "--duration", duration,
        ])

    def self_test_machine(self, machine_id: int) -> str:
        return self._cli(["self-test", "machine", str(machine_id), "--raw", "--debugging"], timeout=1800)

    def delete_machine(self, machine_id: int) -> None:
        self._cli(["delete", "machine", str(machine_id)])
