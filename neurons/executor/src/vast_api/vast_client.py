import json
import logging
import subprocess
from pathlib import Path

import requests

from vast_api.config import VastSettings
from vast_api.errors import ApiFailure
from vast_api.host_ops import HostOps

logger = logging.getLogger(__name__)

VAST_API_BASE = "https://console.vast.ai/api/v0"
# proven listing disk price ($/GB/hr) — a constant, not a knob anyone tunes
LISTING_DISK_PRICE_USD = "0.02"


class VastClient:
    """Vast.ai control plane: REST for reads, the vastai CLI for market ops.

    The account key is read from VAST_ACCOUNT_KEY_FILE and must never be logged.
    It travels only in the Authorization header (never in the URL), and requests'
    own exception text (which embeds the full request URL) never propagates.
    """

    def __init__(self, settings: VastSettings, host: HostOps):
        self.settings = settings
        self.host = host

    def _account_key(self) -> str:
        # the key lives on the HOST filesystem (placed there at enrollment), so it
        # survives executor container recreation; read via nsenter, never logged
        result = self.host.run(["cat", self.settings.VAST_ACCOUNT_KEY_FILE], check=False)
        key = result.stdout.strip()
        if result.returncode != 0 or not key:
            raise ApiFailure(
                "vast_key_missing",
                f"no account key at {self.settings.VAST_ACCOUNT_KEY_FILE} on the host",
            )
        return key

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

    def _ensure_cli_key_file(self) -> None:
        # the CLI reads ~/.vast_api_key on its own; --api-key in argv would be
        # world-readable via /proc/<pid>/cmdline under --pid host
        key = self._account_key()
        path = Path.home() / ".vast_api_key"
        if path.exists() and path.read_text().strip() == key:
            return
        path.touch(mode=0o600)
        path.chmod(0o600)  # touch keeps an existing file's mode — force it
        path.write_text(key + "\n")

    def _cli(self, args: list[str], timeout: int = 600) -> str:
        # run one vastai CLI command; the key never reaches logs, error text, or argv
        self._ensure_cli_key_file()
        cmd = ["vastai"] + args
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
        machines = data.get("machines") if isinstance(data, dict) else data
        if not isinstance(machines, list):
            raise ApiFailure("vast_api_failed", "unexpected /machines/ response shape")
        return machines

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
            "--price_disk", LISTING_DISK_PRICE_USD,
            "--duration", duration,
        ])

    def self_test_machine(self, machine_id: int) -> str:
        return self._cli(["self-test", "machine", str(machine_id), "--raw", "--debugging"], timeout=1800)

    def delete_machine(self, machine_id: int) -> None:
        self._cli(["delete", "machine", str(machine_id)])
