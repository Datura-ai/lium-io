import requests

from vast_api.errors import ApiFailure

VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastClient:
    """Daemon identify — the only Vast call the box makes (plan-key-split).

    The machine_key arrives from the backend inside the signed setup body and
    rotates on every mint; the account key never touches the box. All market
    operations (list/unlist/price/self-test/delete-record) are backend-side.
    """

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
