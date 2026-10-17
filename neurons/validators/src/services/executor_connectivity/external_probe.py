import logging

import aiohttp
from services.executor_connectivity.models import PortPair

from core.config import settings

logger = logging.getLogger(__name__)

PROBER_TIMEOUT = aiohttp.ClientTimeout(total=20)


class ExternalPortProbe:
    """Fetches test ports from outside the validator's network (DAH-4537).

    A host that admits only the validator's address passes our own probe while every renter times
    out (ticket-0491). The prober is an AWS Lambda behind a function URL, so its source address is
    AWS's shared egress, not one a host can allow alone. Source: `external_prober/lambda_function.py`.
    """

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token

    async def reached(self, host: str, ports: list[PortPair], token: str) -> list[PortPair] | None:
        """The ports whose listener answered the outside fetch; None when the prober itself failed,
        which is never a verdict on the node."""
        body = {"targets": [{"host": host, "port": p.external} for p in ports], "timeout": 3}
        try:
            async with aiohttp.ClientSession(timeout=PROBER_TIMEOUT) as session:
                async with session.post(
                    self.url, json=body, headers={"Authorization": f"Bearer {self.token}"}
                ) as resp:
                    if resp.status != 200:
                        logger.warning("external port prober answered %s", resp.status)
                        return None
                    results = (await resp.json(content_type=None))["results"]
        except Exception as e:
            logger.warning("external port prober failed: %s", e)
            return None
        return [p for p, r in zip(ports, results) if (r.get("body") or "").strip() == f"{token}:{p.internal}"]


def external_port_probe() -> ExternalPortProbe | None:
    if not settings.EXTERNAL_PORT_PROBE_URL:
        return None
    return ExternalPortProbe(settings.EXTERNAL_PORT_PROBE_URL, settings.EXTERNAL_PORT_PROBE_TOKEN)
