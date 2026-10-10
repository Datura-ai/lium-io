import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

import requests

from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.host_ops import HostOps

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
DELIVERY_TIMEOUT_SECONDS = 10

# 4xx that the backend can answer again with a 2xx: request timeout, rate limit
RETRYABLE_STATUSES = (408, 429)

# both files live under STATE_DIR_HOST, next to the machine identity, so they
# survive executor-container restarts
EVENTS_CONFIG_FILE = "events_config.json"
EVENTS_STATE_FILE = "contract_events.json"


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ContractEventsPoller:
    """Reports Vast contract start/end edges to the backend (plan-contract-events).

    A live Vast rental IS a C.<contract_id> container in the nested dockerd, so
    the rental window is bounded by the create/destroy of that NAME: the poller
    diffs the current C.* set against the persisted last-seen set and posts one
    event per edge. The last-seen set is persisted BEFORE delivery and
    undelivered events sit in a persisted spool, so restarts neither re-emit
    edges nor lose events (at-least-once; the backend dedups by event_id).
    """

    def __init__(self, settings: VastSettings, host: HostOps, docker_ops: DockerOps):
        self.settings = settings
        self.host = host
        self.docker_ops = docker_ops

    async def run_forever(self) -> None:
        # lifespan background task: one blocking cycle per interval, off-loop
        while True:
            try:
                await asyncio.to_thread(self.poll_once)
            except Exception:
                logger.warning("contract events cycle failed", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def poll_once(self) -> None:
        # one edge-detection + delivery cycle; silent no-op until the machine is
        # enrolled AND the setup body delivered the events config
        if self.docker_ops.get_vast_uns() is None:
            return
        config = self._config()
        if config is None:
            return
        current = set(self.docker_ops.vast_rental_containers())
        state = self._read_json(EVENTS_STATE_FILE) or {}
        last_seen = set(state.get("last_seen", []))
        pending = list(state.get("pending", []))
        edges = [(name, "start") for name in sorted(current - last_seen)]
        edges += [(name, "end") for name in sorted(last_seen - current)]
        if edges:
            machine_id = self._machine_id()
            for name, kind in edges:
                pending.append(
                    {
                        "event_id": str(uuid.uuid4()),
                        "occurred_at": _utcnow_iso(),
                        "executor_id": config.get("executor_id"),
                        "vast_machine_id": machine_id,
                        "contract_id": name,
                        "event": kind,
                    }
                )
            # persist the observed set BEFORE delivery: a crash mid-delivery must
            # not re-emit the same edges — the spool carries them instead
            self._write_state(sorted(current), pending)
        if not pending:
            return
        remaining = self._deliver(config, pending)
        if remaining != pending:
            self._write_state(sorted(current), remaining)

    # --- delivery ---

    def _deliver(self, config: dict, pending: list[dict]) -> list[dict]:
        # at-least-once POSTs: 2xx = delivered, permanent 4xx = misconfiguration →
        # drop loudly (must not loop forever), network/5xx and the retryable 4xx
        # (408, 429) → keep for the next cycle
        remaining = []
        for event in pending:
            try:
                response = requests.post(
                    config["events_url"],
                    headers={"Authorization": f"Bearer {config['events_token']}"},
                    json=event,
                    timeout=DELIVERY_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "contract event %s delivery failed (%s), will retry",
                    event["event_id"],
                    type(exc).__name__,
                )
                remaining.append(event)
                continue
            if 200 <= response.status_code < 300:
                continue
            if (
                400 <= response.status_code < 500
                and response.status_code not in RETRYABLE_STATUSES
            ):
                logger.error(
                    "contract event %s rejected with %s: %s — dropping",
                    event["event_id"],
                    response.status_code,
                    response.text[:300],
                )
                continue
            remaining.append(event)
        return remaining

    # --- persisted state (files under STATE_DIR_HOST) ---

    def _read_json(self, name: str) -> dict | None:
        content = self.host.read_state_file(name)
        if content is None:
            return None
        try:
            return json.loads(content)
        except ValueError:
            logger.warning("corrupt state file %s, ignoring", name)
            return None

    def _config(self) -> dict | None:
        # delivery config persisted by POST /vast/setup; None until the backend
        # sends events_url + events_token
        config = self._read_json(EVENTS_CONFIG_FILE) or {}
        if config.get("events_url") and config.get("events_token"):
            return config
        return None

    def _machine_id(self) -> int | None:
        # same identity file the register stage persists (numeric_machine_id)
        content = self.host.read_state_file("numeric_machine_id")
        if content is not None and content.strip().isdigit():
            return int(content.strip())
        return None

    def _write_state(self, last_seen: list[str], pending: list[dict]) -> None:
        self.host.write_state_file(
            EVENTS_STATE_FILE, json.dumps({"last_seen": last_seen, "pending": pending})
        )
