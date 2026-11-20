"""DAH-6855: the executor keeps the rental egress rules (datura.rental_egress) in place itself.

They are applied when the executor starts, when the rental network is created, when a rental container starts and
every APPLY_INTERVAL_SECONDS, with the same scripts, chains and host lock the validator uses over SSH. So they come
back with the rentals after a reboot without waiting for a validator to reach the host, and `/version` advertises
RENTAL_EGRESS_CAPABILITY so the validator only reads them back. Fail-open as on the validator side: a host without a
DOCKER-USER chain or without iptables logs it and keeps running.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import docker
from core.logger import get_logger
from datura import rental_egress as rules

logger = get_logger(__name__)

# a failed apply (dockerd still starting after a reboot, a busy xtables lock) is tried again this soon
RETRY_SECONDS = 30


async def run_local(command: str):
    """Run one of the shared commands in this container, as the validator's SSH session would."""
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), rules.HELPER_TIMEOUT_SECONDS)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return SimpleNamespace(exit_status=124, stdout="", stderr="timed out")
    return SimpleNamespace(
        exit_status=proc.returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
    )


def apply_once() -> bool:
    """Reconcile the rules to what the host runs now. False when it should be tried again soon."""
    try:
        guarded, error = asyncio.run(rules.apply_rules(run_local))
    except rules.InventoryUnknown as exc:
        logger.warning(f"RENTAL_EGRESS_INVENTORY_UNKNOWN rules left as they are: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 - fail-open: a hardening rule never stops the executor
        logger.warning(f"RENTAL_EGRESS_NOT_APPLIED {type(exc).__name__}: {exc}")
        return False
    if error is None:
        logger.info(f"RENTAL_EGRESS_APPLIED {guarded or 'nothing to guard'}")
        return True
    logger.warning(f"RENTAL_EGRESS_NOT_APPLIED {error}")
    return rules.NO_DOCKER_USER in error


def is_rental_event(event: dict) -> bool:
    """The rental network created, or a rental container started."""
    name = ((event.get("Actor") or {}).get("Attributes") or {}).get("name") or ""
    if event.get("Type") == "network":
        return event.get("Action") == "create" and name == rules.RENTAL_NETWORK_NAME
    return (
        event.get("Type") == "container"
        and event.get("Action") == "start"
        and rules.RENTAL_CONTAINER_NAME.fullmatch(name) is not None
    )


def _watch_forever(due: threading.Event) -> None:
    # resumes from the last event's second after a dropped stream, so a rental started meanwhile is not missed
    since = int(time.time())
    while True:
        try:
            client = docker.from_env()
            for event in client.events(
                decode=True,
                since=since,
                filters={"type": ["container", "network"], "event": ["start", "create"]},
            ):
                since = event.get("time") or since
                if is_rental_event(event):
                    due.set()
        except Exception as e:  # noqa: BLE001 - the stream drops with dockerd restarts; reconnect
            logger.warning(f"rental egress watcher reconnecting: {type(e).__name__}: {e}")
        due.set()  # a reconnect may follow a dockerd restart: reconcile once more
        time.sleep(5)


def _apply_forever(due: threading.Event) -> None:
    wait = rules.APPLY_INTERVAL_SECONDS
    while True:
        due.wait(timeout=wait)
        due.clear()
        wait = rules.APPLY_INTERVAL_SECONDS if apply_once() else RETRY_SECONDS


def start_rental_egress_rules() -> None:
    due = threading.Event()
    due.set()  # at start: a reboot brings rentals back before any validator reaches the host
    threading.Thread(target=_watch_forever, args=(due,), name="rental-egress-watch", daemon=True).start()
    threading.Thread(target=_apply_forever, args=(due,), name="rental-egress-apply", daemon=True).start()
