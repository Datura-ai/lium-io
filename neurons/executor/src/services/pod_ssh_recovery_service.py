"""DAH-6379: a restarted rental pod gets its sshd and platform Jupyter back. A restart reruns only the image CMD,
so the sshd and Jupyter the validator started with `docker exec` are gone; the executor (privileged, host PID
namespace) re-runs the validator's SSH bootstrap and the Jupyter launch on every pod start."""

import threading
import time

import docker

from core.logger import get_logger

logger = get_logger(__name__)

POD_PREFIX = "pod_"
# Written by the validator's create; absent on a pod's first start, so only a restart re-runs it.
# The script adopts a running sshd and locks against a concurrent run, so a validator re-bootstrap is harmless.
SSH_BOOTSTRAP = "/tmp/lium-ssh-bootstrap.sh"
RESTART_SSHD = ["sh", "-c", f"[ -f {SSH_BOOTSTRAP} ] || exit 0; exec sh {SSH_BOOTSTRAP}"]
# Written by run_jupyter.sh when the validator installs Jupyter, with the same token and port, so the link still works
JUPYTER_RESTART = "/tmp/lium-jupyter-restart.sh"
# a live Jupyter (the startup sweep reaches pods that never stopped) is left alone; [j] keeps grep off its own cmdline
RESTART_JUPYTER = [
    "sh",
    "-c",
    f"[ -f {JUPYTER_RESTART} ] || exit 0; grep -qs '/opt/jupyter-env/bin/[j]upyter' /proc/[0-9]*/cmdline && exit 0; "
    f"exec sh {JUPYTER_RESTART}",
]
CIPHER_MOUNT = "/lium-cipher"  # an encrypted pod's gocryptfs volume, as the validator and storage/workspace.py read it


def restart_sshd(client, container_id: str) -> None:
    try:
        container = client.containers.get(container_id)
        # SSH before the validator remounts gocryptfs would let renter writes land unencrypted on the host disk;
        # the validator's mount-then-SSH recovery restores an encrypted pod.
        if any(m.get("Destination") == CIPHER_MOUNT for m in container.attrs.get("Mounts") or []):
            return
        container.exec_run(RESTART_SSHD, detach=True, user="0")  # a non-root image USER cannot start sshd
        container.exec_run(RESTART_JUPYTER, detach=True, user="0")
    except Exception as e:  # noqa: BLE001 - a pod gone again or a dockerd hiccup must not stop the watcher
        logger.warning(f"POD_SSHD_RESTART_FAILED {container_id[:12]} {type(e).__name__}: {e}")


def handle_event(event: dict, client) -> bool:
    """True when the event is a rental pod start and its SSH bootstrap was handled."""
    actor = event.get("Actor") or {}
    name = (actor.get("Attributes") or {}).get("name") or ""
    if event.get("Type") != "container" or event.get("Action") != "start" or not name.startswith(POD_PREFIX):
        return False
    restart_sshd(client, event.get("id") or actor.get("ID") or "")
    return True


def sweep_running_pods(client) -> None:
    """A host or dockerd restart starts pods before the executor's watcher runs, so their starts predate its cursor."""
    for container in client.containers.list(filters={"name": POD_PREFIX, "status": "running"}):
        if container.name.startswith(POD_PREFIX):  # the name filter is a substring match
            restart_sshd(client, container.id)


def _watch_forever() -> None:
    # a reconnect resumes from the last event's second, so a pod restarted while the stream was down is still
    # recovered; the events of that second it replays are skipped by (id, timeNano). The cursor starts at the
    # watcher's start, so a drop before the first event still resumes from there, not from "now"
    since = int(time.time())
    seen: set = set()
    swept = False
    while True:
        try:
            client = docker.from_env()
            if not swept:
                sweep_running_pods(client)
                swept = True
            for event in client.events(decode=True, since=since, filters={"type": "container", "event": "start"}):
                t = event.get("time")
                if t != since:
                    since, seen = t, set()
                key = (event.get("id"), event.get("timeNano"))
                if key in seen:
                    continue
                seen.add(key)
                handle_event(event, client)
        except Exception as e:  # noqa: BLE001 - the stream drops with dockerd restarts; reconnect
            logger.warning(f"pod ssh recovery watcher reconnecting: {type(e).__name__}: {e}")
            time.sleep(5)


def start_pod_ssh_recovery_watcher() -> None:
    threading.Thread(target=_watch_forever, name="pod-ssh-recovery", daemon=True).start()
