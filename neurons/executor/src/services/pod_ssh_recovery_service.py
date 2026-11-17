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
RESTART_JUPYTER = ["sh", "-c", f"[ -f {JUPYTER_RESTART} ] || exit 0; exec sh {JUPYTER_RESTART}"]
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


def _watch_forever() -> None:
    while True:
        try:
            client = docker.from_env()
            for event in client.events(decode=True, filters={"type": "container", "event": "start"}):
                handle_event(event, client)
        except Exception as e:  # noqa: BLE001 - the stream drops with dockerd restarts; reconnect
            logger.warning(f"pod ssh recovery watcher reconnecting: {type(e).__name__}: {e}")
            time.sleep(5)


def start_pod_ssh_recovery_watcher() -> None:
    threading.Thread(target=_watch_forever, name="pod-ssh-recovery", daemon=True).start()
