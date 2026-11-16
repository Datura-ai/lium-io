"""DAH-6379: a rental pod's memory.oom.group=1 makes an OOM kill take the whole pod. Docker gives a restarted pod a
new cgroup with it back at 0, so it is set on every pod start here (the executor runs privileged in the host PID
namespace); the validator also sets it on create and on each rented-pod check."""

import threading
import time

import docker

from core.logger import get_logger

logger = get_logger(__name__)

POD_PREFIX = "pod_"
CGROUP_DIRS = (
    "/proc/1/root/sys/fs/cgroup/system.slice/docker-{id}.scope",  # systemd cgroup driver
    "/proc/1/root/sys/fs/cgroup/docker/{id}",  # cgroupfs driver (the documented daemon.json)
)
# Written by the validator's create; absent on a pod's first start, so only a restart re-runs it. A restart reruns
# only the image CMD, so the sshd the validator started with `docker exec` (lost to an OOM group kill) needs this.
# The script adopts a running sshd and locks against a concurrent run, so a validator re-bootstrap is harmless.
SSH_BOOTSTRAP = "/tmp/lium-ssh-bootstrap.sh"
RESTART_SSHD = ["sh", "-c", f"[ -f {SSH_BOOTSTRAP} ] || exit 0; exec sh {SSH_BOOTSTRAP}"]


def set_oom_group(container_id: str, dirs: tuple[str, ...] = CGROUP_DIRS) -> str:
    """Returns "set", "already" or the reason it could not."""
    for d in dirs:
        path = f"{d.format(id=container_id)}/memory.oom.group"
        try:
            with open(path) as f:
                if f.read().strip() == "1":
                    return "already"
            with open(path, "w") as f:
                f.write("1")
            return "set"
        except FileNotFoundError:
            continue
        except OSError as e:
            return f"write_failed {type(e).__name__}"
    return "no_cgroup_v2_scope"


CIPHER_MOUNT = "/lium-cipher"  # an encrypted pod's gocryptfs volume, as the validator and storage/workspace.py read it


def restart_sshd(client, container_id: str) -> None:
    try:
        container = client.containers.get(container_id)
        # SSH before the validator remounts gocryptfs would let renter writes land unencrypted on the host disk;
        # the validator's mount-then-SSH recovery restores an encrypted pod.
        if any(m.get("Destination") == CIPHER_MOUNT for m in container.attrs.get("Mounts") or []):
            return
        container.exec_run(RESTART_SSHD, detach=True, user="0")  # a non-root image USER cannot start sshd
    except Exception as e:  # noqa: BLE001 - a pod gone again or a dockerd hiccup must not stop the watcher
        logger.warning(f"POD_SSHD_RESTART_FAILED {container_id[:12]} {type(e).__name__}: {e}")


def handle_event(event: dict, dirs: tuple[str, ...] = CGROUP_DIRS, client=None) -> str | None:
    actor = event.get("Actor") or {}
    name = (actor.get("Attributes") or {}).get("name") or ""
    if event.get("Type") != "container" or event.get("Action") != "start" or not name.startswith(POD_PREFIX):
        return None
    container_id = event.get("id") or actor.get("ID") or ""
    result = set_oom_group(container_id, dirs)
    if result == "set":
        logger.info(f"POD_OOM_GROUP_SET {name} {container_id[:12]}")
    elif result != "already":
        logger.warning(f"POD_OOM_GROUP_NOT_SET {name} {container_id[:12]} {result}")
    if client is not None:
        restart_sshd(client, container_id)
    return result


def _watch_forever() -> None:
    while True:
        try:
            client = docker.from_env()
            for event in client.events(decode=True, filters={"type": "container", "event": "start"}):
                handle_event(event, client=client)
        except Exception as e:  # noqa: BLE001 - the stream drops with dockerd restarts; reconnect
            logger.warning(f"pod oom.group watcher reconnecting: {type(e).__name__}: {e}")
            time.sleep(5)


def start_pod_oom_group_watcher() -> None:
    threading.Thread(target=_watch_forever, name="pod-oom-group", daemon=True).start()
