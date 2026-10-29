"""Prepare the pinned loopback plugin without triggering old dockerd's digest-download panic."""

import shlex
from typing import NamedTuple

from core.config import settings
from core.docker_utils import ALPINE_HELPER_IMAGE
from services.const import LOOPBACK_PLUGIN_ALIAS, LOOPBACK_PLUGIN_V3_ALIAS


class LoopbackPluginBuild(NamedTuple):
    tag: str
    digest: str
    # names the plugin's DATA_DIR / STATE_DIR / lock on the host, so two generations never share a directory
    generation: str


LOOPBACK_PLUGIN_BUILDS = {
    LOOPBACK_PLUGIN_ALIAS: LoopbackPluginBuild(
        "daturaai/docker-volume-loopback:1.0.0-lium1",
        "sha256:99eacc306478ae5d03188474d07973078b66109077caa49618c57da211a71309",
        "v2",
    ),
    # DAH-5101: losetup --direct-io=on --sector-size 4096 --nooverlap, data files in whole sectors
    LOOPBACK_PLUGIN_V3_ALIAS: LoopbackPluginBuild(
        "daturaai/docker-volume-loopback:1.0.0-lium7",
        "sha256:5e297f3efcd2a331e67d006f8d1c1638e855f545d1ea14d2f7a02758cda9e631",
        "v3",
    ),
}


def current_loopback_plugin_alias() -> str:
    return LOOPBACK_PLUGIN_V3_ALIAS if settings.LOOPBACK_PLUGIN_V3_ENABLED else LOOPBACK_PLUGIN_ALIAS


def loopback_plugin_probe_marker_path(alias: str) -> str | None:
    # the setup script writes it once a probe volume mounted on this plugin; an enabled plugin
    # without it is not handed out (the probe may have been cut short, or the plugin enabled by
    # hand). None for v2 and the legacy plugin, which run no probe. Lives beside the lock, so an
    # executor restart costs one probe. Named after the pinned digest, so a newly pinned build is
    # probed again on an already probed node.
    if alias != LOOPBACK_PLUGIN_V3_ALIAS:
        return None
    build = LOOPBACK_PLUGIN_BUILDS[alias]
    digest = build.digest.replace(":", "-").replace("/", "-")
    return f"/run/lock/lium-loopback-{build.generation}-{digest}.probe-ok"


# Executed by the executor's Python, not the validator's filesystem. Docker's plugin metadata
# is on the host; the privileged executor has host PID access and can read it via /proc/1/root.
# No digest reference may reach `plugin install`: dockerd <25.0.3 can panic even with --disable.
_SETUP_SCRIPT = r'''
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

docker, alias, tag, digest, generation, probe_image, root, host_root, lock_path, marker_path = sys.argv[1:]
# the validator ends this script with `timeout -k 5`: its SIGTERM must still run the cleanup below
signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
data_dir = "/srv" + root + "/vloopback-" + generation
state_dir = "/srv/run/docker-volume-loopback-" + generation
plugins_dir = host_root + root + "/plugins"

def run_docker(*args):
    return subprocess.run([docker, *args], capture_output=True, text=True, check=False, timeout=120)

def call(*args):
    return run_docker("plugin", *args)

def mount_probe_volume():
    # install and enable mount nothing, and a host may refuse what the plugin asks of its loop
    # device (direct I/O, 4096-byte sectors): one probe volume is mounted before the plugin is
    # handed out, else it stays installed but disabled and the older plugin keeps renting.
    # 1g: the plugin's mkfs.xfs refuses anything under 300 MB; sparse, so a host with little
    # free disk is not refused by the reservation instead of by its loop device
    name = "lium-loopback-probe-" + generation
    try:
        created = run_docker("volume", "create", "-d", alias, "-o", "size=1g", "-o", "sparse=true", name)
        if created.returncode:
            raise RuntimeError("probe volume create failed: " + created.stderr.strip()[-300:])
        mounted = run_docker("run", "--rm", "--network", "none", "-v", name + ":/mnt", probe_image,
                             "sh", "-c", "echo probe > /mnt/probe && sync")
        if mounted.returncode:
            raise RuntimeError("probe volume mount failed: " + mounted.stderr.strip()[-300:])
    finally:
        run_docker("volume", "rm", "-f", name)

def disable_plugin(plugin_id):
    # the backend's health check may hold a volume on the plugin for a moment: retry
    for _ in range(3):
        if call("disable", plugin_id).returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("the plugin could not be disabled")

def inspect():
    result = call("inspect", alias)
    if result.returncode:
        if "no such plugin" in result.stderr.lower() or "not found" in result.stderr.lower():
            return None
        raise RuntimeError("plugin inspect failed")
    plugins = json.loads(result.stdout)
    if len(plugins) != 1:
        raise RuntimeError("plugin inspect returned an unexpected count")
    return plugins[0]

try:
    if not root.startswith("/") or ".." in root.split("/"):
        raise RuntimeError("invalid Docker data root")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plugin = inspect()
        if plugin is None:
            # The daemon may complete an install even if its client reports failure. Inspect
            # the actual result under the lock before deciding whether setup succeeded.
            call("install", tag, "--disable", "--alias", alias, "--grant-all-permissions",
                 "DATA_DIR=" + data_dir, "STATE_DIR=" + state_dir)
            plugin = inspect()
        if plugin is None:
            raise RuntimeError("installation did not produce a plugin")
        plugin_id = plugin["Id"]
        if len(plugin_id) != 64 or any(c not in "0123456789abcdef" for c in plugin_id):
            raise RuntimeError("invalid plugin ID")
        with open(plugins_dir + "/" + plugin_id + "/config.json") as source:
            saved = json.load(source)
        if saved.get("Manifest") != digest or saved.get("plugin", {}).get("Id") != plugin_id:
            raise RuntimeError("downloaded manifest does not match the pinned digest")
        env = plugin.get("Settings", {}).get("Env", [])
        for name, value in (("DATA_DIR", data_dir), ("STATE_DIR", state_dir)):
            if [entry for entry in env if entry.startswith(name + "=")] != [name + "=" + value]:
                raise RuntimeError("data/state directories do not match this Docker data root")
        if not plugin["Enabled"]:
            call("enable", plugin_id)
            plugin = inspect()
        if not plugin or plugin["Id"] != plugin_id or plugin["Enabled"] is not True:
            raise RuntimeError("verified plugin remains disabled")
        if probe_image and not os.path.exists(marker_path):
            # also for a plugin found enabled: an enabled plugin without the marker was never proven
            # to mount (a probe cut short by the validator's deadline, one enabled by hand)
            try:
                mount_probe_volume()
            except BaseException as error:
                reason = str(error) if isinstance(error, RuntimeError) else "probe volume failed: " + type(error).__name__
                try:
                    disable_plugin(plugin_id)
                except RuntimeError as disable_error:
                    reason += "; " + str(disable_error)
                raise RuntimeError(reason) from None
            open(marker_path, "w").close()
        print("verified plugin ready")
except Exception as error:
    # Do not echo a remote banner, registry credentials, or arbitrary filesystem contents.
    if isinstance(error, RuntimeError):
        print(str(error), file=sys.stderr)
    else:
        print("plugin verification failed: " + type(error).__name__, file=sys.stderr)
    sys.exit(1)
'''


def loopback_plugin_setup_command(docker_root_dir: str, alias: str) -> str:
    tag, digest, generation = LOOPBACK_PLUGIN_BUILDS[alias]
    # v2 is set up exactly as before; only the direct-I/O generation mounts a probe volume
    probe_image = "" if alias == LOOPBACK_PLUGIN_ALIAS else ALPINE_HELPER_IMAGE
    arguments = (
        "python3", "-c", _SETUP_SCRIPT, "/usr/bin/docker", alias,
        tag, digest, generation, probe_image, docker_root_dir.rstrip("/"),
        "/proc/1/root", f"/run/lock/lium-loopback-{generation}.lock",
        loopback_plugin_probe_marker_path(alias) or "",
    )
    return shlex.join(arguments)
