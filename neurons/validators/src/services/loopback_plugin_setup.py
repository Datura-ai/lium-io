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
        "daturaai/docker-volume-loopback:1.0.0-lium10",
        "sha256:8a780afe833dee55f60da5a5380dfcbc19c26a51688e7d7cd5353c0c8657897e",
        "v3",
    ),
}


def current_loopback_plugin_alias() -> str:
    return LOOPBACK_PLUGIN_V3_ALIAS if settings.LOOPBACK_PLUGIN_V3_ENABLED else LOOPBACK_PLUGIN_ALIAS


def loopback_plugin_probe_marker_path(alias: str) -> str | None:
    # the setup script writes it once a probe volume mounted on this plugin; an enabled plugin
    # without it is not handed out (the probe may have been cut short, or the plugin enabled by
    # hand). None for v2 and the legacy plugin, which run no probe. Lives beside the lock, so an
    # executor restart costs one probe. The path is read by the backend's health check too, so it
    # stays fixed; the file holds the pinned digest, so a newly pinned build is probed again.
    if alias != LOOPBACK_PLUGIN_V3_ALIAS:
        return None
    return f"/run/lock/lium-loopback-{LOOPBACK_PLUGIN_BUILDS[alias].generation}.probe-ok"


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
# on a plugin that was already enabled when the script started, a failed probe still disables it inside
# the validator's 150 s deadline: 2 leftover removals + the probe + 2 cleanup removals + the disable tries
# is 133 s. After a fresh install and enable (run_docker's 120 s cap each) less of the deadline is left and
# the disable tries can be cut; the SIGTERM handler below then says so on stderr
PROBE_SECONDS = 60
CLEANUP_SECONDS = 10
DISABLE_TRY_SECONDS = 10
DISABLE_TRIES = 3

def end_on_validator_deadline(*_):
    # the validator ends this script with `timeout -k 5`: the reason goes out before the cleanup
    # below runs, so its SIGKILL cannot leave stderr empty
    print("setup cut short by the validator's deadline", file=sys.stderr, flush=True)
    sys.exit(1)

signal.signal(signal.SIGTERM, end_on_validator_deadline)
data_dir = "/srv" + root + "/vloopback-" + generation
state_dir = "/srv/run/docker-volume-loopback-" + generation
plugins_dir = host_root + root + "/plugins"

def run_docker(*args, timeout=120):
    return subprocess.run([docker, *args], capture_output=True, text=True, check=False, timeout=timeout)

def call(*args):
    return run_docker("plugin", *args)

def probe_marker_holds(expected_digest):
    try:
        with open(marker_path) as marker:
            return marker.read().strip() == expected_digest
    except (OSError, UnicodeError):
        # a marker the host cannot even decode is as stale as one of another build: drop it
        return False

# one volume and one container per generation: the lock serialises setups, so a stable name lets
# the next setup reclaim what a killed one left (a `created` container never exits, `--rm` or not)
probe_volume = "lium-loopback-probe-" + generation
probe_container = probe_volume

def remove_probe_leftovers():
    # the container first: a `docker run` cut off by its timeout leaves it running with the volume
    # attached. Returns what stayed behind
    leftovers = []
    removed = run_docker("rm", "-f", probe_container, timeout=CLEANUP_SECONDS)
    if removed.returncode and "no such container" not in removed.stderr.lower():
        leftovers.append("container " + probe_container)
    if run_docker("volume", "rm", "-f", probe_volume, timeout=CLEANUP_SECONDS).returncode:
        leftovers.append("volume " + probe_volume)
    return leftovers

def mount_probe_volume():
    # install and enable mount nothing, and a host may refuse what the plugin asks of its loop
    # device (direct I/O, 4096-byte sectors): one probe volume is mounted before the plugin is
    # handed out, else it stays installed but disabled and the older plugin keeps renting.
    # 1g: the plugin's mkfs.xfs refuses anything under 300 MB; sparse, so a host with little
    # free disk is not refused by the reservation instead of by its loop device
    leftovers = remove_probe_leftovers()
    if leftovers:
        raise RuntimeError("an earlier probe left " + ", ".join(leftovers))
    deadline = time.monotonic() + PROBE_SECONDS
    try:
        created = run_docker("volume", "create", "-d", alias, "-o", "size=1g", "-o", "sparse=true", probe_volume,
                             timeout=deadline - time.monotonic())
        if created.returncode:
            raise RuntimeError("probe volume create failed: " + created.stderr.strip()[-300:])
        mounted = run_docker("run", "--rm", "--name", probe_container, "--network", "none",
                             "-v", probe_volume + ":/mnt", probe_image, "sh", "-c", "echo probe > /mnt/probe && sync /mnt/probe",
                             timeout=max(1, deadline - time.monotonic()))
        if mounted.returncode:
            raise RuntimeError("probe volume mount failed: " + mounted.stderr.strip()[-300:])
    except BaseException:
        remove_probe_leftovers()
        raise
    leftovers = remove_probe_leftovers()
    if leftovers:
        # a volume still in use means the mount did not end cleanly: not a proof
        raise RuntimeError("probe cleanup left " + ", ".join(leftovers))

def disable_plugin(plugin_id):
    # the backend's health check may hold a volume on the plugin for a moment: retry. Short tries,
    # so a hung daemon still ends them inside the validator's deadline with this reason on stderr
    for _ in range(DISABLE_TRIES):
        try:
            if run_docker("plugin", "disable", plugin_id, timeout=DISABLE_TRY_SECONDS).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1)
    raise RuntimeError("the plugin could not be disabled")

def read_saved_config(plugin_id):
    if len(plugin_id) != 64 or any(c not in "0123456789abcdef" for c in plugin_id):
        raise RuntimeError("invalid plugin ID")
    with open(plugins_dir + "/" + plugin_id + "/config.json") as source:
        return json.load(source)

def remove_plugin_of_another_build(plugin):
    # Docker refuses to disable or remove a plugin its volumes still use; the listing only makes
    # that refusal readable. A probe a killed setup left on that build is reclaimed first: its
    # cleanup otherwise runs only after this replacement, and the host would never move on
    leftovers = remove_probe_leftovers()
    if leftovers:
        raise RuntimeError("an earlier probe left " + ", ".join(leftovers))
    listed = run_docker("volume", "ls", "--format", "{{.Driver}}")
    if listed.returncode:
        raise RuntimeError("cannot list the volumes of an earlier build")
    if alias in listed.stdout.split():
        raise RuntimeError("an earlier build of the plugin still owns volumes")
    if plugin["Enabled"]:
        disable_plugin(plugin["Id"])
    if call("rm", plugin["Id"]).returncode:
        raise RuntimeError("the earlier build of the plugin could not be removed")

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
        needs_probe = bool(probe_image) and not probe_marker_holds(digest)
        if needs_probe and os.path.exists(marker_path):
            # the backend's health check tests the marker's existence only: a marker of another
            # build goes before anything below can fail, so both sides see the same fallback
            os.remove(marker_path)
        plugin = inspect()
        if plugin is not None and probe_image and read_saved_config(plugin["Id"]).get("Manifest") != digest:
            # an earlier pinned build under this alias (lium7 once lium8 is pinned) is replaced once
            # no volume uses it, else every setup refuses its manifest and the host never moves on
            remove_plugin_of_another_build(plugin)
            plugin = None
        if plugin is None:
            # The daemon may complete an install even if its client reports failure. Inspect
            # the actual result under the lock before deciding whether setup succeeded.
            call("install", tag, "--disable", "--alias", alias, "--grant-all-permissions",
                 "DATA_DIR=" + data_dir, "STATE_DIR=" + state_dir)
            plugin = inspect()
        if plugin is None:
            raise RuntimeError("installation did not produce a plugin")
        plugin_id = plugin["Id"]
        saved = read_saved_config(plugin_id)
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
        if needs_probe:
            # also for a plugin found enabled: an enabled plugin without the marker was never proven
            # to mount (a probe cut short by the validator's deadline, one enabled by hand); a marker
            # of another build means this one was never probed here
            try:
                mount_probe_volume()
            except BaseException as error:
                reason = str(error) if isinstance(error, RuntimeError) else "probe volume failed: " + type(error).__name__
                try:
                    disable_plugin(plugin_id)
                except RuntimeError as disable_error:
                    reason += "; " + str(disable_error)
                raise RuntimeError(reason) from None
            with open(marker_path, "w") as marker:
                marker.write(digest)
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
