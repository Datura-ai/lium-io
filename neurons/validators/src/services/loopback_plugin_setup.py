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
        "daturaai/docker-volume-loopback:1.0.0-lium11",
        "sha256:30a4244acaea828ca4132e8263d10d95ea746fda2ca6d613519a7ba1df42fd34",
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
    # stays fixed; the file holds the pinned digest and the plugin's ID and settings, so a newly
    # pinned build, a reinstalled plugin or one moved to another data directory is probed again.
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
import re
import signal
import subprocess
import sys
import time

(docker, alias, tag, digest, generation, probe_image, deadline_seconds, root, host_root, lock_path,
 marker_path) = sys.argv[1:]
validator_deadline = time.monotonic() + float(deadline_seconds)
PROBE_SECONDS = 60
CLEANUP_SECONDS = 10
DISABLE_TRY_SECONDS = 10
DISABLE_TRIES = 3
# the shell and Python start before the clock above, so the validator's `timeout` may fire this much
# sooner; it also covers the 1 s timeout floors
START_SECONDS = 5
# the probe ends this long before the validator's deadline, however long the lock, install and enable
# took: the removals before and after it, the volume listing before a disable, every disable try with
# its pause, and START_SECONDS
PROBE_RESERVE_SECONDS = 5 * CLEANUP_SECONDS + DISABLE_TRIES * (DISABLE_TRY_SECONDS + 1) + START_SECONDS
TOO_LITTLE_TIME_TO_PROBE = "too little of the validator's deadline left to probe"
# the validator's state check prints the same: the ID changes with every install, DATA_DIR with `plugin set`
INSTALLATION_FORMAT = "{{.Id}} {{json .Settings.Env}}"

# the docker call running now, so a reason names the step that hung or failed
step = "start"

class DockerTimeout(RuntimeError):
    pass

def end_on_validator_deadline(*_):
    # the validator ends this script with `timeout -k 5`: the reason goes out before the cleanup
    # below runs, so its SIGKILL cannot leave stderr empty
    print("setup cut short by the validator's deadline during " + step, file=sys.stderr, flush=True)
    sys.exit(1)

def without_credentials(url):
    # user info ends at the URL's last "@" (a password may hold "@" or "/" unencoded); the query (a signed URL's
    # signature or token) is a credential too
    return "://" + re.split(r"[?#]", url.group(1).rpartition("@")[2], maxsplit=1)[0]

def stderr_of(result):
    # docker's own message on one line, each URL cut to scheme, host and path
    text = re.sub(r"://([^\s\"']*)", without_credentials, result.stderr or "").strip()[-300:]
    return ": " + " ".join(text.split()) if text else ""

signal.signal(signal.SIGTERM, end_on_validator_deadline)
data_dir = "/srv" + root + "/vloopback-" + generation
state_dir = "/srv/run/docker-volume-loopback-" + generation
plugins_dir = host_root + root + "/plugins"
host_data_dir = host_root + root + "/vloopback-" + generation

def seconds_before_probe_reserve():
    return validator_deadline - PROBE_RESERVE_SECONDS - time.monotonic()

def run_docker(*args, timeout=None):
    global step
    called = step = "docker " + " ".join(args[:2])
    try:
        return run_docker_step(args, timeout)
    finally:
        # a call that returned is no longer where setup waits: a later host read or write names itself
        step = "the step after " + called

def run_docker_step(args, timeout):
    if timeout is None and probe_image:
        # a probed setup's calls before the probe end where its reserve begins: however long the lock
        # and they took, a failed probe still has the time to clean up and disable the plugin
        try:
            return subprocess.run([docker, *args], capture_output=True, text=True, check=False,
                                  timeout=max(1, seconds_before_probe_reserve()))
        except subprocess.TimeoutExpired:
            raise RuntimeError(TOO_LITTLE_TIME_TO_PROBE + " (" + step + " timed out)") from None
    try:
        return subprocess.run([docker, *args], capture_output=True, text=True, check=False,
                              timeout=120 if timeout is None else timeout)
    except subprocess.TimeoutExpired:
        raise DockerTimeout(step + " timed out") from None

def call(*args):
    return run_docker("plugin", *args)

def read_probe_marker():
    global step
    step = "the read of the probe marker"
    try:
        with open(marker_path) as marker:
            return marker.read().strip()
    except (OSError, UnicodeError):
        # a marker the host cannot even decode is as stale as one of another build: drop it
        return ""

def remove_probe_marker():
    if os.path.exists(marker_path):
        os.remove(marker_path)

# one volume and one container per generation: the lock serialises setups, so a stable name lets
# the next setup reclaim what a killed one left (a `created` container never exits, `--rm` or not)
probe_volume = "lium-loopback-probe-" + generation
probe_container = probe_volume

# what the last cleanup could not remove: the plugin keeps the probe's mount lease until Docker's unmount
# reaches it, and a plugin turned off before that keeps the lease for good, so no later probe can remove
# its volume (DAH-7237)
probe_left = []

def remove_probe_leftovers():
    # the container first: a `docker run` cut off by its timeout leaves it running with the volume
    # attached. Returns what stayed behind
    probe_left[:] = ["the probe cleanup"]  # stays if a call below is cut off
    leftovers = []
    removed = run_docker("rm", "-f", probe_container, timeout=CLEANUP_SECONDS)
    if removed.returncode and "no such container" not in removed.stderr.lower():
        leftovers.append("container " + probe_container)
    if run_docker("volume", "rm", "-f", probe_volume, timeout=CLEANUP_SECONDS).returncode:
        leftovers.append("volume " + probe_volume)
    probe_left[:] = leftovers
    return leftovers

def mount_probe_volume():
    # install and enable mount nothing, and a host may refuse what the plugin asks of its loop
    # device (direct I/O, 4096-byte sectors): one probe volume is mounted before the plugin is
    # handed out, else the older plugin keeps renting (the plugin is disabled only if it owns no volume).
    # 1g: the plugin's mkfs.xfs refuses anything under 300 MB; sparse, so a host with little
    # free disk is not refused by the reservation instead of by its loop device
    deadline = min(time.monotonic() + PROBE_SECONDS, validator_deadline - PROBE_RESERVE_SECONDS)
    if deadline <= time.monotonic():
        raise RuntimeError(TOO_LITTLE_TIME_TO_PROBE)
    leftovers = remove_probe_leftovers()
    if leftovers:
        raise RuntimeError("an earlier probe left " + ", ".join(leftovers))
    try:
        created = run_docker("volume", "create", "-d", alias, "-o", "size=1g", "-o", "sparse=true", probe_volume,
                             timeout=max(1, deadline - time.monotonic()))
        if created.returncode:
            raise RuntimeError("probe volume create failed" + stderr_of(created))
        mounted = run_docker("run", "--rm", "--name", probe_container, "--network", "none",
                             "-v", probe_volume + ":/mnt", probe_image, "sh", "-c", "echo probe > /mnt/probe && sync /mnt/probe",
                             timeout=max(1, deadline - time.monotonic()))
        if mounted.returncode:
            raise RuntimeError("probe volume mount failed" + stderr_of(mounted))
    except BaseException:
        remove_probe_leftovers()
        raise
    leftovers = remove_probe_leftovers()
    if leftovers:
        # a volume still in use means the mount did not end cleanly: not a proof
        raise RuntimeError("probe cleanup left " + ", ".join(leftovers))

def disable_plugin(plugin_id):
    # the backend's health check may hold a volume on the plugin for a moment: retry. Short tries that
    # stop before the validator's deadline, so a hung daemon still leaves this reason on stderr even
    # when the lock wait ate the probe's reserve. The pause before a retry counts against the deadline
    # like the try: a pause after the last try that fits would run past it and lose this reason
    for attempt in range(DISABLE_TRIES):
        pause = 1 if attempt else 0
        try_seconds = min(DISABLE_TRY_SECONDS, validator_deadline - START_SECONDS - time.monotonic() - pause)
        if try_seconds < 1:
            break
        time.sleep(pause)
        try:
            if run_docker("plugin", "disable", plugin_id, timeout=try_seconds).returncode == 0:
                return
        except DockerTimeout:
            pass
    raise RuntimeError("the plugin could not be disabled")

def volumes_on_plugin(plugin_id):
    # Docker hides the volumes of a disabled plugin and does not restore a plugin's refcounts after a
    # daemon restart, so its own refusal to disable a plugin in use cannot be relied on. Every regular
    # file in DATA_DIR is a volume (the plugin's List reads that folder), whether the plugin is enabled or
    # not; the daemon's listing names the plugin by alias or ID. The probe volume is setup's own
    global step
    step = "the listing of the plugin's data directory"
    try:
        names = {entry.name for entry in os.scandir(host_data_dir) if entry.is_file(follow_symlinks=False)}
    except FileNotFoundError:
        names = set()
    except OSError:
        raise RuntimeError("cannot list the plugin's data directory") from None
    listed = run_docker("volume", "ls", "--format", "{{.Name}} {{.Driver}}", timeout=CLEANUP_SECONDS)
    if listed.returncode:
        raise RuntimeError("cannot list the volumes" + stderr_of(listed))
    for line in listed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] in (alias, plugin_id, plugin_id[:12]):
            names.add(fields[0])
    names.discard(probe_volume)
    return names

def disable_unless_it_owns_volumes(plugin_id):
    # a disabled plugin hides its volumes, so a rental on it loses its disk at the next restart or create:
    # one that holds any stays enabled and is only not handed out (no marker) for new volumes
    try:
        owned = volumes_on_plugin(plugin_id)
    except RuntimeError as error:
        raise RuntimeError("kept enabled, " + str(error)) from None
    if owned:
        raise RuntimeError("kept enabled: the plugin owns " + str(len(owned)) + " volume(s)")
    if probe_left:
        raise RuntimeError("kept enabled: the probe left " + ", ".join(probe_left))
    disable_plugin(plugin_id)

def read_saved_config(plugin_id):
    global step
    if len(plugin_id) != 64 or any(c not in "0123456789abcdef" for c in plugin_id):
        raise RuntimeError("invalid plugin ID")
    step = "the read of the plugin's saved config"
    with open(plugins_dir + "/" + plugin_id + "/config.json") as source:
        return json.load(source)

def remove_plugin_of_another_build(plugin):
    # Docker does not reliably refuse to disable or remove a plugin its volumes use (volumes_on_plugin).
    # A probe a killed setup left on that build is reclaimed first: its cleanup otherwise runs only
    # after this replacement, and the host would never move on
    leftovers = remove_probe_leftovers()
    if leftovers:
        raise RuntimeError("an earlier probe left " + ", ".join(leftovers))
    if volumes_on_plugin(plugin["Id"]):
        raise RuntimeError("an earlier build of the plugin still owns volumes")
    if plugin["Enabled"]:
        disable_plugin(plugin["Id"])
    if call("rm", plugin["Id"]).returncode:
        raise RuntimeError("the earlier build of the plugin could not be removed")

def probe_proof():
    # what a passed probe proved: this build, this installation of it, mounting in this data directory
    result = call("inspect", "--format", INSTALLATION_FORMAT, alias)
    if result.returncode:
        raise RuntimeError("plugin inspect failed")
    return digest + " " + result.stdout.strip()

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
        # another setup on this host may hold the lock for most of the deadline: a SIGTERM here names the wait
        step = "the wait for the setup lock"
        fcntl.flock(lock, fcntl.LOCK_EX)
        step = "start"
        # the backend's health check compares the marker with the plugin's ID and settings but not with
        # the pinned digest, which it does not know: a marker of another build goes before anything
        # below can fail, so both sides see the same fallback, and one of another installation (the
        # plugin absent, removed and installed again, or set to another data directory) once inspect shows it
        if probe_image and read_probe_marker().split(" ")[0] != digest:
            remove_probe_marker()
        plugin = inspect()
        if probe_image and os.path.exists(marker_path) and (plugin is None or read_probe_marker() != probe_proof()):
            os.remove(marker_path)
        if plugin is not None and probe_image and read_saved_config(plugin["Id"]).get("Manifest") != digest:
            # an earlier pinned build under this alias (lium7 once lium8 is pinned) is replaced once
            # no volume uses it, else every setup refuses its manifest and the host never moves on
            remove_plugin_of_another_build(plugin)
            plugin = None
        if plugin is None:
            # The daemon may complete an install even if its client reports failure. Inspect
            # the actual result under the lock before deciding whether setup succeeded.
            installed = call("install", tag, "--disable", "--alias", alias, "--grant-all-permissions",
                             "DATA_DIR=" + data_dir, "STATE_DIR=" + state_dir)
            plugin = inspect()
            if plugin is None:
                raise RuntimeError("installation did not produce a plugin" + stderr_of(installed))
        plugin_id = plugin["Id"]
        saved = read_saved_config(plugin_id)
        if saved.get("Manifest") != digest or saved.get("plugin", {}).get("Id") != plugin_id:
            raise RuntimeError("downloaded manifest does not match the pinned digest")
        env = plugin.get("Settings", {}).get("Env", [])
        for name, value in (("DATA_DIR", data_dir), ("STATE_DIR", state_dir)):
            if [entry for entry in env if entry.startswith(name + "=")] != [name + "=" + value]:
                raise RuntimeError("data/state directories do not match this Docker data root")
        enabled = None
        if not plugin["Enabled"]:
            if probe_image and seconds_before_probe_reserve() <= 0:
                # enabled now, the plugin would have no time left to be probed or disabled again
                raise RuntimeError(TOO_LITTLE_TIME_TO_PROBE)
            enabled = call("enable", plugin_id)
            plugin = inspect()
        if not plugin or plugin["Id"] != plugin_id or plugin["Enabled"] is not True:
            raise RuntimeError("verified plugin remains disabled" + (stderr_of(enabled) if enabled else ""))
        verified_probe_proof = probe_proof() if probe_image else None
        if verified_probe_proof is not None and read_probe_marker() != verified_probe_proof:
            # also for a plugin found enabled: an enabled plugin without the marker was never proven
            # to mount (a probe cut short by the validator's deadline, one enabled by hand)
            try:
                mount_probe_volume()
            except BaseException as error:
                reason = str(error) if isinstance(error, RuntimeError) else "probe volume failed: " + type(error).__name__
                try:
                    disable_unless_it_owns_volumes(plugin_id)
                except RuntimeError as disable_error:
                    reason += "; " + str(disable_error)
                raise RuntimeError(reason) from None
            step = "the write of the probe marker"
            with open(marker_path, "w") as marker:
                marker.write(verified_probe_proof)
        print("verified plugin ready")
except Exception as error:
    # Do not echo a remote banner, registry credentials, or arbitrary filesystem contents.
    if isinstance(error, RuntimeError):
        print(str(error), file=sys.stderr)
    else:
        print("plugin verification failed: " + type(error).__name__, file=sys.stderr)
    sys.exit(1)
'''


def loopback_plugin_setup_command(docker_root_dir: str, alias: str, deadline_seconds: int) -> str:
    tag, digest, generation = LOOPBACK_PLUGIN_BUILDS[alias]
    # v2 is set up exactly as before; only the direct-I/O generation mounts a probe volume
    probe_image = "" if alias == LOOPBACK_PLUGIN_ALIAS else ALPINE_HELPER_IMAGE
    arguments = (
        "python3", "-c", _SETUP_SCRIPT, "/usr/bin/docker", alias,
        tag, digest, generation, probe_image, str(deadline_seconds), docker_root_dir.rstrip("/"),
        "/proc/1/root", f"/run/lock/lium-loopback-{generation}.lock",
        loopback_plugin_probe_marker_path(alias) or "",
    )
    return shlex.join(arguments)
