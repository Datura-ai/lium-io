"""Prepare v2 without triggering old dockerd's digest-download panic."""

import shlex

LOOPBACK_PLUGIN_TAG = "daturaai/docker-volume-loopback:1.0.0-lium1"
LOOPBACK_PLUGIN_DIGEST = "sha256:99eacc306478ae5d03188474d07973078b66109077caa49618c57da211a71309"

# Executed by the executor's Python, not the validator's filesystem. Docker's plugin metadata
# is on the host; the privileged executor has host PID access and can read it via /proc/1/root.
# No digest reference may reach `plugin install`: dockerd <25.0.3 can panic even with --disable.
_SETUP_SCRIPT = r'''
import fcntl
import json
import os
import subprocess
import sys

docker, alias, tag, digest, root, host_root, lock_path = sys.argv[1:]
data_dir = "/srv" + root + "/vloopback-v2"
state_dir = "/srv/run/docker-volume-loopback-v2"
plugins_dir = host_root + root + "/plugins"

def call(*args):
    return subprocess.run([docker, "plugin", *args], capture_output=True, text=True, check=False)

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
            raise RuntimeError("v2 installation did not produce a plugin")
        plugin_id = plugin["Id"]
        if len(plugin_id) != 64 or any(c not in "0123456789abcdef" for c in plugin_id):
            raise RuntimeError("invalid plugin ID")
        with open(plugins_dir + "/" + plugin_id + "/config.json") as source:
            saved = json.load(source)
        if saved.get("Manifest") != digest or saved.get("plugin", {}).get("Id") != plugin_id:
            raise RuntimeError("v2 downloaded manifest does not match the pinned digest")
        env = plugin.get("Settings", {}).get("Env", [])
        for name, value in (("DATA_DIR", data_dir), ("STATE_DIR", state_dir)):
            if [entry for entry in env if entry.startswith(name + "=")] != [name + "=" + value]:
                raise RuntimeError("v2 data/state directories do not match this Docker data root")
        if not plugin["Enabled"]:
            call("enable", plugin_id)
            plugin = inspect()
        if not plugin or plugin["Id"] != plugin_id or plugin["Enabled"] is not True:
            raise RuntimeError("verified v2 plugin remains disabled")
        print("verified v2 ready")
except Exception as error:
    # Do not echo a remote banner, registry credentials, or arbitrary filesystem contents.
    if isinstance(error, RuntimeError):
        print(str(error), file=sys.stderr)
    else:
        print("v2 verification failed: " + type(error).__name__, file=sys.stderr)
    sys.exit(1)
'''


def loopback_plugin_setup_command(docker_root_dir: str, alias: str) -> str:
    arguments = (
        "python3", "-c", _SETUP_SCRIPT, "/usr/bin/docker", alias,
        LOOPBACK_PLUGIN_TAG, LOOPBACK_PLUGIN_DIGEST, docker_root_dir.rstrip("/"),
        "/proc/1/root", "/run/lock/lium-loopback-v2.lock",
    )
    return shlex.join(arguments)
