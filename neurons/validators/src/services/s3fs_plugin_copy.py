"""DAH-6466: the s3fs volume plugin comes from a copy on the node, so a Volume rental never waits on Docker Hub.

Each Volume gets its own plugin instance (DAH-2512: one credential pair per instance). `docker plugin install`
resolves the plugin on the registry every time, even when its files are already on the host, so every Volume
rental used to make a Docker Hub round trip, and a reset or TLS timeout on it failed the rental (15 Nov node
72d90b60, 18 Nov node 9881eb48). Now the node keeps one copy of the pinned build (its config and files),
made by validation or, on a node that has none yet, by the first Volume rental; every instance is
`docker plugin create` from that copy, which reads only local files.
"""

import shlex

# Pinned by tag and checked by manifest digest after the install: a digest reference must never reach
# `plugin install` (dockerd <25.0.3 can panic, see loopback_plugin_setup). v2.0.9 is the build `latest` has
# carried since Dec 2021; a retagged upstream is refused, never run.
S3FS_PLUGIN_TAG = "mochoa/s3fs-volume-plugin:v2.0.9"
S3FS_PLUGIN_MANIFEST = "sha256:c3a12ea5bc1f86eb86e8b4890cfbda19a0c45be4678bf9008421b85c8d11daf9"
# staging installs the build under this name, copies it and removes it again
S3FS_TEMPLATE_ALIAS = "lium-s3fs-template"
# an install on a slow registry path, a 150 MB copy and a create; the rent path waits for it on a node without the copy
S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS = 300

# Executed by the executor's Python against the host's dockerd. The executor has host PID access, so the
# host's Docker data root is under /proc/1/root (as in loopback_plugin_setup); the docker CLI reads
# `plugin create`'s directory from there.
_SCRIPT = r'''
import fcntl
import json
import os
import shutil
import subprocess
import sys

docker, host_root, tag, manifest, template, alias = sys.argv[1:]

def run(*args, timeout=120):
    return subprocess.run([docker, *args], capture_output=True, text=True, check=False, timeout=timeout)

def tail(text):
    return (text or "").strip()[-300:]

def stage(copy):
    # the node's one registry contact for this build: install it disabled under the template name, check
    # it is the pinned build, keep its config and files, drop the template
    installed = None
    if run("plugin", "inspect", template).returncode:
        installed = run("plugin", "install", tag, "--alias", template, "--grant-all-permissions", "--disable",
                        timeout=240)
    found = run("plugin", "inspect", "-f", "{{.Id}}", template)
    if found.returncode:
        raise RuntimeError("plugin install failed: " + tail(installed.stderr if installed else found.stderr))
    plugin_id = found.stdout.strip()
    if len(plugin_id) != 64 or any(c not in "0123456789abcdef" for c in plugin_id):
        raise RuntimeError("invalid plugin ID")
    plugin_dir = host_root + root + "/plugins/" + plugin_id
    with open(plugin_dir + "/config.json") as saved:
        if json.load(saved).get("Manifest") != manifest:
            run("plugin", "rm", "-f", template)
            raise RuntimeError("downloaded manifest does not match the pinned digest")
    config = run("plugin", "inspect", "-f", "{{json .Config}}", template)
    if config.returncode:
        raise RuntimeError("plugin inspect failed")
    staging = copy + ".tmp"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    with open(staging + "/config.json", "w") as out:
        out.write(config.stdout)
    if subprocess.run(["cp", "-a", plugin_dir + "/rootfs", staging + "/rootfs"], capture_output=True).returncode:
        raise RuntimeError("copying the plugin files failed")
    shutil.rmtree(copy, ignore_errors=True)
    os.rename(staging, copy)
    open(copy + "/ready", "w").close()
    run("plugin", "rm", "-f", template)

def create_instance(copy):
    if run("plugin", "inspect", alias).returncode == 0:
        print("plugin " + alias + " already exists")
        return
    instance = base + "/instance-" + alias
    shutil.rmtree(instance, ignore_errors=True)
    os.makedirs(instance)
    try:
        # the copy's files linked, not copied (same filesystem). The marker gives each instance files of its
        # own: dockerd refuses a second plugin made from byte-identical files ("content ... already exists")
        if subprocess.run(["cp", "-al", copy + "/rootfs", instance + "/rootfs"], capture_output=True).returncode:
            raise RuntimeError("linking the plugin files failed")
        shutil.copyfile(copy + "/config.json", instance + "/config.json")
        with open(instance + "/rootfs/.lium-instance", "w") as marker:
            marker.write(alias + "\n")
        created = run("plugin", "create", alias, instance, timeout=150)
        if created.returncode:
            raise RuntimeError("plugin create failed: " + tail(created.stderr))
    finally:
        shutil.rmtree(instance, ignore_errors=True)

try:
    root = run("info", "-f", "{{.DockerRootDir}}").stdout.strip()
    if not root.startswith("/") or ".." in root.split("/"):
        raise RuntimeError("invalid Docker data root")
    base = host_root + root + "/lium-s3fs-plugin"
    copy = base + "/" + manifest.split(":")[-1][:12]
    os.makedirs(base, exist_ok=True)
    with open(base + "/lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(copy + "/ready"):
            stage(copy)
    if alias:
        create_instance(copy)
    print("s3fs plugin copy ready")
except Exception as error:
    # never echo arbitrary host output beyond the docker error tails above
    print(str(error) if isinstance(error, RuntimeError) else "s3fs plugin copy failed: " + type(error).__name__,
          file=sys.stderr)
    sys.exit(1)
'''


def s3fs_plugin_command(alias: str = "", host_root: str = "/proc/1/root", docker: str = "/usr/bin/docker") -> str:
    """Make sure the node holds the pinned plugin copy; with an alias, also create that Volume's instance from it."""
    return shlex.join(
        ("python3", "-c", _SCRIPT, docker, host_root, S3FS_PLUGIN_TAG, S3FS_PLUGIN_MANIFEST, S3FS_TEMPLATE_ALIAS, alias)
    )
