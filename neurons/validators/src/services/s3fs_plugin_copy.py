"""DAH-6466: the s3fs volume plugin comes from a copy on the node, so a Volume rental never waits on Docker Hub.

Each Volume gets its own plugin instance (DAH-2512: one credential pair per instance). `docker plugin install`
resolves the plugin on the registry every time, even when its files are already on the host, so every Volume
rental used to make a Docker Hub round trip, and a reset or TLS timeout on it failed the rental (15 Nov node
72d90b60, 18 Nov node 9881eb48). Now the node keeps one copy of the pinned build (its config and files),
made by validation or, on a node that has none yet, by the first Volume rental; every instance is
`docker plugin create` from that copy, which reads only local files.
"""

import shlex

# The pinned build, fetched by digest over the registry API: the manifest, the plugin config and every layer are
# checked against their sha256 as they arrive, so nothing but this build can become the copy. v2.0.9 is the build
# `latest` has carried since Dec 2021. Nothing here goes through `docker plugin install` or reads dockerd's own
# plugin store, whose layout is internal to dockerd (lium-io#2345 review).
S3FS_PLUGIN_REPOSITORY = "mochoa/s3fs-volume-plugin"
S3FS_PLUGIN_MANIFEST = "sha256:c3a12ea5bc1f86eb86e8b4890cfbda19a0c45be4678bf9008421b85c8d11daf9"
REGISTRY_URL = "https://registry-1.docker.io"
REGISTRY_TOKEN_URL = "https://auth.docker.io/token?service=registry.docker.io"
# a 57 MB download, a 150 MB unpack and a create; the rent path waits for it on a node without the copy
S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS = 300

# Executed by the executor's Python against the host's dockerd. The executor has host PID access, so the
# host's Docker data root is under /proc/1/root (as in loopback_plugin_setup); the docker CLI reads
# `plugin create`'s directory from there.
_SCRIPT = r'''
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

docker, host_root, registry, token_url, repository, manifest, alias = sys.argv[1:]

def run(*args, timeout=120):
    return subprocess.run([docker, *args], capture_output=True, text=True, check=False, timeout=timeout)

def tail(text):
    return (text or "").strip()[-300:]

def fetch(url, token=None, accept=None):
    request = urllib.request.Request(url)
    if token:
        # never sent on to the blob store a blob request is redirected to
        request.add_unredirected_header("Authorization", "Bearer " + token)
    if accept:
        request.add_header("Accept", accept)
    return urllib.request.urlopen(request, timeout=60)

def pinned(response, digest, out=None):
    # the body, checked against its digest; with `out`, streamed there instead of kept
    hashed, kept = hashlib.sha256(), []
    while chunk := response.read(1 << 20):
        hashed.update(chunk)
        out.write(chunk) if out else kept.append(chunk)
    if "sha256:" + hashed.hexdigest() != digest:
        raise RuntimeError("downloaded plugin does not match the pinned digest")
    return b"".join(kept)

def stage(copy):
    # the node's one registry contact for this build
    try:
        scope = "&scope=repository:" + repository + ":pull"
        token = json.load(fetch(token_url + scope)).get("token")
        blob = registry + "/v2/" + repository + "/blobs/"
        found = json.loads(pinned(fetch(registry + "/v2/" + repository + "/manifests/" + manifest, token,
                                        "application/vnd.docker.distribution.manifest.v2+json"), manifest))
        if found["config"]["mediaType"] != "application/vnd.docker.plugin.v1+json":
            raise RuntimeError("the pinned digest is not a plugin")
        config = pinned(fetch(blob + found["config"]["digest"], token), found["config"]["digest"])
        staging = copy + ".tmp"
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging + "/rootfs")
        for layer in found["layers"]:
            with open(staging + "/layer", "wb") as out:
                pinned(fetch(blob + layer["digest"], token), layer["digest"], out)
            unpacked = subprocess.run(["tar", "-xzf", staging + "/layer", "-C", staging + "/rootfs", "--numeric-owner"],
                                      capture_output=True, text=True)
            if unpacked.returncode:
                raise RuntimeError("unpacking the plugin files failed: " + tail(unpacked.stderr))
            os.remove(staging + "/layer")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        raise RuntimeError("plugin download failed: " + str(getattr(error, "reason", error))[:300])
    with open(staging + "/config.json", "wb") as out:
        out.write(config)
    shutil.rmtree(copy, ignore_errors=True)
    os.rename(staging, copy)
    open(copy + "/ready", "w").close()

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


def s3fs_plugin_command(
    alias: str = "",
    host_root: str = "/proc/1/root",
    docker: str = "/usr/bin/docker",
    registry: str = REGISTRY_URL,
    token_url: str = REGISTRY_TOKEN_URL,
    manifest: str = S3FS_PLUGIN_MANIFEST,
) -> str:
    """Make sure the node holds the pinned plugin copy; with an alias, also create that Volume's instance from it."""
    return shlex.join(
        ("python3", "-c", _SCRIPT, docker, host_root, registry, token_url, S3FS_PLUGIN_REPOSITORY, manifest, alias)
    )
