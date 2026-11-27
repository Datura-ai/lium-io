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
# checked against their sha256 as they arrive, so nothing but this build can become the copy. It is our build of the
# plugin (s3fs_volume_plugin/, published by .github/workflows/s3fs_volume_plugin.yml): the mochoa one carries s3fs 1.86,
# whose retried requests go out unsigned. A republish of that tag needs these three digests updated with it. Nothing
# here reads dockerd's own plugin store, whose layout is internal to dockerd (lium-io#2345 review).
S3FS_PLUGIN_REPOSITORY = "daturaai/compute-subnet-validator"
S3FS_PLUGIN_TAG = "daturaai/compute-subnet-validator:s3fs-volume-plugin"
S3FS_PLUGIN_MANIFEST = "sha256:6c66443497c0e8c4258462b5ef1eae47b6b1da63d41c5ea969260c8dc5617867"
# the uncompressed digest of its one layer (the config's rootfs.diff_ids), which `docker plugin inspect` shows for an
# instance installed by tag: how such an instance is checked to be the pinned build
S3FS_PLUGIN_DIFF_ID = "sha256:85fb3ced1a33e10ce889e49c26f25fc4df4ad3000fd8c55b9f192779ac305750"
# `docker plugin create` recomputes the layer digest from the directory it is given (the instance marker included), so
# an instance made from the copy is told by the description the copy writes into its config, which create keeps
S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX = "lium s3fs plugin copy of "
S3FS_PLUGIN_COPY_DESCRIPTION = S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX + S3FS_PLUGIN_MANIFEST
REGISTRY_URL = "https://registry-1.docker.io"
# a 54 MB download, its unpack and a create; the rent path waits for it on a node without the copy
S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS = 300

# Executed by the executor's Python against the host's dockerd. The executor has host PID access, so the
# host's Docker data root is under /proc/1/root (as in loopback_plugin_setup); the docker CLI reads
# `plugin create`'s directory from there.
_SCRIPT = r"""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

docker, host_root, registry, repository, manifest, diff_id, tag, alias, description_prefix = sys.argv[1:]
description = description_prefix + manifest

def run(*args, timeout=120):
    return subprocess.run([docker, *args], capture_output=True, text=True, check=False, timeout=timeout)

def tail(text):
    return (text or "").strip()[-300:]

def get(url, token=None, accept=None):
    request = urllib.request.Request(url)
    if token:
        # never sent on to the blob store a blob request is redirected to
        request.add_unredirected_header("Authorization", "Bearer " + token)
    if accept:
        request.add_header("Accept", accept)
    return urllib.request.urlopen(request, timeout=60)

def fetch(endpoint, path, tokens, accept=None):
    # the registry's own auth: a 401 names the token service (Docker Hub, or whatever a mirror uses)
    try:
        return get(endpoint + path, tokens.get(endpoint), accept)
    except urllib.error.HTTPError as error:
        challenge = error.headers.get("WWW-Authenticate") or ""
        if error.code != 401 or endpoint in tokens or not challenge.lower().startswith("bearer "):
            raise
        params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        query = urllib.parse.urlencode({k: params[k] for k in ("service", "scope") if k in params})
        answer = json.load(get(params["realm"] + ("&" if "?" in params["realm"] else "?") + query))
        tokens[endpoint] = answer.get("token") or answer.get("access_token")
        return get(endpoint + path, tokens[endpoint], accept)

def pinned(response, digest, out=None):
    # the body, checked against its digest; with `out`, streamed there instead of kept
    hashed, kept = hashlib.sha256(), []
    while chunk := response.read(1 << 20):
        hashed.update(chunk)
        out.write(chunk) if out else kept.append(chunk)
    if "sha256:" + hashed.hexdigest() != digest:
        raise RuntimeError("downloaded plugin does not match the pinned digest")
    return b"".join(kept)

def download(endpoint, staging):
    tokens = {}
    blob = "/v2/" + repository + "/blobs/"
    found = json.loads(pinned(fetch(endpoint, "/v2/" + repository + "/manifests/" + manifest, tokens,
                                    "application/vnd.docker.distribution.manifest.v2+json"), manifest))
    if found["config"]["mediaType"] != "application/vnd.docker.plugin.v1+json":
        raise RuntimeError("the pinned digest is not a plugin")
    config = pinned(fetch(endpoint, blob + found["config"]["digest"], tokens), found["config"]["digest"])
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging + "/rootfs")
    for layer in found["layers"]:
        with open(staging + "/layer", "wb") as out:
            pinned(fetch(endpoint, blob + layer["digest"], tokens), layer["digest"], out)
        # tar detects the compression: the pinned layer is stored as a plain tar though the manifest labels it gzip
        unpacked = subprocess.run(["tar", "-xf", staging + "/layer", "-C", staging + "/rootfs", "--numeric-owner"],
                                  capture_output=True, text=True)
        if unpacked.returncode:
            raise RuntimeError("unpacking the plugin files failed: " + tail(unpacked.stderr))
        os.remove(staging + "/layer")
    with open(staging + "/config.json", "wb") as out:
        out.write(config)

def stage(copy, endpoints):
    # the node's one registry contact for this build, on the node's own registry path: its configured mirrors
    # first, then the registry, through dockerd's proxy when one is set (main sets it up from `docker info`).
    # Every byte is checked against the pinned digests, so a mirror can serve the build but never change it.
    staging = copy + ".tmp"
    reasons = []
    for endpoint in endpoints:
        try:
            download(endpoint, staging)
            break
        except (RuntimeError, urllib.error.URLError, OSError, ValueError, KeyError) as error:
            host = urllib.parse.urlsplit(endpoint).hostname or "registry"
            reasons.append(host + ": " + str(getattr(error, "reason", error))[:200])
    else:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("plugin download failed: " + "; ".join(reasons))
    shutil.rmtree(copy, ignore_errors=True)
    os.rename(staging, copy)
    open(copy + "/ready", "w").close()

def install_instance(failure):
    # a rental on a node whose registry path only dockerd can use (a proxy whose password `docker info` masks,
    # a host-only route): dockerd installs the pinned tag for this Volume, as before the copy, and the instance
    # is kept only if its layer is the pinned build's
    installed = run("plugin", "install", tag, "--alias", alias, "--grant-all-permissions", "--disable", timeout=240)
    if installed.returncode:
        raise RuntimeError(failure + "; plugin install failed: " + tail(installed.stderr))
    layers = run("plugin", "inspect", "-f", "{{json .Config.Rootfs.DiffIds}}", alias)
    try:
        matches = layers.returncode == 0 and json.loads(layers.stdout) == [diff_id]
    except ValueError:
        matches = False
    if not matches:
        run("plugin", "rm", "-f", alias)
        raise RuntimeError(failure + "; the installed plugin does not match the pinned digest")

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
        with open(copy + "/config.json") as source:
            config = json.load(source)
        for key in [k for k in config if k.lower() == "description"]:
            del config[key]
        config["description"] = description
        with open(instance + "/config.json", "w") as out:
            json.dump(config, out)
        with open(instance + "/rootfs/.lium-instance", "w") as marker:
            marker.write(alias + "\n")
        created = run("plugin", "create", alias, instance, timeout=150)
        if created.returncode:
            raise RuntimeError("plugin create failed: " + tail(created.stderr))
    finally:
        shutil.rmtree(instance, ignore_errors=True)

try:
    info = json.loads(run("info", "--format", "{{json .}}").stdout or "{}")
    root = info.get("DockerRootDir") or ""
    if not root.startswith("/") or ".." in root.split("/"):
        raise RuntimeError("invalid Docker data root")
    # dockerd's own proxy settings for the download, exactly as `docker plugin install` would have used them
    for variable, key in (("https_proxy", "HTTPSProxy"), ("http_proxy", "HTTPProxy"), ("no_proxy", "NoProxy")):
        if info.get(key):
            os.environ[variable] = info[key]
    mirrors = [m.rstrip("/") for m in (info.get("RegistryConfig") or {}).get("Mirrors") or [] if m.startswith("http")]
    base = host_root + root + "/lium-s3fs-plugin"
    copy = base + "/" + manifest.split(":")[-1][:12]
    os.makedirs(base, exist_ok=True)
    if alias and run("plugin", "inspect", alias).returncode == 0:
        print("plugin " + alias + " already exists")
        sys.exit(0)
    try:
        with open(base + "/lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(copy + "/ready"):
                stage(copy, mirrors + [registry])
    except RuntimeError as failure:
        if not alias:
            raise
        install_instance(str(failure))
        print("s3fs plugin installed through dockerd: " + str(failure)[:300])
        sys.exit(0)
    if alias:
        create_instance(copy)
    print("s3fs plugin copy ready")
except Exception as error:
    # never echo arbitrary host output beyond the docker error tails above
    print(str(error) if isinstance(error, RuntimeError) else "s3fs plugin copy failed: " + type(error).__name__,
          file=sys.stderr)
    sys.exit(1)
"""


def s3fs_plugin_command(
    alias: str = "",
    host_root: str = "/proc/1/root",
    docker: str = "/usr/bin/docker",
    registry: str = REGISTRY_URL,
    manifest: str = S3FS_PLUGIN_MANIFEST,
    diff_id: str = S3FS_PLUGIN_DIFF_ID,
) -> str:
    """Make sure the node holds the pinned plugin copy; with an alias, also create that Volume's instance from it."""
    return shlex.join(
        (
            "python3",
            "-c",
            _SCRIPT,
            docker,
            host_root,
            registry,
            S3FS_PLUGIN_REPOSITORY,
            manifest,
            diff_id,
            S3FS_PLUGIN_TAG,
            alias,
            S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX,
        )
    )
