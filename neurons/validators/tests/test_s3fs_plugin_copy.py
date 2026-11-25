"""DAH-6466: a Volume's s3fs plugin instance is made from the node's local copy, never pulled at rent time.

18 Nov 18:25Z, node 9881eb48: the rent path's per-volume `docker plugin install` failed on its Docker Hub round trip
(connection reset through the host's registry path), and the retried install failed the same way.
The node script runs here for real against a fake `docker` that keeps its plugins in a state file.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import urllib.parse
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.s3fs_plugin_copy import (
    S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX,
    S3FS_PLUGIN_REPOSITORY,
    S3FS_PLUGIN_TAG,
    s3fs_plugin_command,
)
from services.task.checks import s3fs_plugin_copy as check_module
from services.task.runner import SSHCommandResult

DOCKER_ROOT = "/dockerroot"
UNREACHABLE = "http://127.0.0.1:9"
PINNED_DIFF_ID = "sha256:" + "d" * 64
CONFIG = b'{"Entrypoint":["/usr/bin/tini"],"Interface":{"Socket":"s3fs.sock"}}'

FAKE_DOCKER = r"""
import json, os, sys
state_path, root = os.environ["FAKE_STATE"], os.environ["FAKE_ROOT"]
state = json.load(open(state_path)) if os.path.exists(state_path) else {"plugins": {}, "calls": []}
args = sys.argv[1:]
state["calls"].append(args)
def done(code=0, out="", err=""):
    json.dump(state, open(state_path, "w"))
    sys.stdout.write(out); sys.stderr.write(err); sys.exit(code)
if args[:1] == ["info"]:
    done(out=json.dumps({
        "DockerRootDir": root,
        "HTTPProxy": os.environ.get("FAKE_PROXY", ""),
        "RegistryConfig": {"Mirrors": json.loads(os.environ.get("FAKE_MIRRORS", "[]"))},
    }))
if args[:2] == ["plugin", "inspect"]:
    if args[-1] not in state["plugins"]:
        done(1, err="Error: No such plugin: " + args[-1])
    plugin = state["plugins"][args[-1]]
    if "{{json .Config.Rootfs.DiffIds}}" in args:
        done(out=json.dumps(plugin["diff_ids"]) + "\n")
    if "[{{json .Config.Rootfs.DiffIds}},{{json .Config.Description}}]" in args:
        done(out=json.dumps([plugin["diff_ids"], plugin.get("config", {}).get("description", "")]) + "\n")
    done(out="[{}]")
if args[:2] == ["plugin", "install"]:
    if os.environ.get("FAKE_INSTALL_ERROR"):
        done(1, err=os.environ["FAKE_INSTALL_ERROR"])
    state["plugins"][args[args.index("--alias") + 1]] = {"installed": args[2], "diff_ids": [os.environ["FAKE_DIFF_ID"]]}
    done()
if args[:2] == ["plugin", "rm"]:
    state["plugins"].pop(args[-1], None)
    done()
if args[:2] == ["plugin", "create"]:
    name, directory = args[2], args[3]
    rootfs = directory + "/rootfs"
    # like dockerd, create recomputes the layer digest from the directory, so it never equals the registry's
    state["plugins"][name] = {
        "diff_ids": ["sha256:" + "c" * 64],
        "marker": open(rootfs + "/.lium-instance").read(),
        "tini": open(rootfs + "/usr/bin/tini").read(),
        "config": json.load(open(directory + "/config.json")),
    }
    done()
done(2, err="unexpected " + " ".join(args))
"""


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _layer(compressed: bool = True) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz" if compressed else "w") as tar:
        tini = tarfile.TarInfo("usr/bin/tini")
        tini.size, tini.mode = 4, 0o755
        tar.addfile(tini, io.BytesIO(b"tini"))
    return raw.getvalue()


class _Registry:
    """Docker Hub's token endpoint and registry API for one pinned plugin build, on localhost."""

    def __init__(self, compressed: bool = True):
        layer = _layer(compressed)
        self.manifest = json.dumps(
            {
                "schemaVersion": 2,
                "config": {"mediaType": "application/vnd.docker.plugin.v1+json", "digest": _digest(CONFIG)},
                "layers": [
                    {"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip", "digest": _digest(layer)}
                ],
            }
        ).encode()
        self.pinned = _digest(self.manifest)
        self.blobs = {_digest(CONFIG): CONFIG, _digest(layer): layer}
        self.requests: list[tuple[str, str]] = []
        self.served: list[str] = []
        registry = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                # a proxied request carries the absolute URL; serve it as the registry it names
                split = urllib.parse.urlsplit(self.path)
                path = split.path + ("?" + split.query if split.query else "")
                registry.requests.append((split.hostname, path))
                prefix = f"/v2/{S3FS_PLUGIN_REPOSITORY}/"
                if path.startswith("/token?") and "service=registry.test" in path:
                    body = b'{"token": "t"}'
                elif self.headers.get("Authorization") != "Bearer t":
                    self.send_response(401)
                    self.send_header(
                        "WWW-Authenticate",
                        f'Bearer realm="{registry.url}/token",service="registry.test",scope="repository:x:pull"',
                    )
                    self.end_headers()
                    return
                elif path == prefix + "manifests/" + registry.pinned:
                    body = registry.manifest
                elif path.startswith(prefix + "blobs/") and path.split("/")[-1] in registry.blobs:
                    body = registry.blobs[path.split("/")[-1]]
                else:
                    return self.send_error(404)
                registry.served.append(path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def downloads(self) -> int:
        return sum("/manifests/" in path for path in self.served)


@pytest.fixture
def registry(request):
    served = _Registry(getattr(request, "param", True))
    yield served
    served.server.shutdown()


@pytest.fixture
def node(tmp_path, registry):
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}")
    docker.chmod(0o755)
    host_root = tmp_path / "host"
    (host_root / DOCKER_ROOT.lstrip("/")).mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env |= {"FAKE_STATE": str(tmp_path / "state.json"), "FAKE_ROOT": DOCKER_ROOT, "FAKE_DIFF_ID": PINNED_DIFF_ID}

    def run(alias="", registry_url=None, **fake):
        command = s3fs_plugin_command(
            alias,
            host_root=str(host_root),
            docker=str(docker),
            registry=registry_url or registry.url,
            manifest=registry.pinned,
            diff_id=PINNED_DIFF_ID,
        )
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, env=env | fake)
        state = json.loads((tmp_path / "state.json").read_text())
        return result, state

    run.docker, run.env = docker, env
    run.base = host_root / DOCKER_ROOT.lstrip("/") / "lium-s3fs-plugin"
    run.copy = run.base / registry.pinned.split(":")[-1][:12]
    return run


def test_a_node_without_the_copy_downloads_once_and_every_volume_after_is_made_locally(node, registry):
    first, state = node("s3fs-celium-volume-one")
    second, state = node("s3fs-celium-volume-two")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert registry.downloads() == 1  # one registry contact for the node, by digest
    plugins = state["plugins"]
    assert set(plugins) == {"s3fs-celium-volume-one", "s3fs-celium-volume-two"}
    for alias in ("s3fs-celium-volume-one", "s3fs-celium-volume-two"):
        # each instance has the plugin's files plus its own marker, which keeps dockerd from refusing the second
        assert plugins[alias]["tini"] == "tini" and plugins[alias]["marker"] == alias + "\n"
        # create keeps the config's description, which tells the validator the instance is the pinned build
        description = S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX + registry.pinned
        assert plugins[alias]["config"] == json.loads(CONFIG) | {"description": description}
    assert (node.copy / "ready").exists() and not (node.copy / "rootfs" / ".lium-instance").exists()
    assert not [path for path in node.base.iterdir() if path.name.startswith("instance-") or path.name.endswith(".tmp")]
    # nothing is installed through dockerd and nothing of dockerd's own plugin store is read
    assert not any(call[:2] == ["plugin", "install"] for call in state["calls"])


def test_an_instance_made_from_the_copy_counts_as_the_pinned_build_on_the_rent_path(node, registry, monkeypatch):
    # review on #2345: the rent path read the instance's layer digest, which create recomputes, and so tore down
    # every instance the copy made; it reads what `docker plugin inspect` shows of the created instance
    from services import docker_service

    result, _ = node("s3fs-celium-volume-one")
    assert result.returncode == 0, result.stderr
    inspected = subprocess.run(
        [str(node.docker), "plugin", "inspect", "--format",
         "[{{json .Config.Rootfs.DiffIds}},{{json .Config.Description}}]", "s3fs-celium-volume-one"],
        capture_output=True, text=True, env=node.env,
    ).stdout
    pinned_description = S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX + registry.pinned
    monkeypatch.setattr(docker_service, "S3FS_PLUGIN_COPY_DESCRIPTION", pinned_description)
    assert docker_service._s3fs_plugin_is_current(inspected)
    other_description = S3FS_PLUGIN_COPY_DESCRIPTION_PREFIX + "sha256:0"
    monkeypatch.setattr(docker_service, "S3FS_PLUGIN_COPY_DESCRIPTION", other_description)
    assert not docker_service._s3fs_plugin_is_current(inspected)


@pytest.mark.parametrize("registry", [False], indirect=True)
def test_a_layer_stored_as_a_plain_tar_is_unpacked_too(node, registry):
    # the pinned build's layer is a plain tar though its manifest labels it gzip; dockerd accepts either, so must we
    result, state = node("s3fs-celium-volume-one")

    assert result.returncode == 0, result.stderr
    assert state["plugins"]["s3fs-celium-volume-one"]["tini"] == "tini"


def test_validation_stages_the_copy_and_the_rental_then_needs_no_registry(node, registry):
    staged, _ = node()
    registry.server.shutdown()
    registry.server.server_close()

    rented, state = node("s3fs-celium-volume-one")

    assert staged.returncode == 0 and rented.returncode == 0, rented.stderr
    assert registry.downloads() == 1
    assert "s3fs-celium-volume-one" in state["plugins"]


def test_staging_that_cannot_download_fails_with_the_reason(node):
    result, state = node(registry_url=UNREACHABLE)

    assert result.returncode == 1
    assert "plugin download failed" in result.stderr and "refused" in result.stderr.lower()
    assert not any(call[:2] == ["plugin", "create"] for call in state["calls"])
    assert not (node.copy / "ready").exists()


@pytest.mark.parametrize("tampered", ["manifest", "layer"])
def test_a_download_other_than_the_pinned_build_is_refused(node, registry, tampered):
    # served under the pinned digests, but no longer their bytes
    if tampered == "manifest":
        registry.manifest += b" "
    else:
        layer = next(digest for digest, body in registry.blobs.items() if body != CONFIG)
        registry.blobs[layer] += b"x"

    result, state = node()

    assert result.returncode == 1
    assert "does not match the pinned digest" in result.stderr
    assert state["plugins"] == {} and not (node.copy / "ready").exists()


def test_an_instance_that_exists_is_left_alone(node):
    node("s3fs-celium-volume-one")

    result, state = node("s3fs-celium-volume-one")

    assert result.returncode == 0 and "already exists" in result.stdout
    assert sum(call[:2] == ["plugin", "create"] for call in state["calls"]) == 1


def _result(exit_code: int, stderr: str = "") -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command="",
        command_id="c",
        exit_code=exit_code,
        stdout="",
        stderr=stderr,
        duration_ms=1,
        started_at=now,
        finished_at=now,
        success=exit_code == 0,
    )


class _Runner:
    def __init__(self, exit_code: int):
        self.exit_code, self.commands = exit_code, []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        return _result(self.exit_code, "" if self.exit_code == 0 else "plugin install failed: reset")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "reason", "wait"),
    [
        (0, "S3FS_PLUGIN_COPY_STAGED", check_module.RECHECK_STAGED_SECONDS),
        (1, "S3FS_PLUGIN_COPY_NOT_STAGED", check_module.RETRY_FAILED_SECONDS),
    ],
)
async def test_the_check_stages_an_idle_node_then_waits_and_never_fails_it(context_factory, exit_code, reason, wait):
    runner = _Runner(exit_code)
    ctx = context_factory(runner=runner)
    check = check_module.S3fsPluginCopyCheck()
    clock = SimpleNamespace(now=1000.0)
    clock.monotonic = lambda: clock.now

    with patch.object(check_module, "time", clock), patch.object(check_module, "rental_of_this_node", lambda ctx: None):
        first = await check.run(ctx)
        clock.now += wait - 1
        await check.run(ctx)
        clock.now += 1
        await check.run(ctx)

    assert first.passed and reason in str(first.event)
    assert runner.commands == [s3fs_plugin_command()] * 2


@pytest.mark.asyncio
async def test_the_check_leaves_a_rented_node_alone(context_factory):
    runner = _Runner(0)
    rented = SimpleNamespace(pods=[object()])

    with patch.object(check_module, "rental_of_this_node", lambda ctx: rented):
        result = await check_module.S3fsPluginCopyCheck().run(context_factory(runner=runner))

    assert result.passed and runner.commands == []


def test_the_check_is_a_completion_step_so_it_never_runs_before_the_verdict():
    from services.task.pipeline import Pipeline
    from services.task.pipeline_factory import PipelineFactory

    pipeline = Pipeline(PipelineFactory.build_checks(), sink=None)
    assert [type(c) for c in pipeline.on_completion].count(check_module.S3fsPluginCopyCheck) == 1
    assert not any(isinstance(c, check_module.S3fsPluginCopyCheck) for c in pipeline.checks)


@pytest.mark.asyncio
async def test_the_check_skips_when_the_task_has_less_than_its_bound_left(context_factory):
    import time

    from core.config import settings
    from services.task.checks.task_time import TASK_TIMEOUT_MARGIN_SECONDS

    runner = _Runner(0)
    needed = check_module.S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS + check_module.BUDGET_MARGIN_SECONDS
    elapsed = settings.JOB_TIME_OUT - TASK_TIMEOUT_MARGIN_SECONDS - needed + 5
    ctx = context_factory(runner=runner, started_at_monotonic=time.monotonic() - elapsed)
    check = check_module.S3fsPluginCopyCheck()

    with patch.object(check_module, "rental_of_this_node", lambda ctx: None):
        result = await check.run(ctx)
        assert result.passed and "not enough task time left" in str(result.event) and runner.commands == []
        # not a run: a later run with time left stages at once
        assert (await check.run(context_factory(runner=runner))).passed

    assert runner.commands == [s3fs_plugin_command()]


@pytest.mark.asyncio
async def test_the_check_skips_when_an_express_recheck_deadline_leaves_less_than_its_bound(context_factory):
    # lium-io#2345 review: an express recheck is cancelled at its own deadline, sooner than JOB_TIME_OUT
    import time

    runner = _Runner(0)
    needed = check_module.S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS + check_module.BUDGET_MARGIN_SECONDS
    ctx = context_factory(runner=runner, deadline_monotonic=time.monotonic() + needed - 5)

    with patch.object(check_module, "rental_of_this_node", lambda ctx: None):
        result = await check_module.S3fsPluginCopyCheck().run(ctx)

    assert result.passed and "not enough task time left" in str(result.event) and runner.commands == []


def test_the_nodes_configured_mirror_is_tried_first(node, registry):
    result, state = node(
        "s3fs-celium-volume-one", registry_url=UNREACHABLE, FAKE_MIRRORS=json.dumps([registry.url + "/"])
    )

    assert result.returncode == 0, result.stderr
    assert registry.downloads() == 1 and "s3fs-celium-volume-one" in state["plugins"]
    assert not any(call[:2] == ["plugin", "install"] for call in state["calls"])


def test_the_download_goes_through_dockerds_proxy(node, registry):
    # the registry name does not resolve: only the proxy dockerd is configured with can reach it
    result, state = node("s3fs-celium-volume-one", registry_url="http://registry.invalid", FAKE_PROXY=registry.url)

    assert result.returncode == 0, result.stderr
    assert ("registry.invalid", f"/v2/{S3FS_PLUGIN_REPOSITORY}/manifests/{registry.pinned}") in registry.requests
    assert "s3fs-celium-volume-one" in state["plugins"]


def test_a_rental_that_cannot_download_has_dockerd_install_the_pinned_build(node):
    result, state = node("s3fs-celium-volume-one", registry_url=UNREACHABLE)

    assert result.returncode == 0, result.stderr
    assert "installed through dockerd" in result.stdout
    assert state["plugins"]["s3fs-celium-volume-one"]["installed"] == S3FS_PLUGIN_TAG
    assert not (node.copy / "ready").exists()  # the next staging run still builds the copy


def test_a_build_dockerd_installs_that_is_not_the_pinned_one_is_removed(node):
    result, state = node("s3fs-celium-volume-one", registry_url=UNREACHABLE, FAKE_DIFF_ID="sha256:" + "e" * 64)

    assert result.returncode == 1
    assert "does not match the pinned digest" in result.stderr
    assert state["plugins"] == {}


def test_a_rental_fails_with_both_reasons_when_dockerd_cannot_install_either(node):
    result, state = node(
        "s3fs-celium-volume-one", registry_url=UNREACHABLE, FAKE_INSTALL_ERROR="connection reset by peer"
    )

    assert result.returncode == 1
    assert "plugin download failed" in result.stderr and "connection reset by peer" in result.stderr
    assert state["plugins"] == {}
