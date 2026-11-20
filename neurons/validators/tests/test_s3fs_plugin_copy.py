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
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.s3fs_plugin_copy import S3FS_PLUGIN_REPOSITORY, s3fs_plugin_command
from services.task.checks import s3fs_plugin_copy as check_module
from services.task.runner import SSHCommandResult

DOCKER_ROOT = "/dockerroot"
CONFIG = b'{"Entrypoint":["/usr/bin/tini"],"Interface":{"Socket":"s3fs.sock"}}'

FAKE_DOCKER = r'''
import json, os, sys
state_path, root = os.environ["FAKE_STATE"], os.environ["FAKE_ROOT"]
state = json.load(open(state_path)) if os.path.exists(state_path) else {"plugins": {}, "calls": []}
args = sys.argv[1:]
state["calls"].append(args)
def done(code=0, out="", err=""):
    json.dump(state, open(state_path, "w"))
    sys.stdout.write(out); sys.stderr.write(err); sys.exit(code)
if args[:1] == ["info"]:
    done(out=root + "\n")
if args[:2] == ["plugin", "inspect"]:
    if args[-1] not in state["plugins"]:
        done(1, err="Error: No such plugin: " + args[-1])
    done(out="[{}]")
if args[:2] == ["plugin", "create"]:
    name, directory = args[2], args[3]
    rootfs = directory + "/rootfs"
    state["plugins"][name] = {
        "marker": open(rootfs + "/.lium-instance").read(),
        "tini": open(rootfs + "/usr/bin/tini").read(),
        "config": json.load(open(directory + "/config.json")),
    }
    done()
done(2, err="unexpected " + " ".join(args))
'''


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _layer() -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        tini = tarfile.TarInfo("usr/bin/tini")
        tini.size, tini.mode = 4, 0o755
        tar.addfile(tini, io.BytesIO(b"tini"))
    return raw.getvalue()


class _Registry:
    """Docker Hub's token endpoint and registry API for one pinned plugin build, on localhost."""

    def __init__(self):
        layer = _layer()
        self.manifest = json.dumps({
            "schemaVersion": 2,
            "config": {"mediaType": "application/vnd.docker.plugin.v1+json", "digest": _digest(CONFIG)},
            "layers": [{"mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip", "digest": _digest(layer)}],
        }).encode()
        self.pinned = _digest(self.manifest)
        self.blobs = {_digest(CONFIG): CONFIG, _digest(layer): layer}
        self.requests: list[str] = []
        registry = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                registry.requests.append(self.path)
                prefix = f"/v2/{S3FS_PLUGIN_REPOSITORY}/"
                if self.path.startswith("/token"):
                    body = b'{"token": "t"}'
                elif self.headers.get("Authorization") != "Bearer t":
                    return self.send_error(401)
                elif self.path == prefix + "manifests/" + registry.pinned:
                    body = registry.manifest
                elif self.path.startswith(prefix + "blobs/") and self.path.split("/")[-1] in registry.blobs:
                    body = registry.blobs[self.path.split("/")[-1]]
                else:
                    return self.send_error(404)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def downloads(self) -> int:
        return sum("/manifests/" in path for path in self.requests)


@pytest.fixture
def registry():
    served = _Registry()
    yield served
    served.server.shutdown()


@pytest.fixture
def node(tmp_path, registry):
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}")
    docker.chmod(0o755)
    host_root = tmp_path / "host"
    (host_root / DOCKER_ROOT.lstrip("/")).mkdir(parents=True)
    env = {**os.environ, "FAKE_STATE": str(tmp_path / "state.json"), "FAKE_ROOT": DOCKER_ROOT}

    def run(alias="", registry_url=None):
        command = s3fs_plugin_command(
            alias,
            host_root=str(host_root),
            docker=str(docker),
            registry=registry_url or registry.url,
            token_url=(registry_url or registry.url) + "/token?service=registry.docker.io",
            manifest=registry.pinned,
        )
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, env=env)
        state = json.loads((tmp_path / "state.json").read_text())
        return result, state

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
        assert plugins[alias]["config"] == json.loads(CONFIG)
    assert (node.copy / "ready").exists() and not (node.copy / "rootfs" / ".lium-instance").exists()
    assert not [path for path in node.base.iterdir() if path.name.startswith("instance-") or path.name.endswith(".tmp")]
    # nothing is installed through dockerd and nothing of dockerd's own plugin store is read
    assert not any(call[:2] == ["plugin", "install"] for call in state["calls"])


def test_validation_stages_the_copy_and_the_rental_then_needs_no_registry(node, registry):
    staged, _ = node()
    registry.server.shutdown()
    registry.server.server_close()

    rented, state = node("s3fs-celium-volume-one")

    assert staged.returncode == 0 and rented.returncode == 0, rented.stderr
    assert registry.downloads() == 1
    assert "s3fs-celium-volume-one" in state["plugins"]


def test_a_registry_failure_on_a_node_without_the_copy_fails_with_the_reason(node):
    result, state = node("s3fs-celium-volume-one", registry_url="http://127.0.0.1:9")

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

    result, state = node("s3fs-celium-volume-one")

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
        command="", command_id="c", exit_code=exit_code, stdout="", stderr=stderr, duration_ms=1,
        started_at=now, finished_at=now, success=exit_code == 0,
    )


class _Runner:
    def __init__(self, exit_code: int):
        self.exit_code, self.commands = exit_code, []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        return _result(self.exit_code, "" if self.exit_code == 0 else "plugin install failed: reset")


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_code", "reason", "wait"), [(0, "S3FS_PLUGIN_COPY_STAGED", check_module.RECHECK_STAGED_SECONDS),
                                                          (1, "S3FS_PLUGIN_COPY_NOT_STAGED", check_module.RETRY_FAILED_SECONDS)])
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
